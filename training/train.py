# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import copy
import os
import typing
import warnings

import torch

from datasets.base import DataLoader
import datasets.registry
from foundations import hparams
from foundations import paths
from foundations.step import Step
from models.base import Model, DataParallel
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training.checkpointing import restore_checkpoint
from training import optimizers
from training import standard_callbacks
from training.metric_logger import MetricLogger
from foundations.local import  Platform
try:
    import apex
    NO_APEX = False
except ImportError:
    NO_APEX = True


def train(
    training_hparams: hparams.TrainingHparams,
    model: Model,
    train_loader: DataLoader,
    output_location: str,
    callbacks: typing.List[typing.Callable] = [],
    start_step: Step = None,
    end_step: Step = None
):

    """The main training loop for this framework.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * output_location: The string path where all outputs should be stored.
      * callbacks: A list of functions that are called before each training step and once more
        after the last training step. Each function takes five arguments: the current step,
        the output location, the model, the optimizer, and the logger.
        Callbacks are used for running the test set, saving the logger, saving the state of the
        model, etc. The provide hooks into the training loop for customization so that the
        training loop itself can remain simple.
      * start_step: The step at which the training data and learning rate schedule should begin.
        Defaults to step 0.
      * end_step: The step at which training should cease. Otherwise, training will go for the
        full `training_hparams.training_steps` steps.
    """

    # Create the output location if it doesn't already exist.
    if not os.path.exists(output_location):
        os.makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    model.to(Platform().torch_device)
    optimizer = optimizers.get_optimizer(training_hparams, model)
    step_optimizer = optimizer
    lr_schedule = optimizers.get_lr_schedule(training_hparams, optimizer, train_loader.iterations_per_epoch)

    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model, step_optimizer = apex.amp.initialize(model, optimizer, loss_scale='dynamic', verbosity=0)

    # Handle parallelism if applicable.
    if Platform().is_parallel:
        model = DataParallel(model)

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step, cp_logger = restore_checkpoint(output_location, model, optimizer, train_loader.iterations_per_epoch)
    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()
    with warnings.catch_warnings():  # Filter unnecessary warning.
        warnings.filterwarnings("ignore", category=UserWarning)
        for _ in range(start_step.iteration): lr_schedule.step()

    # Determine when to end training.
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)

    if end_step <= start_step: return

    # Determine when to regain pruned weights
    if training_hparams.regain_pruned_weights_steps is not None:
        zeroing_interval = Step.from_str(training_hparams.regain_pruned_weights_steps,  train_loader.iterations_per_epoch)
        if zeroing_interval > end_step:
            zeroing_interval = end_step
    else:
        zeroing_interval = None

    # Get the initialization of model parameters if distance reguralization is applied
    if training_hparams.distance_penalty is not None:
        init_file_path = os.path.join(output_location, 'model_ep{}_it{}.pth'.format(start_step.ep, start_step.it))
        init_state = torch.load(init_file_path, map_location=Platform().torch_device) if os.path.exists(init_file_path) else None


    # The training loop.
    for ep in range(start_step.ep, end_step.ep + 1):

        # Ensure the data order is different for each epoch.
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))

        for it, (examples, labels) in enumerate(train_loader):

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model, optimizer, logger)

            # Use convergence criterion to decide when to stop
            if training_hparams.use_convergence_stopping_criterion:
                if ('train_loss', step.iteration) in logger.log:
                    train_loss = logger.log[('train_loss', step.iteration)]
                    if train_loss < 0.001: 
                        training_hparams._convergence_training_steps = str(ep)+'ep'+str(it)+'it'
                        model.save(output_location, step)
                        logger.save(output_location)
                        return
            
            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Allow pruned weights to be trainable and updated from 0 at zeroing intervals
            if zeroing_interval is not None and ep == zeroing_interval.ep and it == zeroing_interval.it:
                model = model.model
                model.to(torch.device('cpu'))
                model = PrunedModel(model, Mask.ones_like(model))
                model.to(Platform().torch_device)


            # Otherwise, train.
            examples = examples.to(device=Platform().torch_device)
            labels = labels.to(device=Platform().torch_device)

            step_optimizer.zero_grad()
            model.train()
            if training_hparams.distance_penalty is not None and init_state is not None:
                loss = model.loss_criterion(model(examples), labels) + 0.5*training_hparams.distance_penalty**2*optimizers.distance_reguralization(model, init_state)
            else: 
                loss = model.loss_criterion(model(examples), labels)

            if training_hparams.apex_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # add additional weight decay for BN weights if pruning applied to BN layers
            if training_hparams.bn_sparsity_regularization:
                model.updateBN()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            step_optimizer.step()
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step()


def standard_train(
  model: Model,
  output_location: str,
  dataset_hparams: hparams.DatasetHparams,
  training_hparams: hparams.TrainingHparams,
  start_step: Step = None,
  verbose: bool = True,
  evaluate_every_epoch: bool = True
):
    """Train using the standard callbacks according to the provided hparams."""

    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    if (models.registry.exists(output_location, train_end_step) and
        os.path.exists(paths.logger(output_location))): return

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    callbacks = standard_callbacks.standard_callbacks(
        training_hparams, train_loader, test_loader, start_step=start_step,
        verbose=verbose, evaluate_every_epoch=evaluate_every_epoch)
    train(training_hparams, model, train_loader, output_location, callbacks, start_step=start_step)
