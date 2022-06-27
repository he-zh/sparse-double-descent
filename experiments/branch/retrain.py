# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import copy
import os
from training import checkpointing, standard_callbacks
import datasets.registry
from foundations import hparams, paths
from foundations.step import Step
from experiments.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train


class Branch(base.Branch):
    def branch_function(
        self,
        retrain_d: hparams.DatasetHparams,
        retrain_t: hparams.TrainingHparams,
        start_at_step_zero: bool = False,
    ):
        
        evaluate_every_epoch: bool = True

        # Get the mask and model.
        m = models.registry.load(self.level_root, self.main_desc.train_start_step, self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                 self.main_desc.pruning_hparams.pruning_strategy)
        freeze_pruned_weights = self.main_desc.pruning_hparams.freeze_pruned_weights
        if freeze_pruned_weights == 'init':
            model_for_reset = copy.deepcopy(m)
        elif (freeze_pruned_weights == 'final' or freeze_pruned_weights == 'permuted') and self.level != 0:
            model_for_reset = models.registry.load(self.main_desc.run_path(self.replicate, self.level-1), self.main_desc.train_end_step,
                                                self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                                self.main_desc.pruning_hparams.pruning_strategy)
        else:
            model_for_reset = None
        
        m = PrunedModel(m, Mask.load(self.level_root), model_for_reset, freeze_pruned_weights)

        start_step = Step.from_iteration(0 if start_at_step_zero else self.main_desc.train_start_step.iteration,
                                         datasets.registry.iterations_per_epoch(retrain_d))

         # If the model file for the end of training already exists in this location, do not train.
        iterations_per_epoch = datasets.registry.iterations_per_epoch(retrain_d)
        end_step = Step.from_str(retrain_t.training_steps, iterations_per_epoch)
        if (models.registry.exists(self.branch_root, end_step) and
            os.path.exists(paths.logger(self.branch_root))): return

        train_loader = datasets.registry.get(retrain_d, train=True)
        test_loader = datasets.registry.get(retrain_d, train=False)
        test_eval_callback = standard_callbacks.create_eval_callback('test', test_loader, verbose=self.verbose)
        train_eval_callback = standard_callbacks.create_eval_callback('train', train_loader, verbose=self.verbose)

        # Basic checkpointing and state saving at the beginning and end.
        result = [
                  standard_callbacks.run_at_step(start_step, standard_callbacks.save_model),
                  standard_callbacks.run_at_step(end_step, standard_callbacks.save_model),
                  standard_callbacks.run_at_step(end_step, standard_callbacks.save_logger),
                  standard_callbacks.run_every_epoch(checkpointing.save_checkpoint_callback),
                  ]

        # Test every epoch if requested.
        if self.verbose: result.append(standard_callbacks.run_every_epoch(standard_callbacks.create_timekeeper_callback()))

        # Ensure that testing occurs at least at the beginning and end of training.
        if start_step.it != 0 or not evaluate_every_epoch: result = [standard_callbacks.run_at_step(start_step, test_eval_callback)] + result
        if end_step.it != 0 or not evaluate_every_epoch: result = [standard_callbacks.run_at_step(end_step, test_eval_callback)] + result

        # Do the same for the train set if requested.
        if evaluate_every_epoch: result = [standard_callbacks.run_every_epoch(train_eval_callback)] + result

        if start_step.it != 0 or not evaluate_every_epoch: result = [standard_callbacks.run_at_step(start_step, train_eval_callback)] + result
        
        if end_step.it != 0 or not evaluate_every_epoch: result = [standard_callbacks.run_at_step(end_step, train_eval_callback)] + result
            
        train.train(retrain_t, m, train_loader, self.branch_root, result, start_step=start_step)



    @staticmethod
    def description():
        return "Retrain the model with different hyperparameters."

    @staticmethod
    def name():
        return 'retrain'


