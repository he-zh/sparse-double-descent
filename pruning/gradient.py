# Copyright (c) Facebook, Inc. and its affiliates.

# https://github.com/JJGO/shrinkbench/blob/master/strategies/magnitude.py 

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import copy
import dataclasses
from foundations.local import Platform
import datasets
from training.train import train
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_scope: str = 'global'
    pruning_layers_to_ignore: str = None
    layers_to_prune: str = None

    _name = 'Hyperparameters for Unstructured Magnitude-Gradient Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _pruning_scope = 'A paramter that enables global pruning or layer-wise pruning, choose from global/layer'
    _pruning_layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'
    _layers_to_prune = 'Specify the layers that should be pruned, to prune first/last nth layers in all prunable layers, use first_n / last_n .'
    
class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None, dataset_hparams: hparams.DatasetHparams = None):
        
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()        
        
        model = copy.deepcopy(trained_model)
        
        prunable_tensors = set(model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in model.state_dict().items()
                   if k in prunable_tensors}
                   
        # Compute gradients of parameters
        train_loader = datasets.registry.get(dataset_hparams, train=True)
        train_loader.shuffle(None)
        examples, labels = next(iter(train_loader))
        model.zero_grad()
        model.train()
        loss = model.loss_criterion(model(examples), labels)
        loss.backward()
        # Get the model gradients.
        gradients = {k: v.grad.clone().cpu().detach().numpy()
                     for k, v in model.named_parameters()
                     if k in prunable_tensors and v.grad is not None}

        if pruning_hparams.pruning_scope == 'global':

            # Determine the number of weights that need to be pruned.
            number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
            number_of_weights_to_prune = np.ceil(
                pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

            # Compute importance scores of all the unpruned weights in the model, which is weight*gradient.
            importance_scores = {k: np.abs(v * gradients[k]) for k, v in weights.items()}
            importance_vector = np.concatenate([v[current_mask[k] == 1] for k, v in importance_scores.items()])
            threshold = np.sort(importance_vector)[number_of_weights_to_prune]

            new_mask = Mask({k: np.where(v > threshold, current_mask[k], np.zeros_like(v))
                            for k, v in importance_scores.items()})

        elif pruning_hparams.pruning_scope == 'layer':
            new_mask_dict = {}
            for k, v in weights.items():
                # Determine the number of weights that need to be pruned.
                number_of_remaining_weights = np.sum(current_mask[k])
                number_of_weights_to_prune = np.ceil(
                    pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

                # Compute importance scores of all the unpruned weights in the layer, which is weight*gradient.
                importance_scores = np.abs(v * gradients[k])
                importance_vector = importance_scores[current_mask[k] == 1]
                threshold = np.sort(importance_vector)[number_of_weights_to_prune]

                new_mask_dict[k] = np.where(np.abs(importance_scores[k]) > threshold, current_mask[k], np.zeros_like(v))
            new_mask = Mask(new_mask_dict)
        else: 
            raise ValueError('No such pruning scope: {}'.format(pruning_hparams.pruning_scope))

        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
    
