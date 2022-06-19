# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None
    pruning_scope: str = 'global'
    random_mask_seed: int = None

    _name = 'Hyperparameters for Unstructured Random Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _pruning_layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'
    _pruning_scope = 'A paramter that enables global pruning or layer-wise pruning'
    _random_mask_seed = 'The random seed for generating a random mask'

class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None, dataset_hparams: hparams.DatasetHparams = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()
        
        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

       # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}

        if pruning_hparams.pruning_scope == 'global':

            # Determine the number of weights that need to be pruned.
            number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
            number_of_weights_to_prune = np.ceil(
                pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)
            # create the random scores of weights
            random_scores = {k: np.random.RandomState(pruning_hparams.random_mask_seed).rand(*v.shape) for k, v in weights.items()}
            # Create a vector of all the unpruned weights in the model.
            random_vector = np.concatenate([v[current_mask[k] == 1] for k, v in random_scores.items()])
            threshold = np.sort(np.abs(random_vector))[number_of_weights_to_prune]

            new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                            for k, v in random_scores.items()})

        elif pruning_hparams.pruning_scope == 'layer':
            new_mask_dict = {}
            # create the random score of weights
            np.random.seed(pruning_hparams.random_mask_seed)
            random_scores = {k: np.random.RandomState(pruning_hparams.random_mask_seed).rand(*v.shape) for k, v in weights.items()}
            for k, v in weights.items():
                # Determine the number of weights that need to be pruned.
                number_of_remaining_weights = np.sum(current_mask[k])
                number_of_weights_to_prune = np.ceil(
                    pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)
                # Create a vector of all the unpruned weights in the particular layer.
                random_vector = random_scores[k][current_mask[k] == 1]
                threshold = np.sort(random_vector)[number_of_weights_to_prune]

                new_mask_dict[k] = np.where(random_scores[k] > threshold, current_mask[k], np.zeros_like(v))
            new_mask = Mask(new_mask_dict)

        else: 
            raise ValueError('No such pruning scope: {}'.format(pruning_hparams.pruning_scope))

        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
