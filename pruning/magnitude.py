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
    pruning_scope: str = 'global'
    pruning_layers_to_ignore: str = None
    layers_to_prune: str = None
    prune_max_magnitude: bool = False

    _name = 'Hyperparameters for Unstructured Magnitude Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _pruning_scope = 'A paramter that enables global pruning or layer-wise pruning, choose from global/layer'
    _pruning_layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'
    _layers_to_prune = 'Specify the layers that should be pruned, to prune first/last nth layers in all prunable layers, use first_n / last_n .'
    _prune_max_magnitude = 'An order that control pruner to prune the weights with max magnitude, or min magnitude'
    
class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None, dataset_hparams: hparams.DatasetHparams = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine which layers can be pruned.
        def get_pruning_layers_sequence(layers_to_prune) -> int:
            if len(layers_to_prune.split('_'))==2 and layers_to_prune.split('_')[-1].isdigit() and int(layers_to_prune.split('_')[-1])>0:
                return int(layers_to_prune.split('_')[-1])
            else:
                raise ValueError('unrecognized pruning hparameters: {}'.format(layers_to_prune))

        if pruning_hparams.layers_to_prune is not None and pruning_hparams.pruning_scope == 'layer':
            if pruning_hparams.layers_to_prune.startswith('first'):
                pruning_layers_sequence = get_pruning_layers_sequence(pruning_hparams.layers_to_prune)
                prunable_tensors = set(trained_model.prunable_layer_names[:pruning_layers_sequence])
            elif pruning_hparams.layers_to_prune.startswith('last'):
                pruning_layers_sequence = get_pruning_layers_sequence(pruning_hparams.layers_to_prune)
                prunable_tensors = set(trained_model.prunable_layer_names[-pruning_layers_sequence:])  
            else: raise ValueError('unrecognized pruning hparameters: {}'.format(pruning_hparams.layers_to_prune))
        elif pruning_hparams.layers_to_prune is not None and pruning_hparams.pruning_scope != 'layer':
            raise ValueError('pruning hparameters: layers_to_prune={} should be associated with pruning_cope=layer'.format(pruning_hparams.layers_to_prune))
        else:
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

            # Create a vector of all the unpruned weights in the model.
            weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
            if pruning_hparams.prune_max_magnitude:
                abs_weight_vector = np.flip(np.sort(np.abs(weight_vector)))       
            else:
                abs_weight_vector = np.sort(np.abs(weight_vector))  

            threshold = abs_weight_vector[number_of_weights_to_prune]

            new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                            for k, v in weights.items()})

        elif pruning_hparams.pruning_scope == 'layer':
            new_mask_dict = {}
            for k, v in weights.items():
                # Determine the number of weights that need to be pruned.
                number_of_remaining_weights = np.sum(current_mask[k])
                number_of_weights_to_prune = np.ceil(
                    pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

                # Create a vector of all the unpruned weights in the particular layer.
                weight_vector = v[current_mask[k] == 1]
                if pruning_hparams.prune_max_magnitude:
                    abs_weight_vector = np.flip(np.sort(np.abs(weight_vector)))                    
                else:
                    abs_weight_vector = np.sort(np.abs(weight_vector)) 
                threshold = abs_weight_vector[number_of_weights_to_prune]

                new_mask_dict[k] = np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
            new_mask = Mask(new_mask_dict)
        else: 
            raise ValueError('No such pruning scope: {}'.format(pruning_hparams.pruning_scope))

        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
