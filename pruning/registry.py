# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import copy
from functools import partial

from foundations.hparams import PruningHparams
from pruning import magnitude, network_slimming, random, gradient

registered_strategies = {'magnitude': magnitude.Strategy, 'random': random.Strategy, 'gradient': gradient.Strategy,
                         'network_slimming': network_slimming.Strategy}


def get(pruning_hparams: PruningHparams):
    """Get the pruning function."""

    return partial(registered_strategies[pruning_hparams.pruning_strategy].prune,
                   copy.deepcopy(pruning_hparams))


def get_pruning_hparams(pruning_strategy: str) -> type:
    """Get a complete lottery schema as specialized for a particular pruning strategy."""

    if pruning_strategy not in registered_strategies:
        raise ValueError('Pruning strategy {} not found.'.format(pruning_strategy))

    return registered_strategies[pruning_strategy].get_pruning_hparams()
