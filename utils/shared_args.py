# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

from dataclasses import dataclass

from utils import arg_utils
from foundations.hparams import Hparams
import models.registry


@dataclass
class JobArgs(Hparams):
    """Arguments shared across lottery ticket jobs."""

    replicate: int = 1
    default_hparams: str = None
    quiet: bool = False
    evaluate_only_at_end: bool = False

    _name: str = 'High-Level Arguments'
    _description: str = 'Arguments that determine how the job is run and where it is stored.'
    _replicate: str = 'The index of this particular replicate. ' \
                      'Use a different replicate number to run another copy of the same experiment.'
    _default_hparams: str = 'Populate all arguments with the default hyperparameters for this model.'
    _quiet: str = 'Suppress output logging about the training status.'
    _evaluate_only_at_end: str = 'Run the test set only before and after training. Otherwise, will run every epoch.'


def maybe_get_default_hparams():
    default_hparams = arg_utils.maybe_get_arg('default_hparams')
    return models.registry.get_default_hparams(default_hparams) if default_hparams else None
