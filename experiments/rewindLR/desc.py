# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import argparse
import copy
from dataclasses import dataclass, replace
import os
from typing import Union

from utils import arg_utils
from datasets import registry as datasets_registry
from foundations.desc import Desc
from foundations import hparams
from foundations.step import Step
import pruning.registry
from foundations.local import  Platform
@dataclass
class RewindingDesc(Desc):
    """The hyperparameters necessary to describe a pruning and rewinding backbone."""

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    pruning_hparams: hparams.PruningHparams
    pretrain_dataset_hparams: hparams.DatasetHparams = None
    pretrain_training_hparams: hparams.TrainingHparams = None

    @staticmethod
    def name_prefix(): return 'rewindLR'

    @staticmethod
    def _add_pretrain_argument(parser):
        help_text = \
            'Perform a pre-training phase prior to running the main process. Setting this argument will enable '\
            'arguments to control how the dataset and training during this pre-training phase. Rewinding '\
            'is a specific case of pre-training where rewinding uses the same dataset and training procedure '\
            'as the main training run.'
        parser.add_argument('--pretrain', action='store_true', help=help_text)

    @staticmethod
    def _add_rewinding_argument(parser):
        help_text = \
            'The number of steps for which to train the network before the learning rate rewinding process begins. This is ' \
            'the \'learning rate rewinding\' step as described in COMPARING REWINDING AND FINE-TUNING IN NEURAL NETWORK PRUNING, '\
            'which means pruned models use the final weight values from the end of main training and the learning rate schedule from ' \
            'the last pretraining epochs. Can be expressed as a number of epochs (\'160ep\') or a number  of iterations (\'50000it\'). '\
            'If this flag is present, no other pretraining arguments  may be set. Pretraining will be conducted using the same dataset '\
            ' and training hyperparameters as for the main training run. For the full range of pre-training options, use --pretrain.'
        parser.add_argument('--rewinding_steps', type=str, help=help_text)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'RewindingDesc' = None):
        # Add the rewinding/pretraining arguments.
        learning_rate_rewinding_steps = arg_utils.maybe_get_arg('learning_rate_rewinding_steps')
        pretrain = arg_utils.maybe_get_arg('pretrain', boolean_arg=True)

        if learning_rate_rewinding_steps is not None and pretrain: raise ValueError('Cannot set --learning_rate_rewinding_steps and --pretrain')
        pretraining_parser = parser.add_argument_group(
            'Rewinding/Pretraining Arguments', 'Arguments that control how the network is pre-trained')
        RewindingDesc._add_rewinding_argument(pretraining_parser)
        RewindingDesc._add_pretrain_argument(pretraining_parser)

        # Get the proper pruning hparams.

        pruning_strategy = arg_utils.maybe_get_arg('pruning_strategy')
        if defaults and not pruning_strategy: pruning_strategy = defaults.pruning_hparams.pruning_strategy
        if pruning_strategy:
            pruning_hparams = pruning.registry.get_pruning_hparams(pruning_strategy)
            if defaults and defaults.pruning_hparams.pruning_strategy == pruning_strategy:
                def_ph = defaults.pruning_hparams
            else:
                def_ph = None
        else:
            pruning_hparams = hparams.PruningHparams
            def_ph = None

        # Add the main arguments.
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        pruning_hparams.add_args(parser, defaults=def_ph if defaults else None)

        # Handle pretraining.
        if pretrain:
            if defaults: def_th = replace(defaults.training_hparams, training_steps='0ep')
            hparams.TrainingHparams.add_args(parser, defaults=def_th if defaults else None,
                                             name='Training Hyperparameters for Pretraining', prefix='pretrain')
            hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None,
                                            name='Dataset Hyperparameters for Pretraining', prefix='pretrain')

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> 'RewindingDesc':
        # Get the main arguments.
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        pruning_hparams = pruning.registry.get_pruning_hparams(args.pruning_strategy).create_from_args(args)

        # Create the desc.
        desc = cls(model_hparams, dataset_hparams, training_hparams, pruning_hparams)

        # Handle pretraining.
        if args.pretrain and not Step.str_is_zero(args.pretrain_training_steps):
            desc.pretrain_dataset_hparams = hparams.DatasetHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_dataset_hparams._name = 'Pretraining ' + desc.pretrain_dataset_hparams._name
            desc.pretrain_training_hparams = hparams.TrainingHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_training_hparams._name = 'Pretraining ' + desc.pretrain_training_hparams._name
        elif 'rewinding_steps' in args and args.rewinding_steps and not Step.str_is_zero(args.rewinding_steps):
            desc.pretrain_dataset_hparams = copy.deepcopy(dataset_hparams)
            desc.pretrain_dataset_hparams._name = 'Pretraining for Learning Rate Rewinding ' + desc.pretrain_dataset_hparams._name
            desc.pretrain_training_hparams = copy.deepcopy(training_hparams)
            desc.pretrain_training_hparams._name = 'Pretraining for Learning Rate Rewinding ' + desc.pretrain_training_hparams._name
            desc.pretrain_training_hparams.training_steps = args.rewinding_steps
            
        return desc

    def str_to_step(self, s: str, pretrain: bool = False) -> Step:
        dataset_hparams = self.pretrain_dataset_hparams if pretrain else self.dataset_hparams
        iterations_per_epoch = datasets_registry.iterations_per_epoch(dataset_hparams)
        return Step.from_str(s, iterations_per_epoch)

    @property
    def pretrain_end_step(self):
        return self.str_to_step(self.pretrain_training_hparams.training_steps, True)

    @property
    def train_start_step(self):
        if self.pretrain_training_hparams: return self.str_to_step(self.pretrain_training_hparams.training_steps)
        else: return self.str_to_step('0it')

    @property
    def train_end_step(self):
        return self.str_to_step(self.training_hparams.training_steps) if self.training_hparams._convergence_training_steps is None \
               else self.str_to_step(self.training_hparams._convergence_training_steps)

    @property
    def pretrain_outputs(self):
        return datasets_registry.num_classes(self.pretrain_dataset_hparams)

    @property
    def train_outputs(self):
        return datasets_registry.num_classes(self.dataset_hparams)

    def run_path(self, replicate: int, pruning_level: Union[str, int], experiment: str = 'main'):
        """The location where any run is stored."""

        if not isinstance(replicate, int) or replicate <= 0:
            raise ValueError('Bad replicate: {}'.format(replicate))

        return os.path.join(Platform().root, self.hashname,
                            f'replicate_{replicate}', f'level_{pruning_level}', experiment)

    @property
    def display(self):
        ls = [self.dataset_hparams.display, self.model_hparams.display,
              self.training_hparams.display, self.pruning_hparams.display]
        if self.pretrain_training_hparams:
            ls += [self.pretrain_dataset_hparams.display, self.pretrain_training_hparams.display]
        return '\n'.join(ls)
