# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import argparse
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
class FinetuningDesc(Desc):
    """The hyperparameters necessary to describe a pruning and finetuning backbone."""

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    pruning_hparams: hparams.PruningHparams
    finetuning_hparams: hparams.FinetuningHparams 
    pretrain_dataset_hparams: hparams.DatasetHparams = None
    pretrain_training_hparams: hparams.TrainingHparams = None

    @staticmethod
    def name_prefix(): return 'finetune'

    @staticmethod
    def _add_pretrain_argument(parser):
        help_text = \
            'Perform a pre-training phase prior to running the main pruning and finetuning process. Setting this argument '\
            'will enable arguments to control how the dataset and training during this pre-training phase. '
        parser.add_argument('--pretrain', action='store_true', help=help_text)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'FinetuningDesc' = None):

        # Add the finetuning/pretraining arguments.
        pretrain = arg_utils.maybe_get_arg('pretrain', boolean_arg=True)

        pretraining_parser = parser.add_argument_group(
            'Pretraining Arguments', 'Arguments that control how the network is pre-trained')
        FinetuningDesc._add_pretrain_argument(pretraining_parser)

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
        hparams.FinetuningHparams.add_args(parser, defaults=defaults.finetuning_hparams if defaults else None, prefix='finetune')
        pruning_hparams.add_args(parser, defaults=def_ph if defaults else None)
        # Set the finetuning arguments
        # def_ft = replace(defaults.training_hparams, **defaults.finetuning_hparams.__dict__)
        # hparams.TrainingHparams.add_args(parser, defaults=def_ft if defaults else None, prefix='finetune')

        # Handle pretraining.
        if pretrain:
            if defaults: def_th = replace(defaults.training_hparams, training_steps='0ep')
            hparams.TrainingHparams.add_args(parser, defaults=def_th if defaults else None,
                                             name='Training Hyperparameters for Pretraining', prefix='pretrain')
            hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None,
                                            name='Dataset Hyperparameters for Pretraining', prefix='pretrain')

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> 'FinetuningDesc':
        # Get the main arguments.
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        pruning_hparams = pruning.registry.get_pruning_hparams(args.pruning_strategy).create_from_args(args)
        ft_hparams = hparams.FinetuningHparams.create_from_args(args, prefix='finetune')
        finetuning_hparams = replace(training_hparams, **ft_hparams.__dict__)

        # Create the desc.
        desc = cls(model_hparams, dataset_hparams, training_hparams, pruning_hparams, finetuning_hparams)

        # Handle pretraining.
        if args.pretrain and not Step.str_is_zero(args.pretrain_training_steps):
            desc.pretrain_dataset_hparams = hparams.DatasetHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_dataset_hparams._name = 'Pretraining ' + desc.pretrain_dataset_hparams._name
            desc.pretrain_training_hparams = hparams.TrainingHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_training_hparams._name = 'Pretraining ' + desc.pretrain_training_hparams._name

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
    def finetune_start_step(self):
        return self.str_to_step('0it')
 
    @property
    def finetune_end_step(self):
        return self.str_to_step(self.finetuning_hparams.training_steps) if self.finetuning_hparams._convergence_training_steps is None \
               else self.str_to_step(self.finetuning_hparams._convergence_training_steps)

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
              self.training_hparams.display, self.pruning_hparams.display,
              self.finetuning_hparams.display]
        if self.pretrain_training_hparams:
            ls += [self.pretrain_dataset_hparams.display, self.pretrain_training_hparams.display]
        return '\n'.join(ls)
