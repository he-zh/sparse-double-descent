# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import argparse
from sys import platform
from dataclasses import dataclass
import os

from datasets import registry as datasets_registry
from foundations import desc
from foundations import hparams
from foundations.step import Step
# from lottery.desc import LotteryDesc
from foundations.local import  Platform

@dataclass
class TrainingDesc(desc.Desc):
    """The hyperparameters necessary to describe a training run."""

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams

    @staticmethod
    def name_prefix(): return 'train'

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'TrainingDesc' = None):
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingDesc':
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        return TrainingDesc(model_hparams, dataset_hparams, training_hparams)

    @property
    def end_step(self):
        iterations_per_epoch = datasets_registry.iterations_per_epoch(self.dataset_hparams)
        return Step.from_str(self.training_hparams.training_steps, iterations_per_epoch)

    @property
    def train_outputs(self):
        return datasets_registry.num_classes(self.dataset_hparams)

    def run_path(self, replicate, experiment='main'):
        return os.path.join(Platform().root, self.hashname, f'replicate_{replicate}', experiment)

    @property
    def display(self):
        return '\n'.join([self.dataset_hparams.display, self.model_hparams.display, self.training_hparams.display])
