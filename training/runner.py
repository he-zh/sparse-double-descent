# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import argparse
from dataclasses import dataclass

from utils import shared_args
from foundations.runner import Runner
import models.registry
from training import train
from training.desc import TrainingDesc


@dataclass
class TrainingRunner(Runner):
    replicate: int
    desc: TrainingDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return "Train a model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        TrainingDesc.add_args(parser, shared_args.maybe_get_default_hparams())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return TrainingRunner(args.replicate, TrainingDesc.create_from_args(args),
                              not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.replicate))

    def run(self):
        if self.verbose:
            print('='*82 + f'\nTraining a Model (Replicate {self.replicate})\n' + '-'*82)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.replicate)}' + '\n' + '='*82 + '\n')
        self.desc.save(self.desc.run_path(self.replicate))
        train.standard_train(
            models.registry.get(self.desc.model_hparams, outputs=self.desc.train_outputs), self.desc.run_path(self.replicate),
            self.desc.dataset_hparams, self.desc.training_hparams, verbose=self.verbose, evaluate_every_epoch=self.evaluate_every_epoch)
