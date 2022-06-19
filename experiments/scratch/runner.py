# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import argparse
import copy

from utils import shared_args
from dataclasses import dataclass
from foundations.runner import Runner
import models.registry
from experiments.scratch.desc import ScratchDesc
import pruning.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train


@dataclass
class ScratchRunner(Runner):
    replicate: int
    levels: int
    desc: ScratchDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return 'Run a pruning and retraining from re-initialized scratch experiment.'

    @staticmethod
    def _add_levels_argument(parser):
        help_text = \
            'The number of levels of iterative pruning to perform. At each level, the network is trained to ' \
            'completion, pruned, and retrained, preparing it for the next iteration. The full network is trained ' \
            'at level 0, and level 1 is the first level at which pruning occurs. Set this argument to 0 to ' \
            'just train the full network or to N to prune the network N times.'
        parser.add_argument('--levels', required=True, type=int, help=help_text)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # Get preliminary information.
        defaults = shared_args.maybe_get_default_hparams()

        # Add the job arguments.
        shared_args.JobArgs.add_args(parser)
        scratch_parser = parser.add_argument_group(
        'Scratch Hyperparameters', 'Hyperparameters that control the pruning and retraining from scratch process.')
        ScratchRunner._add_levels_argument(scratch_parser)
        ScratchDesc.add_args(parser, defaults)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'ScratchRunner':
        return ScratchRunner(args.replicate, args.levels, ScratchDesc.create_from_args(args),
                             not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.replicate, 0))

    def run(self) -> None:
        if self.verbose:
            print('='*82 + f'\nThe Pruning and Retraining from Scratch Experiment (Replicate {self.replicate})\n' + '-'*82)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.replicate, 0)}' + '\n' + '='*82 + '\n')

        self.desc.save(self.desc.run_path(self.replicate, 0))
        if self.desc.pretrain_training_hparams: self._pretrain()

        for level in range(self.levels+1):
            self._establish_initial_weights(level)
            self._prune_level(level)
            self._train_level(level)

    # Helper methods for running the pruning and rewinding process.
    def _pretrain(self):
        location = self.desc.run_path(self.replicate, 'pretrain')
        if models.registry.exists(location, self.desc.pretrain_end_step): return

        if self.verbose: print('-'*82 + '\nPretraining\n' + '-'*82)
        model = models.registry.get(self.desc.model_hparams, outputs=self.desc.pretrain_outputs,
                                    pruning_strategy = self.desc.pruning_hparams.pruning_strategy)
        train.standard_train(model, location, self.desc.pretrain_dataset_hparams, self.desc.pretrain_training_hparams,
                             verbose=self.verbose, evaluate_every_epoch=self.evaluate_every_epoch)

    def _establish_initial_weights(self, level):
        location = self.desc.run_path(self.replicate, level)
        if models.registry.exists(location, self.desc.train_start_step): return

        new_model = models.registry.get(self.desc.model_hparams, outputs=self.desc.train_outputs,
                                        pruning_strategy = self.desc.pruning_hparams.pruning_strategy)

        # If there was a pretrained model, retrieve its final weights and adapt them for training (only for level 0
        # models of other levels will be loaded with re-initialization).
        if self.desc.pretrain_training_hparams is not None and level == 0:
            pretrain_loc = self.desc.run_path(self.replicate, 'pretrain')
            old = models.registry.load(pretrain_loc, self.desc.pretrain_end_step,
                                       self.desc.model_hparams, self.desc.pretrain_outputs, 
                                       self.desc.pruning_hparams.pruning_strategy)
            state_dict = {k: v for k, v in old.state_dict().items()}

            # Select a new output layer if number of classes differs.
            if self.desc.train_outputs != self.desc.pretrain_outputs:
                state_dict.update({k: new_model.state_dict()[k] for k in new_model.output_layer_names})

            new_model.load_state_dict(state_dict)

        new_model.save(location, self.desc.train_start_step)

    def _train_level(self, level: int):
        location = self.desc.run_path(self.replicate, level)
        if models.registry.exists(location, self.desc.train_end_step): return
        # use the randomly re-initialized weights
        model = models.registry.load(location, self.desc.train_start_step,
                                     self.desc.model_hparams, self.desc.train_outputs, 
                                     self.desc.pruning_hparams.pruning_strategy)
                                     
        freeze_pruned_weights = self.desc.pruning_hparams.freeze_pruned_weights
        if freeze_pruned_weights == 'init' and level != 0:
            model_for_reset = copy.deepcopy(model)
        elif (freeze_pruned_weights == 'final' or freeze_pruned_weights == 'permuted') and level != 0:
            model_for_reset = models.registry.load(self.desc.run_path(self.replicate, level-1), self.desc.train_end_step,
                                                self.desc.model_hparams, self.desc.train_outputs, 
                                                self.desc.pruning_hparams.pruning_strategy)   
        else:
            model_for_reset = None
        
        pruned_model = PrunedModel(model, Mask.load(location), model_for_reset, freeze_pruned_weights)
        pruned_model.save(location, self.desc.train_start_step)
        if self.verbose:
            print('-'*82 + '\nPruning Level {}\n'.format(level) + '-'*82)
        train.standard_train(pruned_model, location, self.desc.dataset_hparams, self.desc.training_hparams,
                             start_step=self.desc.train_start_step, verbose=self.verbose,
                             evaluate_every_epoch=self.evaluate_every_epoch)

    def _prune_level(self, level: int):
        new_location = self.desc.run_path(self.replicate, level)
        if Mask.exists(new_location): return

        if level == 0:
            Mask.ones_like(models.registry.get(self.desc.model_hparams, self.desc.train_outputs,
                                               pruning_strategy = self.desc.pruning_hparams.pruning_strategy)).save(new_location)
        else:
            old_location = self.desc.run_path(self.replicate, level-1)
            model = models.registry.load(old_location, self.desc.train_end_step,
                                         self.desc.model_hparams, self.desc.train_outputs, 
                                         self.desc.pruning_hparams.pruning_strategy)
            pruning.registry.get(self.desc.pruning_hparams)(model, Mask.load(old_location), self.desc.dataset_hparams).save(new_location)
