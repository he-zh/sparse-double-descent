# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import abc
import argparse
import inspect
from dataclasses import dataclass, field, fields, make_dataclass
import sys
from typing import List

from experiments.branch.desc import make_BranchDesc

from experiments.finetune.desc import FinetuningDesc
from experiments.lottery.desc import LotteryDesc
from experiments.rewindLR.desc import RewindingDesc
from experiments.scratch.desc import ScratchDesc
from foundations.desc import Desc
from foundations.hparams import Hparams
from foundations.runner import Runner
from utils import arg_utils, shared_args

main_descs = {'finetune': FinetuningDesc, 'lottery': LotteryDesc, 'rewindLR': RewindingDesc, 'scratch': ScratchDesc}

@dataclass
class Branch(Runner):
    """A branch. Implement `branch_function`, add a name and description, and add to the registry."""
    replicate: int
    levels: str
    desc: Desc
    verbose: bool = False
    level: int = None

    # Interface that needs to be overriden for each branch.
    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """A description of this branch. Override this."""
        pass

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """The name of this branch. Override this."""
        pass

    @abc.abstractmethod
    def branch_function(self) -> None:
        """The method that is called to execute the branch.

        Override this method with any additional arguments that the branch will need.
        These arguments will be converted into command-line arguments for the branch.
        Each argument MUST have a type annotation. The first argument must still be self.
        """
        pass

    # Interface that is useful for writing branches.
    @property
    def main_experiment(self) -> str:
        """The main experiments on which the branch is based"""
        
        main_experiment = sys.argv[1].split('_')[0]
        return main_experiment

    @property
    def main_desc(self) -> Desc:
        """The main description of this experiment."""

        return self.desc.main_desc

    @property
    def experiment_name(self) -> str:
        """The name of this experiment."""

        return self.desc.hashname

    @property
    def branch_root(self) -> str:
        """The root for where branch results will be stored for a specific invocation of run()."""

        return self.main_desc.run_path(self.replicate, self.level, self.experiment_name)

    @property
    def zero_branch_root(self) -> str:
        """The level_0 folder root of the main experiment."""

        return self.main_desc.run_path(self.replicate, 0, self.experiment_name)

    @property
    def level_root(self) -> str:
        """The root of the main experiment on which this branch is based."""

        return self.main_desc.run_path(self.replicate, self.level)

    # Interface that deals with command line arguments.
    @dataclass
    class ArgHparams(Hparams):
        levels: str
        pretrain_training_steps: str = None

        _name: str = 'Experiments Hyperparameters'
        _description: str = 'Hyperparameters that control the pruning and retraining process.'
        _levels: str = \
            'The pruning levels on which to run this branch. Can include a comma-separate list of levels or ranges, '\
            'e.g., 1,2-4,9'
        _pretrain_training_steps: str = 'The number of steps to train the network prior to the branch process.'

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        defaults = shared_args.maybe_get_default_hparams()
        shared_args.JobArgs.add_args(parser)
        Branch.ArgHparams.add_args(parser)
        cls.BranchDesc.add_args(parser, defaults)

    @staticmethod
    def level_str_to_int_list(levels: str):
        level_list = []
        elements = levels.split(',')
        for element in elements:
            if element.isdigit():
                level_list.append(int(element))
            elif len(element.split('-')) == 2:
                level_list += list(range(int(element.split('-')[0]), int(element.split('-')[1]) + 1))
            else:
                raise ValueError(f'Invalid level: {element}')
        return sorted(list(set(level_list)))

    @classmethod
    def create_from_args(cls, args: argparse.Namespace):
        levels = Branch.level_str_to_int_list(args.levels)

        return cls(args.replicate, levels, cls.BranchDesc.create_from_args(args), not args.quiet)

    @classmethod
    def create_from_hparams(cls, replicate, levels: List[int], desc: Desc, hparams: Hparams, verbose=False):
        return cls(replicate, levels, cls.BranchDesc(desc, hparams), verbose)

    def display_output_location(self):
        print(self.branch_root)

    def run(self):
        for self.level in self.levels:
            if self.verbose:
                print('='*82)
                print(f'Branch {self.name()} (Replicate {self.replicate}, Level {self.level})\n' + '-'*82)
                print(f'{self.main_desc.display}\n{self.desc.branch_hparams.display}')
                print(f'Output Location: {self.branch_root}\n' + '='*82 + '\n')

            args = {f.name: getattr(self.desc.branch_hparams, f.name)
                    for f in fields(self.BranchHparams) if not f.name.startswith('_')}
            self.branch_function(**args)

    # Initialize instances and subclasses (metaprogramming).
    def __init_subclass__(cls):
        """Metaprogramming: modify the attributes of the subclass based on information in run().

        The goal is to make it possible for users to simply write a single run() method and have
        as much functionality as possible occur automatically. Specifically, this function converts
        the annotations and defaults in run() into a `BranchHparams` property.
        """

        fields = []
        for arg_name, parameter in list(inspect.signature(cls.branch_function).parameters.items())[1:]:
            t = parameter.annotation
            if t == inspect._empty: raise ValueError(f'Argument {arg_name} needs a type annotation.')
            elif t in [str, float, int, bool] or (isinstance(t, type) and issubclass(t, Hparams)):
                if parameter.default != inspect._empty: fields.append((arg_name, t, field(default=parameter.default)))
                else: fields.append((arg_name, t))
            else:
                raise ValueError('Invalid branch type: {}'.format(parameter.annotation))
        
        main_experiment = sys.argv[1].split('_')[0] if len(sys.argv) > 1 else None
        if main_experiment is not None and len(sys.argv[1].split('_'))==2:
            if main_experiment not in main_descs.keys(): 
                raise ValueError('{} has not been registered as a main experiment'.format(main_experiment))

            fields += [('_name', str, 'Branch Arguments'), ('_description', str, 'Arguments specific to the branch.')]
            setattr(cls, 'BranchHparams', make_dataclass('BranchHparams', fields, bases=(Hparams,)))
            setattr(cls, 'BranchDesc', make_BranchDesc(cls.BranchHparams, main_descs[main_experiment], cls.name()))
