# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

from foundations.runner import Runner
from training.runner import TrainingRunner
from experiments.rewindLR.runner import RewindingRunner
from experiments.lottery.runner import LotteryRunner
from experiments.finetune.runner import FinetuningRunner
from experiments.scratch.runner import ScratchRunner
from experiments.branch.runner import BranchRunner

registered_runners = {'train': TrainingRunner, 'rewindLR': RewindingRunner, 'lottery': LotteryRunner, 'finetune': FinetuningRunner, 
                      'scratch': ScratchRunner, 'branch': BranchRunner}


def get(runner_name: str) -> Runner:
    if runner_name not in registered_runners:
        raise ValueError('No such runner: {}'.format(runner_name))
    else:
        return registered_runners[runner_name]