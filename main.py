# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import argparse
import sys
import numpy as np
from numpy import random
import os
import torch

from experiments import runner_registry
from foundations.local import Platform
from utils import arg_utils


def main():
    # The welcome message.
    welcome = '='*82 + '\nA Framework on Sparse Double Descent Based on open-lth\n' + '-'*82

    # Choose an initial command.
    helptext = welcome + "\nChoose a command to run:"
    startup_path = sys.argv[0].split('/')[-1]
    for name, runner in runner_registry.registered_runners.items():
        if name != 'branch':
            helptext += "\n    * {} {} [...] => {}".format(startup_path, name, runner.description())
        else:
            for _name, _runner in runner_registry.registered_runners.items():
                if _name == name: continue
                helptext += "\n    * {} {}_{} [...] => {}".format(startup_path, _name, name, runner.description())
    helptext += '\n' + '='*82

    runner_name = arg_utils.maybe_get_arg('subcommand', positional=True)
    if runner_name is None or runner_name.split('_')[-1] not in runner_registry.registered_runners:
        print(helptext)
        sys.exit(1)

    runner_name = runner_name.split('_')[-1]
    # Add the arguments for that command.
    usage = '\n' + welcome + '\n'
    usage += 'main.py {} [...] => {}'.format(runner_name, runner_registry.get(runner_name).description())
    usage += '\n' + '='*82 + '\n'

    parser = argparse.ArgumentParser(usage=usage, conflict_handler='resolve')
    parser.add_argument('subcommand')
    parser.add_argument('--display_output_location','-d', action='store_true',
                        help='Display the output location for this job.')

    parser.add_argument('--gpu', type = str, default='3', help='The GPU devices to run the job') 
    
    # Get the platform arguments.
    Platform.add_args(parser)

    # Add arguments for the various runners.
    runner_registry.get(runner_name).add_args(parser)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.fix_all_random_seeds:
        for key in args.__dict__.keys():
            if 'torch_seed' in key or 'data_order_seed' in key or 'transformation_seed' in key or 'random_mask_seed' in key:
                setattr(args, key, args.fix_all_random_seeds)
        # args.torch_seed = args.fix_all_random_seeds
        # args.data_order_seed = args.fix_all_random_seeds
        # args.transformation_seed = args.fix_all_random_seeds
        # if hasattr(args, 'random_mask_seed'): args.random_mask_seed = args.fix_all_random_seeds

    platform = Platform.create_from_args(args)
    torch_seed = platform.torch_seed
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.display_output_location:
        runner_registry.get(runner_name).create_from_args(args).display_output_location()
        sys.exit(0)

    experiment_runner =  runner_registry.get(runner_name).create_from_args(args)
    experiment_runner.run()


if __name__ == '__main__':
    main()
