# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

from dataclasses import dataclass
import foundations
import os
import torch
import pathlib

from foundations.hparams import Hparams

@dataclass
class Platform(Hparams):
    fix_all_random_seeds: int = None
    torch_seed: int = None

    _name: str = 'Platform Hyperparameters'
    _description: str = 'Hyperparameters that control the plaform on which the job is run.'
    _fix_all_random_seeds: int = 'The random seed to control cpu, gpu, data loader and random mask, this will make reproducibility possible'
    _torch_seed: str = 'The pytorch random seed that controls the randomness for cpu and cuda, like model initialization'

    # Manage the available devices.

    @property
    def device_str(self):
        # GPU device.
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 'cuda'
        # CPU device.
        else:
            return 'cpu'
    @property
    def device_ids(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_ids = [int(x) for x in range(torch.cuda.device_count())]
            return device_ids
        else: return None

    @property
    def torch_device(self):
        return torch.device(self.device_str)

    @property
    def is_parallel(self):
        return torch.cuda.is_available() and torch.cuda.device_count() > 1

    # important root for datasets and stored files

    @property
    def root(self):
        return os.path.join(pathlib.Path.home(), '/data/hezheng/result')

    @property
    def dataset_root(self):
        return os.path.join(pathlib.Path.home(), '/data/hezheng/datasets/')

    @property
    def tiny_imagenet_root(self):
        return os.path.join(pathlib.Path.home(), '/data/hezheng/datasets/tiny-imagenet-200')
