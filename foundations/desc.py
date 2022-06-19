# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import abc
import argparse
from dataclasses import dataclass, fields
import hashlib
import os
from foundations.hparams import Hparams
from foundations import paths


@dataclass
class Desc(abc.ABC):
    """The bundle of hyperparameters necessary for a particular kind of job. Contains many hparams objects.

    Each hparams object should be a field of this dataclass.
    """

    @staticmethod
    @abc.abstractmethod
    def name_prefix() -> str:
        """The name to prefix saved runs with."""

        pass

    @property
    def hashname(self) -> str:
        """The name under which experiments with these hyperparameters will be stored."""

        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [str(fields_dict[k]) for k in sorted(fields_dict) if isinstance(fields_dict[k], Hparams)]
        hash_str = hashlib.md5(';'.join(hparams_strs).encode('utf-8')).hexdigest()
        return f'{self.name_prefix()}_{hash_str}'

    @staticmethod
    @abc.abstractmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'Desc' = None) -> None:
        """Add the necessary command-line arguments."""

        pass

    @staticmethod
    @abc.abstractmethod
    def create_from_args(args: argparse.Namespace) -> 'Desc':
        """Create from command line arguments."""

        pass

    def save(self, output_location):
        if not os.path.exists(output_location): os.makedirs(output_location)
        if os.path.exists(paths.hparams(output_location)):
            return
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [fields_dict[k].display for k in sorted(fields_dict) if isinstance(fields_dict[k], Hparams)]
        with open(paths.hparams(output_location), 'w') as fp:
            fp.write('\n'.join(hparams_strs))
