# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import argparse
from dataclasses import dataclass

from foundations import desc

from foundations.desc import Desc


def make_BranchDesc(BranchHparams: type, MainDesc: Desc, name: str):
    @dataclass
    class BranchDesc(desc.Desc):
        main_desc: MainDesc
        branch_hparams: BranchHparams

        @staticmethod
        def name_prefix(): return 'branch_' + name

        @staticmethod
        def add_args(parser: argparse.ArgumentParser, defaults: Desc = None):
            MainDesc.add_args(parser, defaults)
            BranchHparams.add_args(parser)

        @classmethod
        def create_from_args(cls, args: argparse.Namespace):
            return BranchDesc(MainDesc.create_from_args(args), BranchHparams.create_from_args(args))

    return BranchDesc
