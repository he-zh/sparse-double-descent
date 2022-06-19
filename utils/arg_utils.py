# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import argparse


def maybe_get_arg(arg_name, positional=False, position=0, boolean_arg=False):
    parser = argparse.ArgumentParser(add_help=False)
    prefix = '' if positional else '--'
    if positional:
        for i in range(position):
            parser.add_argument(f'arg{i}')
    if boolean_arg:
        parser.add_argument(prefix + arg_name, action='store_true')
    else:
        parser.add_argument(prefix + arg_name, type=str, default=None)
    try:
        args = parser.parse_known_args()[0]
        return getattr(args, arg_name) if arg_name in args else None
    except:
        return None
