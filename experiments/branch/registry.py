# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

from experiments.branch.base import Branch
from experiments.branch import retrain, randomly_prune, randomly_reinitialize, oneshot_prune

registered_branches = {
    'randomly_prune': randomly_prune.Branch,
    'randomly_reinitialize': randomly_reinitialize.Branch,
    'retrain': retrain.Branch,
    'oneshot_prune': oneshot_prune.Branch
}


def get(branch_name: str) -> Branch:
    if branch_name not in registered_branches:
        raise ValueError('No such branch: {}'.format(branch_name))
    else:
        return registered_branches[branch_name]
