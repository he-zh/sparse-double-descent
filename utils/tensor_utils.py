# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import random
from pruning.mask import Mask
from models.base import Model
import torch

import typing


def vectorize(state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a state dict into a single column Tensor in a repeatable way."""

    return torch.cat([state_dict[k].reshape(-1) for k in sorted(state_dict.keys())])


def unvectorize(vector: torch.Tensor, reference_state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a vector back into a state dict with the same shapes as reference state_dict."""

    if len(vector.shape) > 1: raise ValueError('vector has more than one dimension.')

    state_dict = {}
    for k in sorted(reference_state_dict.keys()):
        if vector.nelement() == 0: raise ValueError('Ran out of values.')

        size, shape = reference_state_dict[k].nelement(), reference_state_dict[k].shape
        this, vector = vector[:size], vector[size:]
        state_dict[k] = this.reshape(shape)

    if vector.nelement() > 0: raise ValueError('Excess values.')
    return state_dict


def perm(N, seed: int = None):
    """Generate a tensor with the numbers 0 through N-1 ordered randomly."""

    gen = torch.Generator()
    if seed is not None: gen.manual_seed(seed)
    perm = torch.normal(torch.zeros(N), torch.ones(N), generator=gen)
    return torch.argsort(perm)


def shuffle_tensor(tensor: torch.Tensor, seed: int = None):
    """Randomly shuffle the elements of a tensor."""

    shape = tensor.shape
    return tensor.reshape(-1)[perm(tensor.nelement(), seed=seed)].reshape(shape)


def shuffle_state_dict(state_dict: typing.Dict[str, torch.Tensor], seed: int = None):
    """Randomly shuffle each of the tensors in a state_dict."""

    output = {}
    for i, k in enumerate(sorted(state_dict.keys())):
        output[k] = shuffle_tensor(state_dict[k], seed=None if seed is None else seed+i)
    return output

def shuffle_model_params(model: Model, mask: Mask, seed: int = None):
    """
    Shuffle the pruned parameters in a model layerwise
    """
    for name, param in model.named_parameters():
        if name in model.prunable_layer_names:
            data_numpy = param.data.cpu().numpy()
            data_shuffle = data_numpy[mask[name]==0]
            random.Random(seed).shuffle(data_shuffle)
            data_numpy[mask[name]==0] = data_shuffle
    return model
