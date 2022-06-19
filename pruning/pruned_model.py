# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

from utils.tensor_utils import shuffle_model_params
import torch
from models.base import Model
from pruning.mask import Mask

import numpy as np


class PrunedModel(Model):
    @staticmethod
    def to_mask_name(name):
        return 'mask_' + name.replace('.', '___')

    def __init__(self, model: Model, mask: Mask, model_for_reset: Model = None, freeze_pruned_weights: str = 'zero'):
        if isinstance(model, PrunedModel): raise ValueError('Cannot nest pruned models.')
        super(PrunedModel, self).__init__()
        self.model = model
        self.model_for_reset = shuffle_model_params(model_for_reset, mask, seed=0) if freeze_pruned_weights=='permuted' and \
        model_for_reset is not None else model_for_reset
        self.freeze_type = freeze_pruned_weights

        for k in self.model.prunable_layer_names:
            if k not in mask: raise ValueError('Missing mask value {}.'.format(k))
            if not np.array_equal(mask[k].shape, np.array(self.model.state_dict()[k].shape)):
                raise ValueError('Incorrect mask shape {} for tensor {}.'.format(mask[k].shape, k))

        for k in mask:
            if k not in self.model.prunable_layer_names:
                raise ValueError('Key {} found in mask but is not a valid model tensor.'.format(k))

        for k, v in mask.items(): self.register_buffer(PrunedModel.to_mask_name(k), v.float())
        self._apply_mask() # reset the parameters 

    def _apply_mask(self):
        for name, param in self.model.named_parameters():
            if hasattr(self, PrunedModel.to_mask_name(name)):
                if self.freeze_type == 'zero'  or self.model_for_reset is None:
                    param.data *= getattr(self, PrunedModel.to_mask_name(name))
                elif self.freeze_type == 'init' or self.freeze_type == 'final' or self.freeze_type == 'permuted':
                    value = self.model_for_reset.state_dict()[name]
                    mask_reverse = torch.abs(getattr(self, PrunedModel.to_mask_name(name)) - 1)
                    param.data = param.data * getattr(self, PrunedModel.to_mask_name(name)) + value * mask_reverse
                elif self.freeze_type == 'gaussian':
                    gen = torch.Generator()
                    gen.manual_seed(seed=0)
                    value = torch.normal(mean=0, std=0.01, size=param.data.size, generator=gen)
                    mask_reverse = torch.abs(getattr(self, PrunedModel.to_mask_name(name)) - 1)
                    param.data = param.data * getattr(self, PrunedModel.to_mask_name(name)) + value * mask_reverse
                else: 
                    raise ValueError('Freezing pruned weights as type {} is not supported.'.format(self.freeze_type))

    def updateBN(self):
        # https://github.com/Eric-mingjie/network-slimming/blob/b395dc07521cbc38f741d971a18fe3f6423c9ab1/main.py#L126
        if self.model.prunable_layer_type == 'BN':
            scale_sparse_rate = 0.0001
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.grad.data.add_(scale_sparse_rate * torch.sign(m.weight.data)) # L1 Norm

    def forward(self, x):
        self._apply_mask()
        return self.model.forward(x)
    
    @property
    def prunable_layer_type(self):
        return self.model.prunable_layer_type

    @property
    def prunable_layer_names(self):
        return self.model.prunable_layer_names

    @property
    def output_layer_names(self):
        return self.model.output_layer_names

    @property
    def loss_criterion(self):
        return self.model.loss_criterion


    def save(self, save_location, save_step):
        self.model.save(save_location, save_step)

    @staticmethod
    def default_hparams(): raise NotImplementedError()
    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError()
    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError()
