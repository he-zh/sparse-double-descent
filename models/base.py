# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import abc
import torch
import typing
import os
from foundations.step import Step



class Model(torch.nn.Module, abc.ABC):
    """The base class used by all models in this codebase."""

    _prunable_layer_type: str = 'default'

    @staticmethod
    @abc.abstractmethod
    def is_valid_model_name(model_name: str) -> bool:
        """Is the model name string a valid name for models in this class?"""

        pass

    @staticmethod
    @abc.abstractmethod
    def get_model_from_name(
        model_name: str,
        outputs: int,
        initializer: typing.Callable[[torch.nn.Module], None]
    ) -> 'Model':
        """Returns an instance of this class as described by the model_name string."""

        pass

    @property
    def prunable_layer_type(self) -> str:
        """The type of nn.module that is valid for pruning.

        By default, only the weights of convolutional and linear layers are prunable.
        If network-slimming pruning is utilized, then BN will be set as the prunable type.
        """
        return self._prunable_layer_type

    @prunable_layer_type.setter
    def prunable_layer_type(self, type: str):
        if type in ['default', 'BN']:
            self._prunable_layer_type = type
        else:
            raise ValueError('Not recognized prunabel_layer_type: {}'.format(type))

    @property
    def prunable_layer_names(self) -> typing.List[str]:
        """A list of the names of Tensors of this model that are valid for pruning.

        By default, only the weights of convolutional and linear layers are prunable.
        If network-slimming pruning is utilized, then the weights and biases of batch normlization
        layers will be set as prunable.
        """
        if self.prunable_layer_type == 'BN':
            return [name + m for name, module in self.named_modules() if
                    isinstance(module, torch.nn.modules.BatchNorm2d) for m in ['.weight', '.bias']]
        else:
            return [name + '.weight' for name, module in self.named_modules() if
                    isinstance(module, torch.nn.modules.conv.Conv2d) or
                    isinstance(module, torch.nn.modules.linear.Linear)]

    @property
    @abc.abstractmethod
    def output_layer_names(self) -> typing.List[str]:
        """A list of the names of the Tensors of the output layer of this model."""

        pass

    @property
    @abc.abstractmethod
    def loss_criterion(self) -> torch.nn.Module:
        """The loss criterion to use for this model."""

        pass

    def updateBN(self):
        """
        Add additional subgradient descent of batch normalization weights on the sparsity-induced penalty term 
        for network-slimming pruning
        """
        pass

    def save(self, save_location: str, save_step: Step):
        if not os.path.exists(save_location): os.makedirs(save_location)
        torch.save(self.state_dict(), os.path.join(save_location, 'model_ep{}_it{}.pth'.format(save_step.ep, save_step.it)))


class DataParallel(Model, torch.nn.DataParallel):
    def __init__(self, module: Model):
        super(DataParallel, self).__init__(module=module)
        
    @property
    def prunable_layer_type(self): return self.module.prunable_layer_type

    @property
    def prunable_layer_names(self): return self.module.prunable_layer_names

    @property
    def output_layer_names(self): return self.module.output_layer_names

    @property
    def loss_criterion(self): return self.module.loss_criterion

    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError

    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError

    @staticmethod
    def default_hparams(): raise NotImplementedError

    def updateBN(self):
        return self.module.updateBN()

    def save(self, save_location: str, save_step: Step):
        self.module.save(save_location, save_step)


class DistributedDataParallel(Model, torch.nn.parallel.DistributedDataParallel):
    def __init__(self, module: Model, device_ids):
        super(DistributedDataParallel, self).__init__(module=module, device_ids=device_ids)

    @property
    def prunable_layer_type(self): return self.module.prunable_layer_type

    @property
    def prunable_layer_names(self): return self.module.prunable_layer_names

    @property
    def output_layer_names(self): return self.module.output_layer_names

    @property
    def loss_criterion(self): return self.module.loss_criterion

    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError

    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError

    @staticmethod
    def default_hparams(): raise NotImplementedError

    def updateBN(self):
        return self.module.updateBN()

    def save(self, save_location: str, save_step: Step):
        self.module.save(save_location, save_step)
