# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py

from functools import partial
import torch
import torchvision

from foundations import hparams
from models import base
from pruning import magnitude
from experiments.finetune.desc import FinetuningDesc

class ResNet(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=100, width=64):
        """To make it possible to vary the width, we need to override the constructor of the torchvision resnet."""

        torch.nn.Module.__init__(self)  # Skip the parent constructor. This replaces it.
        self._norm_layer = torch.nn.BatchNorm2d
        self.inplanes = width
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # The initial convolutional layer.
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The subsequent blocks.
        self.layer1 = self._make_layer(block, width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, width*2, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, width*4, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, width*8, layers[3], stride=2, dilate=False)

        # The last layers.
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(width*8*block.expansion, num_classes)

        # Default init.
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Model(base.Model):
    """A residual neural network as originally designed for ImageNet."""

    def __init__(self, model_fn, initializer, outputs=None):
        super(Model, self).__init__()

        self.model = model_fn(num_classes=outputs or 100)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.apply(initializer)

    def forward(self, x):
        return self.model(x)

    @property
    def output_layer_names(self):
        return ['model.fc.weight', 'model.fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_pytorch_resnet_') and
                5 >= len(model_name.split('_')) >= 4 and
                model_name.split('_')[3].isdigit() and
                int(model_name.split('_')[3]) in [18, 34, 50, 101, 152, 200])

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=100):
        """Name: cifar_pytorch_resnet_D[_W].

        D is the model depth (e.g., 50 for ResNet-50). W is the model width - the number of filters in the first
        residual layers. By default, this number is 64."""

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        num = int(model_name.split('_')[3])
        if num == 18: model_fn = partial(ResNet, torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
        elif num == 34: model_fn = partial(ResNet, torchvision.models.resnet.BasicBlock, [3, 4, 6, 3])
        elif num == 50: model_fn = partial(ResNet, torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        elif num == 101: model_fn = partial(ResNet, torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
        elif num == 152: model_fn = partial(ResNet, torchvision.models.resnet.Bottleneck, [3, 8, 36, 3])
        elif num == 200: model_fn = partial(ResNet, torchvision.models.resnet.Bottleneck, [3, 24, 36, 3])
        elif num == 269: model_fn = partial(ResNet, torchvision.models.resnet.Bottleneck, [3, 30, 48, 8])

        if len(model_name.split('_')) == 5:
            width = int(model_name.split('_')[4])
            model_fn = partial(model_fn, width=width)

        return Model(model_fn, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_pytorch_resnet_18',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar100',
            batch_size=128,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep',
        )

        pruning_hparams = magnitude.PruningHparams(
            pruning_strategy='magnitude',
            pruning_fraction=0.2,
            pruning_scope='global',
        )

        finetuning_hparams = hparams.FinetuningHparams(
            lr=0.001,
            training_steps='160ep',
        )
        
        return FinetuningDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams, finetuning_hparams)

