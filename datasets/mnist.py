# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import os
from PIL import Image
import numpy as np
import torchvision

from datasets import base
from foundations.local import  Platform


class Dataset(base.ImageDataset):
    """The MNIST dataset."""

    @staticmethod
    def num_train_examples(): return 60000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation, num_workers):
        # No augmentation for MNIST.
        train_set = torchvision.datasets.MNIST(
            train=True, root=os.path.join(Platform().dataset_root, 'mnist'), download=True)
        return Dataset(train_set.data, train_set.targets)

    @staticmethod
    def get_test_set(num_workers):
        test_set = torchvision.datasets.MNIST(
            train=False, root=os.path.join(Platform().dataset_root, 'mnist'), download=True)
        return Dataset(test_set.data, test_set.targets)

    def __init__(self,  examples, labels):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms)

    def asymmetric_noisy_labels(self, seed:int, fraction: float) -> None:
        """Inject asymmetric label noise into the specified fraction of the dataset by pair flipping."""
        # https://github.com/xiaoboxia/CDR/blob/main/utils.py

        P = np.eye(10)
        # 2 -> 7
        P[2, 2], P[2, 7] = 1. - fraction, fraction
        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - fraction, fraction
        P[6, 6], P[6, 5] = 1. - fraction, fraction
        # 3 -> 8
        P[3, 3], P[3, 8] = 1. - fraction, fraction

        self.multiclass_labels_noisify(seed=seed, trans_matrix=P)


    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')


DataLoader = base.DataLoader
