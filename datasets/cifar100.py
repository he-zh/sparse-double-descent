# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import numpy as np
import os
from PIL import Image
import sys
import torchvision

from datasets import base
from foundations.local import  Platform
from numpy.testing import assert_array_almost_equal

class CIFAR100(torchvision.datasets.CIFAR100):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-100 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR100 if the print
    message doesn't bother you.
    """

    def download(self):
        with open(os.devnull, 'w') as fp:
            sys.stdout = fp
            super(CIFAR100, self).download()
            sys.stdout = sys.__stdout__


class Dataset(base.ImageDataset):
    """The CIFAR-100 dataset."""

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 100

    @staticmethod
    def get_train_set(use_augmentation, num_workers):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR100(train=True, root=os.path.join(Platform().dataset_root, 'cifar100'), download=True)
        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])

    @staticmethod
    def get_test_set(num_workers):
        test_set = CIFAR100(train=False, root=os.path.join(Platform().dataset_root, 'cifar100'), download=True)
        return Dataset(test_set.data, np.array(test_set.targets))

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def asymmetric_noisy_labels(self, seed:int, fraction: float) -> None:
        """Inject asymmetric label noise into the specified fraction of the dataset by pair flipping,
        by flipping each class into the next class within the same super-class."""
        P = np.eye(self.num_classes())
        # n = self.cfg_trainer['percent']
        nb_superclasses = 20
        nb_subclasses = 5

        def build_transition_matrix(size, fraction):
            trans_matrix = (1. - fraction) * np.eye(size)
            for i in np.arange(size - 1):
                trans_matrix[i, i + 1] = fraction

            # adjust last row
            trans_matrix[size - 1, 0] = fraction

            assert_array_almost_equal(trans_matrix.sum(axis=1), 1, 1)
            return trans_matrix

        # if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i+1) * nb_subclasses
            P[init:end, init:end] = build_transition_matrix(nb_subclasses, fraction)

        # y_train_noisy = self.multiclass_noisify(self.train_labels, P=P,
        #                                     random_state=0)
        # actual_noise = (y_train_noisy != self.train_labels).mean()
        # assert actual_noise > 0.0
        # self.train_labels = y_train_noisy
        self.multiclass_labels_noisify(seed=seed, trans_matrix=P)
 

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
