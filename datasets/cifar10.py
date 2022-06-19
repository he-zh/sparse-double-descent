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

class CIFAR10(torchvision.datasets.CIFAR10):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        with open(os.devnull, 'w') as fp:
            sys.stdout = fp
            super(CIFAR10, self).download()
            sys.stdout = sys.__stdout__


class Dataset(base.ImageDataset):
    """The CIFAR-10 dataset."""

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation, num_workers):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR10(train=True, root=os.path.join(Platform().dataset_root, 'cifar10'), download=True)
        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])

    @staticmethod
    def get_test_set(num_workers):
        test_set = CIFAR10(train=False, root=os.path.join(Platform().dataset_root, 'cifar10'), download=True)
        return Dataset(test_set.data, np.array(test_set.targets))

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def asymmetric_noisy_labels(self, seed:int, fraction: float) -> None:
        """Inject asymmetric label noise into the specified fraction of the dataset by pair flipping."""        
        # https://github.com/shengliu66/ELR/blob/master/ELR/data_loader/cifar10.py
        _labels = self._labels.copy()
        for i in range(self.num_classes()):
            indices = np.where(_labels == i)[0]
            num_to_noisify_label_i = np.ceil(len(indices) * fraction).astype(int)
            np.random.RandomState(seed=seed+i).shuffle(indices)
            for j, idx in enumerate(indices):
                if j < num_to_noisify_label_i:
                    # self.noise_indx.append(idx)
                    # truck -> automobile
                    if i == 9:
                        self._labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self._labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self._labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self._labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self._labels[idx] = 7
                

    def example_to_image(self, example):
        return Image.fromarray(example)



DataLoader = base.DataLoader
