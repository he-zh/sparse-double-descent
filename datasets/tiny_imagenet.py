# Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

import concurrent
import numpy as np
import os
from PIL import Image
import torchvision
from datasets import base
from foundations.local import Platform


def _get_samples(root, y_name, y_num):
    y_dir = os.path.join(root, y_name)
    # y_dir = os.path.join(y_dir, 'images')
    if not os.path.isdir(y_dir): return []
    output = [(os.path.join(y_dir, f), y_num) for f in os.listdir(y_dir) if f.lower().endswith('jpeg')]
    return output


class Dataset(base.ImageDataset):
    """Tiny-ImageNet"""

    def __init__(self, loc: str, image_transforms, num_workers=0):
        # Load the data.
        classes = sorted(os.listdir(loc))
        samples = []

        if num_workers > 0:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
            futures = [executor.submit(_get_samples, loc, y_name, y_num) for y_num, y_name in enumerate(classes)]
            for d in concurrent.futures.wait(futures)[0]: samples += d.result()
        else:
            for y_num, y_name in enumerate(classes):
                samples += _get_samples(loc, y_name, y_num)

        examples, labels = zip(*samples)
        super(Dataset, self).__init__(
            np.array(examples), np.array(labels), image_transforms,
            [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @staticmethod
    def num_train_examples(): return 100000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 200

    @staticmethod
    def _augment_transforms():
        return [
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.RandomHorizontalFlip()
        ]

    @staticmethod
    def _transforms():
        return [torchvision.transforms.Resize(45), torchvision.transforms.CenterCrop(32)]

    @staticmethod
    def get_train_set(use_augmentation, num_workers):
        transforms = Dataset._augment_transforms() if use_augmentation else Dataset._transforms()
        return Dataset(os.path.join(Platform().tiny_imagenet_root, 'train'), transforms, num_workers)

    @staticmethod
    def get_test_set(num_workers):
        return Dataset(os.path.join(Platform().tiny_imagenet_root, 'val'), Dataset._transforms(), num_workers)

    @staticmethod
    def example_to_image(example):
        with open(example, 'rb') as fp:
            return Image.open(fp).convert('RGB')


DataLoader = base.DataLoader
