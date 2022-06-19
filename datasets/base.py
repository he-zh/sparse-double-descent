# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import abc
from sys import platform
import numpy as np
from PIL import Image
import torch
import torchvision
from foundations.local import  Platform


class Dataset(abc.ABC, torch.utils.data.Dataset):
    """The base class for all datasets in this framework."""

    @staticmethod
    @abc.abstractmethod
    def num_test_examples() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def num_train_examples() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def num_classes() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_train_set(use_augmentation: bool) -> 'Dataset':
        pass

    @staticmethod
    @abc.abstractmethod
    def get_test_set() -> 'Dataset':
        pass

    def __init__(self, examples: np.ndarray, labels):
        """Create a dataset object.

        examples is a numpy array of the examples (or the information necessary to get them).
        Only the first dimension matters for use in this abstract class.

        labels is a numpy array of the labels. Each entry is a zero-indexed integer encoding
        of the label.
        """

        if examples.shape[0] != labels.shape[0]:
            raise ValueError('Different number of examples ({}) and labels ({}).'.format(
                             examples.shape[0], labels.shape[0]))
        self._examples = examples
        self._labels = labels if isinstance(labels, np.ndarray) else labels.numpy().astype(np.int64)
        self._ori_labels = labels.copy() if isinstance(labels, np.ndarray) else labels.numpy().copy().astype(np.int64)
        self._subsampled = False

    def randomize_labels(self, seed: int, fraction: float) -> None:
        """Randomize the labels of the specified fraction of the dataset."""

        num_to_randomize = np.ceil(len(self._labels) * fraction).astype(int)
        randomized_labels = np.random.RandomState(seed=seed).randint(self.num_classes(), size=num_to_randomize)
        examples_to_randomize = np.random.RandomState(seed=seed+1).permutation(len(self._labels))[:num_to_randomize]
        self._labels[examples_to_randomize] = randomized_labels

    def multiclass_labels_noisify(self, seed: int, trans_matrix: np.ndarray) -> None:
        """ Flip classes according to transition probability matrix.
        It expects a number between 0 and the number of classes - 1."""
        
        for idx in np.arange(self._labels.shape[0]):
            l = self._labels[idx]
            # draw a vector with only an 1
            flipped = np.random.RandomState(seed+idx).multinomial(1, trans_matrix[l, :], 1)[0]
            self._labels[idx] = np.where(flipped == 1)[0]

    def symmetric_noisy_labels(self, seed:int, fraction: float) -> None:
        """Generate symmetric label noise of the specified fraction of the dataset."""

        P = np.ones((self.num_classes(), self.num_classes()))
        P = (fraction / (self.num_classes() - 1)) * P
        for i in range(0, self.num_classes()):
            P[i, i] = 1. - fraction
        self.multiclass_labels_noisify(seed=seed+2, trans_matrix=P)

    def pairflip_noisy_labels(self, seed:int, fraction: float) -> None:
        """Generate label noise by flipping each class to its adjacent class"""
        P = np.eye(self.num_classes())
        for i in range(0, self.num_classes()-1):
            P[i, i], P[i, i+1] = 1 - fraction, fraction
        P[self.num_classes()-1, self.num_classes()-1], P[self.num_classes()-1, 0] = 1 - fraction, fraction
        self.multiclass_labels_noisify(seed=seed+3, trans_matrix=P)

    def subsample(self, seed: int, fraction: float) -> None:
        """Subsample the dataset."""

        if self._subsampled:
            raise ValueError('Cannot subsample more than once.')
        self._subsampled = True

        examples_to_retain = np.ceil(len(self._labels) * fraction).astype(int)
        examples_to_retain = np.random.RandomState(seed=seed+1).permutation(len(self._labels))[:examples_to_retain]
        self._examples = self._examples[examples_to_retain]
        self._labels = self._labels[examples_to_retain]
        self._ori_labels = self._ori_labels[examples_to_retain]

    def __len__(self):
        return self._labels.size

    def __getitem__(self, index):
        """If there is custom logic for example loading, this method should be overridden."""

        return self._examples[index], self._labels[index]


class ImageDataset(Dataset):
    @abc.abstractmethod
    def example_to_image(self, example: np.ndarray) -> Image: pass

    def __init__(self, examples, labels, image_transforms=None, tensor_transforms=None,
                 joint_image_transforms=None, joint_tensor_transforms=None):
        super(ImageDataset, self).__init__(examples, labels)
        self._image_transforms = image_transforms or []
        self._tensor_transforms = tensor_transforms or []
        self._joint_image_transforms = joint_image_transforms or []
        self._joint_tensor_transforms = joint_tensor_transforms or []

        self._composed = None

    def __getitem__(self, index):
        if not self._composed:
            self._composed = torchvision.transforms.Compose(
                self._image_transforms + [torchvision.transforms.ToTensor()] + self._tensor_transforms)

        example, label = self._examples[index], self._labels[index]
        example = self.example_to_image(example)
        for t in self._joint_image_transforms: example, label = t(example, label)
        example = self._composed(example)
        for t in self._joint_tensor_transforms: example, label = t(example, label)
        return example, label

    def blur(self, blur_factor: float) -> None:
        """Add a transformation that blurs the image by downsampling by blur_factor."""

        def blur_transform(image):
            size = list(image.size)
            image = torchvision.transforms.Resize([int(s / blur_factor) for s in size])(image)
            image = torchvision.transforms.Resize(size)(image)
            return image
        self._image_transforms.append(blur_transform)

    def unsupervised_rotation(self, seed: int):
        """Switch the task to unsupervised rotation."""

        self._labels = np.random.RandomState(seed=seed).randint(4, size=self._labels.size)

        def rotate_transform(image, label):
            return torchvision.transforms.RandomRotation(label*90)(image), label
        self._joint_image_transforms.append(rotate_transform)


class ShuffleSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_examples):
        self._num_examples = num_examples
        self._seed = -1

    def __iter__(self):
        if self._seed == -1:
            indices = list(range(self._num_examples))
        elif self._seed is None:
            indices = torch.randperm(self._num_examples).tolist()
        else:
            g = torch.Generator()
            if self._seed is not None: g.manual_seed(self._seed)
            indices = torch.randperm(self._num_examples, generator=g).tolist()

        return iter(indices)

    def __len__(self):
        return self._num_examples

    def shuffle_dataorder(self, seed: int):
        self._seed = seed



class DataLoader(torch.utils.data.DataLoader):
    """A wrapper that makes it possible to access the custom shuffling logic."""

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool = True):

        self._sampler = ShuffleSampler(len(dataset))

        self._iterations_per_epoch = np.ceil(len(dataset) / batch_size).astype(int)

        super(DataLoader, self).__init__(
            dataset, batch_size, sampler=self._sampler, num_workers=num_workers,
            pin_memory=pin_memory and Platform().torch_device.type == 'cuda')

    def shuffle(self, seed: int):
        self._sampler.shuffle_dataorder(seed)

    @property
    def iterations_per_epoch(self):
        return self._iterations_per_epoch
