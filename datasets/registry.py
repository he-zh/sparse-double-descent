# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import numpy as np

from datasets import base, mnist, cifar10, cifar100, tiny_imagenet
from foundations.hparams import DatasetHparams


registered_datasets = {'mnist': mnist,'cifar10': cifar10, 'cifar100': cifar100, 
                       'tiny_imagenet': tiny_imagenet}


def get(dataset_hparams: DatasetHparams, train: bool = True, subsample_labels_type: str = None):
    """Get the train or test set corresponding to the hyperparameters."""

    seed = dataset_hparams.transformation_seed or 0

    # Get the dataset itself.
    if dataset_hparams.dataset_name in registered_datasets:
        use_augmentation = train and not dataset_hparams.do_not_augment
        if train:
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_train_set(use_augmentation,
                                                                                              dataset_hparams.num_workers)
        else:
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_test_set(dataset_hparams.num_workers)
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    # Transform the dataset.
    if dataset_hparams.random_labels_fraction is not None and dataset_hparams.noisy_labels_fraction is not None:
        raise ValueError('random_labels_fraction and noisy_labels_fraction cannot be assigned at the same time.')

    if train and dataset_hparams.random_labels_fraction is not None:
        dataset.randomize_labels(seed=seed, fraction=dataset_hparams.random_labels_fraction)
    
    if train and dataset_hparams.noisy_labels_fraction is not None:
        if dataset_hparams.noisy_labels_type == 'symmetric':
            dataset.symmetric_noisy_labels(seed=seed, fraction=dataset_hparams.noisy_labels_fraction)
        elif dataset_hparams.noisy_labels_type == 'asymmetric':
            dataset.asymmetric_noisy_labels(seed=seed, fraction=dataset_hparams.noisy_labels_fraction)
        elif dataset_hparams.noisy_labels_type == 'pairflip':
            dataset.pairflip_noisy_labels(seed=seed, fraction=dataset_hparams.noisy_labels_fraction)
        elif dataset_hparams.noisy_labels_type is None:
            raise ValueError('Please specify the type of noisy labels.')
        else: 
            raise ValueError('Noisy label type of {} is not implemented.'.format(dataset_hparams.noisy_labels_type))
    
    if train and dataset_hparams.subsample_fraction is not None:
        dataset.subsample(seed=seed, fraction=dataset_hparams.subsample_fraction)

    if train and dataset_hparams.blur_factor is not None:
        if not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can blur images.')
        else:
            dataset.blur(seed=seed, blur_factor=dataset_hparams.blur_factor)

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        elif not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can only do unsupervised rotation to images.')
        else:
            dataset.unsupervised_rotation(seed=seed)

    # Create the loader.
    return registered_datasets[dataset_hparams.dataset_name].DataLoader(
        dataset, batch_size=dataset_hparams.batch_size, num_workers=dataset_hparams.num_workers)


def iterations_per_epoch(dataset_hparams: DatasetHparams):
    """Get the number of iterations per training epoch."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_train_examples = registered_datasets[dataset_hparams.dataset_name].Dataset.num_train_examples()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.subsample_fraction is not None:
        num_train_examples *= dataset_hparams.subsample_fraction

    return np.ceil(num_train_examples / dataset_hparams.batch_size).astype(int)


def num_classes(dataset_hparams: DatasetHparams):
    """Get the number of classes."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_classes = registered_datasets[dataset_hparams.dataset_name].Dataset.num_classes()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        else:
            return 4

    return num_classes
