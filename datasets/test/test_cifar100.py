# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import numpy as np

from datasets import cifar100
from testing import test_case


class TestDataset(test_case.TestCase):
    def setUp(self):
        super(TestDataset, self).setUp()
        self.test_set = cifar100.Dataset.get_test_set(num_workers=0)
        self.train_set = cifar100.Dataset.get_train_set(use_augmentation=True, num_workers=0)
        self.train_set_noaugment = cifar100.Dataset.get_train_set(use_augmentation=False, num_workers=0)

    def test_not_none(self):
        self.assertIsNotNone(self.test_set)
        self.assertIsNotNone(self.train_set)
        self.assertIsNotNone(self.train_set_noaugment)

    def test_size(self):
        self.assertEqual(cifar100.Dataset.num_classes(), 100)
        self.assertEqual(cifar100.Dataset.num_train_examples(), 50000)
        self.assertEqual(cifar100.Dataset.num_test_examples(), 10000)

    # test random labels
    def test_randomize_labels_half(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0.5)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 5048)

    def test_randomize_labels_none(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 10000)

    def test_randomize_labels_all(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 1)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 97)
    
    # test symmetric noisy labels
    def test_symmetric_noisy_labels_half(self):
        # labels_before = self.test_set._labels.tolist()
        labels_before = self.test_set._labels.copy()
        self.test_set.symmetric_noisy_labels(0, 0.5)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 4964)
        for i in range(0,100):
            i_labels_after = labels_after[labels_before==i]
            num_i_class = labels_before.tolist().count(i)
            for j in range(100):
                num_i_to_j_class = i_labels_after.tolist().count(j)
                frac = num_i_to_j_class/num_i_class
                if i == j :
                    self.assertAlmostEqual( frac, 0.5, delta=0.11)
                else:
                    self.assertAlmostEqual( frac, 0.5/99, delta=0.1)

    def test_symmetric_noisy_labels_none(self):
        labels_before = self.test_set._labels.copy()
        self.test_set.symmetric_noisy_labels(0, 0)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 10000)

    def test_symmetric_noisy_labels_all(self):
        labels_before = self.test_set._labels.copy()
        self.test_set.symmetric_noisy_labels(0, 1)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 0)

    # test asymmetric noisy labels
    def test_asymmetric_noisy_labels_half(self):
        # labels_before = self.test_set._labels.tolist()
        labels_before = self.test_set._labels.copy()
        self.test_set.asymmetric_noisy_labels(1, 0.5)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 4954)
        nb_superclasses = 20
        nb_subclasses = 5
        i = 0
        for sup in range(nb_superclasses):
            for sub in range(nb_subclasses):
                num_i_class = labels_before.tolist().count(i)
                i_labels_after = labels_after[labels_before==i]
                if sub == nb_subclasses - 1:
                    num_i_to_j_class = i_labels_after.tolist().count(sup * nb_subclasses)
                else:
                    num_i_to_j_class = i_labels_after.tolist().count(i + 1)
                frac = num_i_to_j_class/num_i_class
                self.assertAlmostEqual( frac , 0.5, delta=0.2)
                i += 1


    def test_asymmetric_noisy_labels_none(self):
        labels_before = self.test_set._labels.copy()
        self.test_set.asymmetric_noisy_labels(0, 0)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 10000)

    def test_asymmetric_noisy_labels_all(self):
        labels_before = self.test_set._labels.copy()
        self.test_set.asymmetric_noisy_labels(0, 1)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 0)

    def test_subsample(self):
        # Subsample the test set.

        self.test_set.subsample(0, 0.1)
        self.assertEqual(len(self.test_set), 1000)

        self.train_set.subsample(0, 0.1)
        self.assertEqual(len(self.train_set), 5000)

    def test_subsample_twice(self):
        self.train_set.subsample(1, 0.1)
        with self.assertRaises(ValueError):
            self.train_set.subsample(1, 0.1)


test_case.main()