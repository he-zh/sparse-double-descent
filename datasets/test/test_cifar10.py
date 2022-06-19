# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import numpy as np

from datasets import cifar10
from testing import test_case


class TestDataset(test_case.TestCase):
    def setUp(self):
        super(TestDataset, self).setUp()
        self.test_set = cifar10.Dataset.get_test_set(num_workers=0)
        self.train_set = cifar10.Dataset.get_train_set(use_augmentation=True, num_workers=0)
        self.train_set_noaugment = cifar10.Dataset.get_train_set(use_augmentation=False, num_workers=0)

    def test_not_none(self):
        self.assertIsNotNone(self.test_set)
        self.assertIsNotNone(self.train_set)
        self.assertIsNotNone(self.train_set_noaugment)

    def test_size(self):
        self.assertEqual(cifar10.Dataset.num_classes(), 10)
        self.assertEqual(cifar10.Dataset.num_train_examples(), 50000)
        self.assertEqual(cifar10.Dataset.num_test_examples(), 10000)

    # test random labels
    def test_randomize_labels_half(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0.5)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 5503)

    def test_randomize_labels_none(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 10000)

    def test_randomize_labels_all(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 1)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 1020)
    
    # test symmetric noisy labels
    def test_symmetric_noisy_labels_half(self):
        # labels_before = self.test_set._labels.tolist()
        labels_before = self.test_set._labels.copy()
        self.test_set.symmetric_noisy_labels(0, 0.5)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 4867)
        for i in range(0,10):
            i_labels_after = labels_after[labels_before==i]
            for j in range(10):
                num_i_to_j_class = i_labels_after.tolist().count(j)
                num_i_class = labels_before.tolist().count(i)
                if i == j :
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0.5, delta=0.1)
                else:
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0.5/9, delta=0.1)

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

    # test pairflip noisy labels
    def test_pairflip_noisy_labels_quarter(self):
        # labels_before = self.test_set._labels.tolist()
        labels_before = self.test_set._labels.copy()
        self.test_set.pairflip_noisy_labels(0, 0.25)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 7467)
        for i in range(0,10):
            i_labels_after = labels_after[labels_before==i]
            for j in range(10):
                num_i_to_j_class = i_labels_after.tolist().count(j)
                num_i_class = labels_before.tolist().count(i)
                if i == j :
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0.75, delta=0.1)
                elif j == i+1 or (j == 0 and i == 9):
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0.25, delta=0.1)
                else:
                    self.assertEqual( num_i_to_j_class/num_i_class, 0)

    def test_pairflip_noisy_labels_none(self):
        labels_before = self.test_set._labels.copy()
        self.test_set.pairflip_noisy_labels(0, 0)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 10000)

    def test_pairflip_noisy_labels_all(self):
        labels_before = self.test_set._labels.copy()
        self.test_set.pairflip_noisy_labels(0, 1)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 0)

    # test asymmetric noisy labels
    def test_asymmetric_noisy_labels_half(self):
        # labels_before = self.test_set._labels.tolist()
        labels_before = self.test_set._labels.copy()
        self.test_set.asymmetric_noisy_labels(0, 0.5)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 7500)
        for i in [0,1,6,7,8]:
            i_labels_after = labels_after[labels_before==i]
            for j in range(10):
                num_i_to_j_class = i_labels_after.tolist().count(j)
                num_i_class = labels_before.tolist().count(i)
                if i == j:
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 1)
                else:
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0)
        for i in [9,2,3,5,4]: #[9,2,3,5,4]
            i_labels_after = labels_after[labels_before==i]
            for j in range(10):
                num_i_to_j_class = i_labels_after.tolist().count(j)
                num_i_class = labels_before.tolist().count(i)
                if i == j:
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0.5, delta=0.1)
                elif (i==9 and j==1) or (i==2 and j==0) or (i==3 and j==5) or (i==5 and j==3) or (i==4 and j==7):
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0.5, delta=0.1)
                else:
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0, delta=0.1)

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
        self.assertEqual(examples_match, 5000)

    def test_subsample(self):
        # Subsample the test set.
        labels_test = [3, 8, 8, 0, 6, 6, 1, 6, 3, 1]
        subsampled_labels_with_seed_zero_test = [7, 9, 3, 8, 0, 1, 0, 6, 3, 7]

        self.assertEqual(self.test_set._labels[:10].tolist(), labels_test)
        self.test_set.subsample(0, 0.1)
        self.assertEqual(len(self.test_set), 1000)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsampled_labels_with_seed_zero_test)

        # Evaluate with a different seed.
        subsampled_labels_with_seed_one_test = [1, 6, 4, 7, 9, 1, 7, 2, 8, 5]
        self.test_set = cifar10.Dataset.get_test_set(num_workers=0)
        self.test_set.subsample(1, 0.1)
        self.assertEqual(len(self.test_set), 1000)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsampled_labels_with_seed_one_test)

        # Subsample the train set.
        labels_train = [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]
        subsampled_labels_with_seed_zero_train = [6, 7, 3, 6, 8, 1, 2, 9, 5, 2]

        self.assertEqual(self.train_set._labels[:10].tolist(), labels_train)
        self.train_set.subsample(0, 0.1)
        self.assertEqual(len(self.train_set), 5000)
        self.assertEqual(self.train_set._labels[:10].tolist(), subsampled_labels_with_seed_zero_train)

        # Evaluate with a different seed.
        subsampled_labels_with_seed_one_train = [2, 1, 4, 2, 5, 6, 4, 3, 8, 2]
        self.train_set = cifar10.Dataset.get_train_set(use_augmentation=True, num_workers=0)
        self.train_set.subsample(1, 0.1)
        self.assertEqual(len(self.train_set), 5000)
        self.assertEqual(self.train_set._labels[:10].tolist(), subsampled_labels_with_seed_one_train)

    def test_subsample_twice(self):
        self.train_set.subsample(1, 0.1)
        with self.assertRaises(ValueError):
            self.train_set.subsample(1, 0.1)

    def test_subsample_noisy_examples(self):
        labels_before = self.test_set._labels.copy()
        self.test_set.symmetric_noisy_labels(0, 1)
        self.test_set.subsample_noisy_examples()
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 0)

    def test_subsample_clean_examples(self):
        self.test_set.symmetric_noisy_labels(0, 1)
        self.test_set.subsample_clean_examples()
        self.assertEqual(len(self.test_set._labels), 0)

test_case.main()