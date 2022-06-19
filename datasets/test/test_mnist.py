# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import numpy as np

from datasets import mnist
from testing import test_case


class TestDataset(test_case.TestCase):
    def setUp(self):
        super(TestDataset, self).setUp()
        self.test_set = mnist.Dataset.get_test_set(num_workers=0)
        self.train_set = mnist.Dataset.get_train_set(use_augmentation=True, num_workers=0)
        self.train_set_noaugment = mnist.Dataset.get_train_set(use_augmentation=False, num_workers=0)

    def test_not_none(self):
        self.assertIsNotNone(self.test_set)
        self.assertIsNotNone(self.train_set)
        self.assertIsNotNone(self.train_set_noaugment)

    def test_size(self):
        self.assertEqual(mnist.Dataset.num_classes(), 10)
        self.assertEqual(mnist.Dataset.num_train_examples(), 60000)
        self.assertEqual(mnist.Dataset.num_test_examples(), 10000)

    # test random labels
    def test_randomize_labels_half(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0.5)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 5472)

    def test_randomize_labels_none(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 10000)

    def test_randomize_labels_all(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 1)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 989)
    
    # test symmetric noisy labels
    def test_symmetric_noisy_labels_half(self):
        # labels_before = self.test_set._labels.tolist()
        labels_before = self.test_set._labels.copy()
        self.test_set.symmetric_noisy_labels(0, 0.5)
        labels_after = self.test_set._labels
        examples_match = np.sum(np.equal(labels_before, labels_after).astype(int))
        self.assertEqual(examples_match, 4873)
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
        self.assertEqual(examples_match, 7466)
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
        self.assertEqual(examples_match, 8047)
        for i in [0,1,4,7,8,9]:
            i_labels_after = labels_after[labels_before==i]
            for j in range(10):
                num_i_to_j_class = i_labels_after.tolist().count(j)
                num_i_class = labels_before.tolist().count(i)
                if i == j:
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 1)
                else:
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0)
        for i in [2,3,5,6]: #[9,2,3,5,4]
            i_labels_after = labels_after[labels_before==i]
            for j in range(10):
                num_i_to_j_class = i_labels_after.tolist().count(j)
                num_i_class = labels_before.tolist().count(i)
                if i == j:
                    self.assertAlmostEqual( num_i_to_j_class/num_i_class, 0.5, delta=0.1)
                elif (i==2 and j==7) or (i==5 and j==6) or (i==6 and j==5) or (i==3 and j==8):
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
        self.assertEqual(examples_match, 6108)

    def test_subsample(self):
        # Subsample the test set.
        labels_test = [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
        subsampled_labels_with_seed_zero_test = [6, 9, 2, 6, 7, 6, 1, 4, 7, 1]

        self.assertEqual(self.test_set._labels[:10].tolist(), labels_test)
        self.test_set.subsample(0, 0.1)
        self.assertEqual(len(self.test_set), 1000)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsampled_labels_with_seed_zero_test)

        # Evaluate with a different seed.
        subsampled_labels_with_seed_one_test = [8, 8, 7, 0, 9, 8, 0, 2, 3, 0]
        self.test_set = mnist.Dataset.get_test_set(num_workers=0)
        self.test_set.subsample(1, 0.1)
        self.assertEqual(len(self.test_set), 1000)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsampled_labels_with_seed_one_test)

        # Subsample the train set.
        labels_train = [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]
        subsampled_labels_with_seed_zero_train = [3, 2, 7, 8, 2, 3, 5, 2, 8, 9]

        self.assertEqual(self.train_set._labels[:10].tolist(), labels_train)
        self.train_set.subsample(0, 0.1)
        self.assertEqual(len(self.train_set), 6000)
        self.assertEqual(self.train_set._labels[:10].tolist(), subsampled_labels_with_seed_zero_train)

        # Evaluate with a different seed.
        subsampled_labels_with_seed_one_train = [2, 1, 5, 0, 6, 0, 8, 9, 5, 5]
        self.train_set = mnist.Dataset.get_train_set(use_augmentation=True, num_workers=0)
        self.train_set.subsample(1, 0.1)
        self.assertEqual(len(self.train_set), 6000)
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