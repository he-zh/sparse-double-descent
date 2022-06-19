# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import numpy as np
import sys
from torch.cuda import device, init
sys.path.append("./")
import models.registry
from pruning.random import Strategy
from pruning.random import PruningHparams
from testing import test_case


class TestRandom(test_case.TestCase):
    def setUp(self):
        super(TestRandom, self).setUp()

        model_hparams = models.registry.get_default_hparams('cifar_resnet_20').model_hparams
        self.model = models.registry.get(model_hparams)

    def test_get_pruning_hparams(self):
        self.assertTrue(issubclass(Strategy.get_pruning_hparams(), PruningHparams))

    def test_globally_prune(self):
        hparams = PruningHparams('random', pruning_fraction=0.2)
        m = Strategy.prune(hparams, self.model)

        # Check that the mask only contains entries for the prunable layers.
        self.assertEqual(set(m.keys()), set(self.model.prunable_layer_names))

        # Check that the masks are the same sizes as the tensors.
        for k in self.model.prunable_layer_names:
            self.assertEqual(list(m[k].shape), list(self.model.state_dict()[k].shape))

        # Check that the right fraction of weights was pruned among prunable layers.
        m = m.numpy()
        total_pruned = np.sum([np.sum(1 - v) for v in m.values()])
        total_weights = np.sum([v.size for v in m.values()])
        actual_fraction = float(total_pruned) / total_weights
        self.assertGreaterEqual(actual_fraction, hparams.pruning_fraction)
        self.assertGreater(hparams.pruning_fraction + 0.0001, actual_fraction)

        # Ensure the random masks generated with random seed are the same
        hparams_seed1 = PruningHparams('random', pruning_fraction=0.2, random_mask_seed=1)
        m_seed1 = Strategy.prune(hparams_seed1, self.model)
        m_seed2 = Strategy.prune(hparams_seed1, self.model)
        for k in m:
            self.assertTrue((m_seed2[k]==m_seed1[k]).all())

    def test_globally_iterative_pruning(self):
        hparams = PruningHparams('random', pruning_fraction=0.2)
        m = Strategy.prune(hparams, self.model)
        m2 = Strategy.prune(hparams, self.model, m)

        # Ensure that all weights pruned before are still pruned here.
        m, m2 = m.numpy(), m2.numpy()
        self.assertEqual(set(m.keys()), set(m2.keys()))
        for k in m:
            self.assertTrue(np.all(m[k] >= m2[k]))

        total_pruned = np.sum([np.sum(1 - v) for v in m2.values()])
        total_weights = np.sum([v.size for v in m2.values()])
        actual_fraction = float(total_pruned) / total_weights
        expected_fraction = 1 - (1 - hparams.pruning_fraction) ** 2
        self.assertGreaterEqual(actual_fraction, expected_fraction)
        self.assertGreater(expected_fraction + 0.0001, actual_fraction)

        # Check that mask generated with a random seed will share same random importance scores of weights
        hparams_seed1 = PruningHparams('random', pruning_fraction=0.2, random_mask_seed=1)
        m_seed1 = Strategy.prune(hparams_seed1, self.model)
        m_seed1_2 = Strategy.prune(hparams_seed1, self.model, m_seed1)
        hparams_seed2 = PruningHparams('random', pruning_fraction=0.36, random_mask_seed=1)
        m_seed2 = Strategy.prune(hparams_seed2, self.model)
        m_seed1, m_seed1_2, m_seed2 = m_seed1.numpy(), m_seed1_2.numpy(), m_seed2.numpy()
        for k in m:
            self.assertTrue(np.all(m_seed1[k] >= m_seed2[k]))
        same_pruned = 0
        for k in m:
            same_pruned += np.sum(m_seed2[k]==m_seed1_2[k])
        same_fraction = same_pruned/total_weights
        self.assertGreaterEqual(0.0001, 1-same_fraction)

    def test_layer_wise_prune(self):
        hparams = PruningHparams('random', pruning_fraction = 0.2, pruning_scope = 'layer')
        m = Strategy.prune(hparams, self.model)

        # Check that the mask only contains entries for the prunable layers.
        self.assertEqual(set(m.keys()), set(self.model.prunable_layer_names))

        # Check that the masks are the same sizes as the tensors.
        for k in self.model.prunable_layer_names:
            self.assertEqual(list(m[k].shape), list(self.model.state_dict()[k].shape))

        # Check that the right fraction of weights was pruned among each prunable layer.
        m = m.numpy()
        for k in m:
            layer_pruned = np.sum(1 - m[k])
            layer_weights = np.sum(m[k].size)
            layer_fraction = float(layer_pruned) / layer_weights
            self.assertGreaterEqual(layer_fraction, hparams.pruning_fraction)
            self.assertGreater(hparams.pruning_fraction + 0.1, layer_fraction)
        
        # Check that the right fraction of weights was pruned among all prunable layers.
        total_pruned = np.sum([np.sum(1 - v) for v in m.values()])
        total_weights = np.sum([v.size for v in m.values()])
        actual_fraction = float(total_pruned) / total_weights
        self.assertGreaterEqual(actual_fraction, hparams.pruning_fraction)
        self.assertGreater(hparams.pruning_fraction + 0.001, actual_fraction)

        # Ensure the random masks generated with random seed are the same
        hparams_seed1 = PruningHparams('random', pruning_fraction=0.2, random_mask_seed=1)
        m_seed1 = Strategy.prune(hparams_seed1, self.model)
        m_seed2 = Strategy.prune(hparams_seed1, self.model)
        for k in m:
            self.assertTrue((m_seed2[k]==m_seed1[k]).all())

    def test_layer_wise_iterative_pruning(self):
        hparams = PruningHparams('random', pruning_fraction= 0.2, pruning_scope='layer')
        m = Strategy.prune(hparams, self.model)
        m2 = Strategy.prune(hparams, self.model, m)

        # Ensure that all weights pruned before are still pruned here.
        m, m2 = m.numpy(), m2.numpy()
        self.assertEqual(set(m.keys()), set(m2.keys()))
        for k in m:
            self.assertTrue(np.all(m[k] >= m2[k]))

        for k in m:
            layer_pruned = np.sum(1 - m2[k])
            layer_weights = np.sum(m2[k].size)
            layer_fraction = float(layer_pruned) / layer_weights
            expected_fraction = 1 - (1 - hparams.pruning_fraction) ** 2
            self.assertGreaterEqual(layer_fraction, expected_fraction)
            # self.assertGreater(expected_fraction + 0.0001, layer_fraction)
        total_pruned = np.sum([np.sum(1 - v) for v in m2.values()])
        total_weights = np.sum([v.size for v in m2.values()])
        actual_fraction = float(total_pruned) / total_weights
        expected_fraction = 1 - (1 - hparams.pruning_fraction) ** 2
        self.assertGreaterEqual(actual_fraction, expected_fraction)
        self.assertGreater(expected_fraction + 0.001, actual_fraction)

        # Check that mask generated with a random seed will share same random importance scores of weights
        hparams_seed1 = PruningHparams('random', pruning_fraction=0.2, pruning_scope='layer', random_mask_seed=1)
        m_seed1 = Strategy.prune(hparams_seed1, self.model)
        m_seed1_2 = Strategy.prune(hparams_seed1, self.model, m_seed1)
        hparams_seed2 = PruningHparams('random', pruning_fraction=0.36, pruning_scope='layer', random_mask_seed=1)
        m_seed2 = Strategy.prune(hparams_seed2, self.model)
        m_seed1, m_seed1_2, m_seed2 = m_seed1.numpy(), m_seed1_2.numpy(), m_seed2.numpy()
        for k in m:
            self.assertTrue(np.all(m_seed1[k] >= m_seed2[k]))
        same_pruned = 0
        for k in m:
            same_pruned += np.sum(m_seed2[k]==m_seed1_2[k])
        same_fraction = same_pruned/total_weights
        self.assertGreaterEqual(0.0001, 1-same_fraction)

    def test_prune_layers_to_ignore(self):
        hparams = PruningHparams('random', 0.2)
        layers_to_ignore = sorted(self.model.prunable_layer_names)[:5]
        hparams.pruning_layers_to_ignore = ','.join(layers_to_ignore)

        m = Strategy.prune(hparams, self.model).numpy()

        # Ensure that the ignored layers were, indeed, ignored.
        for k in layers_to_ignore:
            self.assertTrue(np.all(m[k] == 1))

        # Ensure that the expected fraction of parameters was still pruned.
        total_pruned = np.sum([np.sum(1 - v) for v in m.values()])
        total_weights = np.sum([v.size for v in m.values()])
        actual_fraction = float(total_pruned) / total_weights
        self.assertGreaterEqual(actual_fraction, hparams.pruning_fraction)
        self.assertGreater(hparams.pruning_fraction + 0.0001, actual_fraction)





test_case.main()
