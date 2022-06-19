
import numpy as np
import sys
sys.path.append("./")
from torch.cuda import device, init
import models.registry
from pruning.network_slimming import Strategy
from pruning.network_slimming import PruningHparams
from testing import test_case


class TestNetworkSlimming(test_case.TestCase):
    def setUp(self):
        super(TestNetworkSlimming, self).setUp()
        self.hparams_global = PruningHparams('network_slimming', pruning_fraction=0.2)
        model_hparams = models.registry.get_default_hparams('cifar_resnet_20').model_hparams
        self.model = models.registry.get(model_hparams, 10, 'network_slimming')

    def test_get_pruning_hparams(self):
        self.assertTrue(issubclass(Strategy.get_pruning_hparams(), PruningHparams))

    def test_globally_prune(self):
        
        m = Strategy.prune(self.hparams_global, self.model)

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
        self.assertGreaterEqual(actual_fraction, self.hparams_global.pruning_fraction)
        self.assertGreater(self.hparams_global.pruning_fraction + 0.01, actual_fraction)

        # Ensure that the right threshold was chosen.
        pruned_weights = [self.model.state_dict()[k].numpy()[m[k] == 0] for k in m if 'weight' in k]
        threshold = np.max(np.abs(np.concatenate(pruned_weights)))
        unpruned_weights = [self.model.state_dict()[k].numpy()[m[k] == 1] for k in m if 'weight' in k]
        self.assertTrue(np.all(np.abs(np.concatenate(unpruned_weights)) > threshold))

        # Ensure that biases are pruned along with weights
        for k in m:
            if 'weight' in k:
                self.assertTrue(np.all(m[k] == m[k.replace('weight', 'bias')]))

    def test_globally_iterative_pruning(self):
        m = Strategy.prune(self.hparams_global, self.model)
        m2 = Strategy.prune(self.hparams_global, self.model, m)

        # Ensure that all weights pruned before are still pruned here.
        m, m2 = m.numpy(), m2.numpy()
        self.assertEqual(set(m.keys()), set(m2.keys()))
        for k in m:
            self.assertTrue(np.all(m[k] >= m2[k]))

        total_pruned = np.sum([np.sum(1 - v) for v in m2.values()])
        total_weights = np.sum([v.size for v in m2.values()])
        actual_fraction = float(total_pruned) / total_weights
        expected_fraction = 1 - (1 - self.hparams_global.pruning_fraction) ** 2
        self.assertGreaterEqual(actual_fraction, expected_fraction)
        self.assertGreater(expected_fraction + 0.01, actual_fraction)


    # def test_layer_wise_prune(self):
    #     m = Strategy.prune(self.hparams_layer, self.model)

    #     # Check that the mask only contains entries for the prunable layers.
    #     self.assertEqual(set(m.keys()), set(self.model.prunable_layer_names))

    #     # Check that the masks are the same sizes as the tensors.
    #     for k in self.model.prunable_layer_names:
    #         self.assertEqual(list(m[k].shape), list(self.model.state_dict()[k].shape))

    #     # Check that the right fraction of weights was pruned among each prunable layer.
    #     m = m.numpy()
    #     for k in m:
    #         layer_pruned = np.sum(1 - m[k])
    #         layer_weights = np.sum(m[k].size)
    #         layer_fraction = float(layer_pruned) / layer_weights
    #         self.assertGreaterEqual(layer_fraction, self.hparams_layer.pruning_fraction)
    #         self.assertGreater(self.hparams_layer.pruning_fraction + 0.1, layer_fraction)

    #         # Ensure that the right threshold was chosen.
    #         pruned_weights = self.model.state_dict()[k].numpy()[m[k] == 0]
    #         threshold = np.max(np.abs(pruned_weights))
    #         unpruned_weights = self.model.state_dict()[k].numpy()[m[k] == 1]
    #         self.assertTrue(np.all(np.abs(unpruned_weights) > threshold))
        
    #     # Check that the right fraction of weights was pruned among all prunable layers.
    #     total_pruned = np.sum([np.sum(1 - v) for v in m.values()])
    #     total_weights = np.sum([v.size for v in m.values()])
    #     actual_fraction = float(total_pruned) / total_weights
    #     self.assertGreaterEqual(actual_fraction, self.hparams_layer.pruning_fraction)
    #     self.assertGreater(self.hparams_layer.pruning_fraction + 0.001, actual_fraction)

    # def test_layer_wise_iterative_pruning(self):
    #     m = Strategy.prune(self.hparams_layer, self.model)
    #     m2 = Strategy.prune(self.hparams_layer, self.model, m)

    #     # Ensure that all weights pruned before are still pruned here.
    #     m, m2 = m.numpy(), m2.numpy()
    #     self.assertEqual(set(m.keys()), set(m2.keys()))
    #     for k in m:
    #         self.assertTrue(np.all(m[k] >= m2[k]))

    #     for k in m:
    #         layer_pruned = np.sum(1 - m2[k])
    #         layer_weights = np.sum(m2[k].size)
    #         layer_fraction = float(layer_pruned) / layer_weights
    #         expected_fraction = 1 - (1 - self.hparams_layer.pruning_fraction) ** 2
    #         self.assertGreaterEqual(layer_fraction, expected_fraction)
    #         # self.assertGreater(expected_fraction + 0.0001, layer_fraction)
    #     total_pruned = np.sum([np.sum(1 - v) for v in m2.values()])
    #     total_weights = np.sum([v.size for v in m2.values()])
    #     actual_fraction = float(total_pruned) / total_weights
    #     expected_fraction = 1 - (1 - self.hparams_layer.pruning_fraction) ** 2
    #     self.assertGreaterEqual(actual_fraction, expected_fraction)
    #     self.assertGreater(expected_fraction + 0.001, actual_fraction)

    def test_globally_prune_layers_to_ignore(self):
        layers_to_ignore = sorted(self.model.prunable_layer_names)[:4]
        self.hparams_global.pruning_layers_to_ignore = ','.join(layers_to_ignore)

        m = Strategy.prune(self.hparams_global, self.model).numpy()

        # Ensure that the ignored layers were, indeed, ignored.
        for k in layers_to_ignore:
            self.assertTrue(np.all(m[k] == 1))

        # Ensure that the expected fraction of parameters was still pruned.
        total_pruned = np.sum([np.sum(1 - v) for v in m.values()])
        total_weights = np.sum([v.size for v in m.values()])
        actual_fraction = float(total_pruned) / total_weights
        self.assertGreaterEqual(actual_fraction, self.hparams_global.pruning_fraction)
        self.assertGreater(self.hparams_global.pruning_fraction + 0.01, actual_fraction)

    # def test_layer_wise_prune_layers_to_ignore(self):
    #     layers_to_ignore = sorted(self.model.prunable_layer_names)[:5]
    #     self.hparams_layer.pruning_layers_to_ignore = ','.join(layers_to_ignore)

    #     m = Strategy.prune(self.hparams_layer, self.model).numpy()

    #     # Ensure that the ignored layers were, indeed, ignored.
    #     for k in layers_to_ignore:
    #         self.assertTrue(np.all(m[k] == 1))

    #     # Ensure that the expected fraction of parameters was still pruned.
    #     total_pruned = np.sum([np.sum(1 - v) for v in m.values()])
    #     total_weights = np.sum([v.size for v in m.values()])
    #     actual_fraction = float(total_pruned) / total_weights
    #     self.assertGreaterEqual(self.hparams_layer.pruning_fraction, actual_fraction )





test_case.main()
