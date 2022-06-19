# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import numpy as np
import os
import torch
import copy
from foundations import paths
from foundations.step import Step
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from testing import test_case
from testing.toy_model import InnerProductModel


class TestPrunedModel(test_case.TestCase):
    def setUp(self):
        super(TestPrunedModel, self).setUp()
        self.model = InnerProductModel(10)
        self.example = torch.ones(1, 10)
        self.label = torch.ones(1) * 40.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def test_mask_with_ones_forward(self):
        mask = Mask.ones_like(self.model)
        pruned_model = PrunedModel(self.model, mask)
        result = pruned_model(self.example).item()
        self.assertEqual(45.0, result)

    def test_mask_with_ones_backward(self):
        mask = Mask.ones_like(self.model)
        pruned_model = PrunedModel(self.model, mask)

        self.optimizer.zero_grad()
        pruned_model.loss_criterion(pruned_model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertEqual(44.0, pruned_model(self.example).item())

    def test_with_missing_mask_value(self):
        mask = Mask.ones_like(self.model)
        del mask['layer.weight']

        with self.assertRaises(ValueError):
            PrunedModel(self.model, mask)

    def test_with_excess_mask_value(self):
        mask = Mask.ones_like(self.model)
        mask['layer2.weight'] = np.ones(20)

        with self.assertRaises(ValueError):
            PrunedModel(self.model, mask)

    def test_with_incorrect_shape(self):
        mask = Mask.ones_like(self.model)
        mask['layer.weight'] = np.ones(30)

        with self.assertRaises(ValueError):
            PrunedModel(self.model, mask)

    def test_with_mask_zero(self):
        mask = Mask()
        mask['layer.weight'] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        pruned_model = PrunedModel(self.model, mask)

        # Check that the forward pass gives the correct value.
        self.assertEqual(20.0, pruned_model(self.example).item())

        # Check that the appropriate weights have been zeroed out.
        self.assertTrue(np.array_equal(self.model.state_dict()['layer.weight'].numpy(),
                                       np.array([0, 0, 2, 0, 4, 0, 6, 0, 8, 0])))

        # Perform a backward pass.
        self.optimizer.zero_grad()
        pruned_model.loss_criterion(pruned_model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertEqual(22.0, pruned_model(self.example).item())
        
        # Verify the weights.
        self.assertTrue(np.allclose(self.model.state_dict()['layer.weight'].numpy(),
                                    np.array([0.4, 0, 2.4, 0, 4.4, 0, 6.4, 0, 8.4, 0])))

    def test_with_mask_final(self):
        mask = Mask()
        mask['layer.weight'] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        model_for_reset = copy.deepcopy(self.model)

        pruned_model = PrunedModel(self.model, mask, model_for_reset, 'final')

        # Check that the forward pass gives the correct value.
        self.assertEqual(45.0, pruned_model(self.example).item())

        # Check that the appropriate weights have been zeroed out.
        self.assertTrue(np.array_equal(self.model.state_dict()['layer.weight'].numpy(),
                                       np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))

        # Perform a backward pass.
        self.optimizer.zero_grad()
        pruned_model.loss_criterion(pruned_model(self.example), self.label).backward()

        self.optimizer.step()
        self.assertEqual(44.5, pruned_model(self.example).item())

        # Verify the weights.
        self.assertTrue(np.allclose(self.model.state_dict()['layer.weight'].numpy(),
                                    np.array([-0.1, 1, 1.9, 3, 3.9, 5, 5.9, 7, 7.9, 9])))

        # Test a second run
        mask['layer.weight'] = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
        model_for_reset = copy.deepcopy(self.model)
        pruned_model = PrunedModel(self.model, mask, model_for_reset, 'final')
        # Check that the forward pass gives the correct value.
        self.assertEqual(44.5, pruned_model(self.example).item())
        # Check that the appropriate weights have been zeroed out.
        self.assertTrue(np.allclose(self.model.state_dict()['layer.weight'].numpy(),
                                    np.array([-0.1, 1, 1.9, 3, 3.9, 5, 5.9, 7, 7.9, 9])))
        # Perform a backward pass.
        self.optimizer.zero_grad()
        pruned_model.loss_criterion(pruned_model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertAlmostEquals(44.23, pruned_model(self.example).item(), 5)
        # Perform a backward pass.
        self.optimizer.zero_grad()
        pruned_model.loss_criterion(pruned_model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertAlmostEquals(43.9762, pruned_model(self.example).item(), 5)
        # Verify the weights.
        self.assertTrue(np.allclose(self.model.state_dict()['layer.weight'].numpy(),
                                    np.array([-0.1, 1, 1.9, 3, 3.7254, 5, 5.7254, 7, 7.7254, 9])))

    def test_with_mask_permuted(self):
        mask = Mask()
        mask['layer.weight'] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        model_for_reset = copy.deepcopy(self.model)

        pruned_model = PrunedModel(self.model, mask, model_for_reset, 'permuted')

        # Check that the forward pass gives the correct value.
        self.assertEqual(45.0, pruned_model(self.example).item())

        # Check that the appropriate weights have been zeroed out.
        self.assertTrue(np.array_equal(self.model.state_dict()['layer.weight'].numpy(),
                                       np.array([0, 5, 2, 3, 4, 1, 6, 9, 8, 7])))

        # Perform a backward pass.
        self.optimizer.zero_grad()
        pruned_model.loss_criterion(pruned_model(self.example), self.label).backward()

        self.optimizer.step()
        self.assertEqual(44.5, pruned_model(self.example).item())

        # Verify the weights.
        self.assertTrue(np.allclose(self.model.state_dict()['layer.weight'].numpy(),
                                    np.array([-0.1, 5, 1.9, 3, 3.9, 1, 5.9, 9, 7.9, 7])))

        # Test a second run
        mask['layer.weight'] = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
        model_for_reset = copy.deepcopy(self.model)
        pruned_model = PrunedModel(self.model, mask, model_for_reset, 'permuted')
        # Check that the forward pass gives the correct value.
        self.assertEqual(44.5, pruned_model(self.example).item())
        # Check that the appropriate weights have been zeroed out.
        self.assertTrue(np.allclose(self.model.state_dict()['layer.weight'].numpy(),
                                    np.array([1, 1.9, 5, -0.1, 3.9, 9, 5.9, 3, 7.9, 7])))
        # Perform a backward pass.
        self.optimizer.zero_grad()
        pruned_model.loss_criterion(pruned_model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertAlmostEquals(44.23, pruned_model(self.example).item(), 5)


    def test_save(self):
        state1 = self.get_state(self.model)

        mask = Mask.ones_like(self.model)
        pruned_model = PrunedModel(self.model, mask)
        pruned_model.save(self.root, Step.zero(20))

        self.assertTrue(os.path.exists(paths.model(self.root, Step.zero(20))))

        self.model.load_state_dict(torch.load(paths.model(self.root, Step.zero(20))))
        self.assertStateEqual(state1, self.get_state(self.model))

    def test_state_dict(self):
        mask = Mask.ones_like(self.model)
        pruned_model = PrunedModel(self.model, mask)

        state_dict = pruned_model.state_dict()
        self.assertEqual(set(['model.layer.weight', 'mask_layer___weight']), state_dict.keys())

    def test_load_state_dict(self):
        mask = Mask.ones_like(self.model)
        mask['layer.weight'] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.model.layer.weight.data = torch.arange(10, 20, dtype=torch.float32)
        pruned_model = PrunedModel(self.model, mask)
        self.assertEqual(70.0, pruned_model(self.example).item())

        self.optimizer.zero_grad()
        pruned_model.loss_criterion(pruned_model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertEqual(67.0, pruned_model(self.example).item())

        # Save the old state dict.
        state_dict = pruned_model.state_dict()

        # Create a new model.
        self.model = InnerProductModel(10)
        mask = Mask.ones_like(self.model)
        pruned_model = PrunedModel(self.model, mask)
        self.assertEqual(45.0, pruned_model(self.example).item())

        # Load the state dict.
        pruned_model.load_state_dict(state_dict)
        self.assertEqual(67.0, pruned_model(self.example).item())

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.optimizer.zero_grad()
        pruned_model.loss_criterion(pruned_model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertEqual(64.3, np.round(pruned_model(self.example).item(), 1))


test_case.main()
