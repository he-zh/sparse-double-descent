# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import copy
import json
import numpy as np
import os
import torch

import datasets.registry
from foundations import paths
from foundations.step import Step
from experiments.scratch.runner import ScratchRunner
from experiments.scratch.desc import ScratchDesc
import models.registry
from pruning.mask import Mask
from testing import test_case


class TestRunner(test_case.TestCase):
    def setUp(self):
        super(TestRunner, self).setUp()
        desc = models.registry.get_default_hparams('cifar_resnet_8_2')
        self.desc = ScratchDesc(desc.model_hparams, desc.dataset_hparams, desc.training_hparams, desc.pruning_hparams)

    def to_step(self, s):
        return Step.from_str(s, datasets.registry.iterations_per_epoch(self.desc.dataset_hparams))

    def assertLevelFilesPresent(self, level_root, start_step, end_step, masks=False):
        with self.subTest(level_root=level_root):
            self.assertTrue(os.path.exists(paths.model(level_root, start_step)))
            self.assertTrue(os.path.exists(paths.model(level_root, end_step)))
            self.assertTrue(os.path.exists(paths.logger(level_root)))
            if masks:
                self.assertTrue(os.path.exists(paths.mask(level_root)))
                self.assertTrue(os.path.exists(paths.sparsity_report(level_root)))

    def test_level0_2it(self):
        self.desc.training_hparams.training_steps = '2it'
        ScratchRunner(replicate=2, levels=0, desc=self.desc, verbose=False).run()
        level_root = self.desc.run_path(2, 0)

        # Ensure the important files are there.
        self.assertLevelFilesPresent(level_root, self.to_step('0it'), self.to_step('2it'))

        # Ensure that the mask is all 1's.
        mask = Mask.load(level_root)
        for v in mask.numpy().values(): self.assertTrue(np.all(np.equal(v, 1)))
        with open(paths.sparsity_report(level_root)) as fp:
            sparsity_report = json.loads(fp.read())
        self.assertEqual(sparsity_report['unpruned'] / sparsity_report['total'], 1)
    

    def test_level3_2it(self):
        self.desc.training_hparams.training_steps = '2it'
        ScratchRunner(replicate=2, levels=3, desc=self.desc, verbose=False).run()

        level0_weights = paths.model(self.desc.run_path(2, 0), self.to_step('0it'))
        level0_weights = {k: v.cpu().numpy() for k, v in torch.load(level0_weights).items()}

        for level in range(0, 4):
            level_root = self.desc.run_path(2, level)
            self.assertLevelFilesPresent(level_root, self.to_step('0it'), self.to_step('2it'))

            # Check the mask.
            pct = 0.8**level
            mask = Mask.load(level_root).numpy()

            # Check the mask itself.
            total, total_present = 0.0, 0.0
            for v in mask.values():
                total += v.size
                total_present += np.sum(v)
            self.assertTrue(np.allclose(pct, total_present / total, atol=0.01))

            # Check the sparsity report.
            with open(paths.sparsity_report(level_root)) as fp:
                sparsity_report = json.loads(fp.read())
            self.assertTrue(np.allclose(pct, sparsity_report['unpruned'] / sparsity_report['total'], atol=0.01))

            # Ensure that initial weights of each level are different from the original initialization.
            if level != 0:
                level_weights = paths.model(level_root, self.to_step('0it'))
                level_weights = {k: v.cpu().numpy() for k, v in torch.load(level_weights).items()}
                for k in level0_weights:
                    if 'weight' in k:
                        self.assertFalse((level_weights[k]==level0_weights[k] * mask.get(k, 1)).all())

            # Ensure that initial weights of each level are different from the initial weights of the previous level.
            if level != 0:
                previous_level_start_weights = paths.model(self.desc.run_path(2, level-1), self.to_step('0it'))
                previous_level_start_weights = {k: v.cpu().numpy() for k, v in torch.load(previous_level_start_weights).items()}
                for k in level_weights:
                    if 'weight' in k:
                        self.assertFalse((level_weights[k]==previous_level_start_weights[k] * mask.get(k, 1)).all())
                # self.assertStateAllNotEqual(level_weights, {k: v * mask.get(k, 1) for k, v in previous_level_end_weights.items()})
        


test_case.main()
