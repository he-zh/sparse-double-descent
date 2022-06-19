# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license

import numpy as np
import os
import unittest
import shutil
from foundations.local import Platform

class Platform(Platform):
    @property
    def device_str(self):
        return 'cpu'

    @property
    def is_parallel(self):
        return False

    @property
    def root(self):
        return '/data/hezheng/pruning-robustness/TESTING'


class TestCase(unittest.TestCase):
    def setUp(self):
        platform = Platform()
        self.root = platform.root

    # def tearDown(self):
    #     if os.path.exists(self.root): shutil.rmtree(self.root)
    #     platforms.platform._PLATFORM = self.saved_platform

    @staticmethod
    def get_state(model):
        """Get a copy of the state of a model."""

        return {k: v.clone().detach().cpu().numpy() for k, v in model.state_dict().items()}

    def assertStateEqual(self, state1, state2):
        """Assert that two models states are equal."""

        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for k in state1:
            self.assertTrue(np.array_equal(state1[k], state2[k]))

    def assertStateAllNotEqual(self, state1, state2):
        """Assert that two models states are not equal in any tensor."""

        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for k in state1:
            self.assertFalse(np.array_equal(state1[k], state2[k]))


def main():
    if __name__ == '__main__':
        unittest.main()
