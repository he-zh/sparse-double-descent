# Copyright (c) Facebook, Inc. and its affiliates.

# This code is from OpenLTH repository https://github.com/facebookresearch/open_lth 
# licensed under the MIT license
import os
from foundations import paths
from foundations.step import Step

class MetricLogger:
    def __init__(self):
        self.log = {}

    def add(self, name: str, step: Step, value: float):
        self.log[(name, step.iteration)] = value

    def __str__(self):
        return '\n'.join(['{},{},{}'.format(k[0], k[1], v) for k, v in self.log.items()])

    @staticmethod
    def create_from_string(as_str):
        logger = MetricLogger()
        if len(as_str.strip()) == 0:
            return logger

        rows = [row.split(',') for row in as_str.strip().split('\n')]
        logger.log = {(name, int(iteration)): float(value) for name, iteration, value in rows}
        return logger

    @staticmethod
    def create_from_file(filename):
        with open(paths.logger(filename)) as fp:
            as_str = fp.read()
        return MetricLogger.create_from_string(as_str)

    def save(self, location):
        if not os.path.exists(location):
            os.makedirs(location)
        with open(paths.logger(location), 'w') as fp:
            fp.write(str(self))

    def get_data(self, desired_name):
        d = {k[1]: v for k, v in self.log.items() if k[0] == desired_name}
        return [(k, d[k]) for k in sorted(d.keys())]
