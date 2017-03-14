#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from .utils.config import Config as C
from .utils.constants import *

__author__ = 'fyabc'


def train_baseline():
    print('Training baseline')

    for iteration in range(C[StartIteration] + 1, C['max_iteration']):
        pass

    print(C[StartIteration], C['max_iteration'])

    if C[StartIteration] <= 0:
        print('Do not have model to load')
