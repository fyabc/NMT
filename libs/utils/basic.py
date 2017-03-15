# -*- coding: utf-8 -*-

"""Some basic utilities."""

from __future__ import print_function, unicode_literals

import os

import numpy as np

from .config import Config

__author__ = 'fyabc'

# The float type of Theano. Default to 'float32'.
# fX = config.floatX
fX = Config['floatX']


def floatX(value):
    return np.asarray(value, dtype=fX)


###############################
# Data loading and processing #
###############################
def load_list(filename, dtype=float):
    if not os.path.exists(filename):
        return []

    with open(filename, 'r') as f:
        return [dtype(l.strip()) for l in f]


def save_list(l, filename):
    with open(filename, 'w') as f:
        for i in l:
            f.write(str(i) + '\n')
