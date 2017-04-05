# -*- coding: utf-8 -*-

"""Some basic utilities."""

from __future__ import print_function

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


def p_(*args):
    """Get the name of tensor with the prefix (layer name) and variable name(s)."""

    return '_'.join(str(arg) for arg in args)


def slice_(_x, n, _dim):
    """Utility function to slice a tensor."""

    if _x.ndim == 3:
        return _x[:, :, n * _dim:(n + 1) * _dim]
    return _x[:, n * _dim:(n + 1) * _dim]


__all__ = [
    'fX',
    'floatX',
    'load_list',
    'save_list',
    'p_',
    'slice_',
]
