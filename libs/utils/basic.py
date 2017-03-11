# -*- coding: utf-8 -*-

"""Some basic utilities."""

from __future__ import print_function, unicode_literals

import os
import gzip
from ._compat import pkl

import numpy as np

from .config import Config

__author__ = 'fyabc'


# The float type of Theano. Default to 'float32'.
# fX = config.floatX
fX = Config['floatX']


def floatX(value):
    return np.asarray(value, dtype=fX)


def f_open(filename, mode='rb', unpickle=True):
    if filename.endswith('.gz'):
        _open = gzip.open
    else:
        _open = open

    if unpickle:
        with _open(filename, 'rb') as f:
            return pkl.load(f)
    else:
        return open(filename, mode)


def average(sequence):
    if sequence is None:
        return 0.0
    if len(sequence) == 0:
        return 0.0
    return sum(sequence) / len(sequence)


def get_rank(a):
    """Get the rank of numpy array a.

    >>> import numpy as np
    >>> get_rank(np.array([10, 15, -3, 9, 1]))
    array([3, 4, 0, 2, 1])
    """

    temp = a.argsort()
    ranks = np.empty_like(a)
    ranks[temp] = np.arange(len(a))

    return ranks


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
