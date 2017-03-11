# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys

__author__ = 'fyabc'


PY3 = sys.version_info.major == 3


if PY3:
    import pickle as pkl
else:
    import cPickle as pkl
