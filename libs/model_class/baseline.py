#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from collections import OrderedDict

from .base import Model

__author__ = 'fyabc'


class BaselineModel(Model):
    """The baseline model.

    [BLEU] ? at iteration ?
    """

    def init_np_parameters(self):
        np_params = OrderedDict()

        return np_params

    def build_model(self):
        pass
