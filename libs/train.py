#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from .utils.main import real_main
from .utils.config import Config

__author__ = 'fyabc'


def train_baseline():
    print('Training baseline')
    print(Config)


def train(more_args=None):
    real_main({
        'baseline': train_baseline,
    }, more_args)
