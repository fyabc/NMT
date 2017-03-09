#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys

from .config import Config

__author__ = 'fyabc'


def parse_args(args=None, config=Config):
    args = args or sys.argv

    for i, arg in enumerate(args):
        pass


def preprocess_config():
    pass
