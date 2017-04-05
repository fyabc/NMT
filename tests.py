# -*- coding: utf-8 -*-

from __future__ import print_function

from pprint import pprint

from libs.utils.config import Config

__author__ = 'fyabc'


def test_config():
    pprint(Config)


def test_preprocess():
    from libs.utils.args import parse_args

    parse_args(None, Config)

    pprint(Config)


if __name__ == '__main__':
    # test_config()
    test_preprocess()
    pass
