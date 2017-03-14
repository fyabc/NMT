#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from libs.train import train_baseline
from libs.translate import translate

from libs.utils.main import real_main

__author__ = 'fyabc'


def main():
    # Can add some args here
    more_args = [

    ]

    real_main({
        'baseline': train_baseline,
        'translate': translate,
    }, more_args)


if __name__ == '__main__':
    main()
