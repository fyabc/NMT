#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

from libs.train import train_main
from libs.utils.main import real_main

__author__ = 'fyabc'


def main():
    # Can add some args here
    more_args = [

    ]

    real_main({
        'baseline': train_main,
    }, more_args)


if __name__ == '__main__':
    main()
