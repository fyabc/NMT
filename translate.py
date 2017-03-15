# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from libs.utils.main import real_main
from libs.utils.path import model_iteration_name
from libs.utils.config import Config as C
from libs.utils.constants import *

__author__ = 'fyabc'


def translate():
    print('Translating model')
    print('Model name:', model_iteration_name(C[K_Model], C[K_StartIteration]))

    if C[K_StartIteration] <= 0:
        print('Do not have model to load')


def main():
    # Can add some args here
    more_args = [

    ]

    real_main(translate, more_args, action='translate')


if __name__ == '__main__':
    main()

