#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import time
import traceback
import platform
import pprint

import numpy as np

from .my_logging import message, finalize_logging_file, init_logging_file, get_logging_file
from .preprocess import preprocess_config
from .config import Config

__author__ = 'fyabc'


def process_before_train(more_args=None, **kwargs):
    args = sys.argv + (more_args or [])

    action = kwargs.pop('action', 'train')

    # Save config only in training
    save_config = action == 'train'

    # Preprocess config.
    preprocess_config(args, save=save_config)

    # Set logging file.
    init_logging_file(append=Config['append'])

    # Set random seed.
    np.random.seed(Config['seed'])

    message('[Message before run]')
    message('Running on node: {}'.format(platform.node()))
    message('Start Time: {}'.format(time.ctime()))

    message('The configuration and hyperparameters are:')
    pprint.pprint(Config, stream=sys.stderr)
    logging_file = get_logging_file()
    if logging_file != sys.stderr:
        pprint.pprint(Config, stream=logging_file)

    message('[Message before run done]')


def process_after_train():
    message('[Message after run]')
    message('End Time: {}'.format(time.ctime()))
    message('[Message after run done]')
    finalize_logging_file()


def real_main(call_table_or_func, more_args=None, **kwargs):
    try:
        process_before_train(more_args, **kwargs)

        if isinstance(call_table_or_func, dict):
            train_func = call_table_or_func.get(Config['type'].lower(), None)

            if train_func is None:
                raise KeyError('Unknown train type {}'.format(Config['type']))

            train_func()
        else:
            call_table_or_func()
    except:
        message(traceback.format_exc())
    finally:
        process_after_train()


__all__ = [
    'process_before_train',
    'process_after_train',
    'real_main',
]
