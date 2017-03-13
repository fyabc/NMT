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


def _parse_job_name():
    job_name = Config['job_name']

    if job_name is None:
        return


def process_before_train(more_args=None):
    args = sys.argv + (more_args or [])

    preprocess_config(args)

    # Set job name.
    _parse_job_name()

    # todo: other settings, e.g. set logging file

    # Set logging file.
    init_logging_file(append=Config['append'])

    # Set random seed.
    np.random.seed(Config['seed'])

    message('[Message before train]')
    message('Running on node: {}'.format(platform.node()))
    message('Start Time: {}'.format(time.ctime()))

    message('The configuration and hyperparameters are:')
    pprint.pprint(Config, stream=sys.stderr)
    logging_file = get_logging_file()
    if logging_file != sys.stderr:
        pprint.pprint(Config, stream=logging_file)

    message('[Message before train done]')


def process_after_train():
    message('[Message after train]')
    message('End Time: {}'.format(time.ctime()))
    message('[Message after train done]')
    finalize_logging_file()


def real_main(call_table, more_args=None):
    process_before_train(more_args)

    try:
        train_func = call_table.get(Config['type'].lower(), None)

        if train_func is None:
            raise KeyError('Unknown train type {}'.format(Config['type']))

        train_func()
    except:
        message(traceback.format_exc())
    finally:
        process_after_train()


__all__ = [
    'process_before_train',
    'process_after_train',
    'real_main',
]
