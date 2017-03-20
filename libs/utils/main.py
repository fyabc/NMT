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
from .config import C
from .. import multiverso as mv

__author__ = 'fyabc'


def process_before_train(more_args=None, **kwargs):
    args = sys.argv + (more_args or [])

    # Initialize multiverso and get worker id
    mv.init(sync=True)
    worker_id = mv.worker_id()
    is_master = mv.is_master_worker()

    action = kwargs.pop('action', 'train')

    # Save config only in training, and only saved by master
    save_config = action == 'train' and is_master

    # Preprocess config.
    preprocess_config(
        args,
        save=save_config,
        worker_id=worker_id,
        is_master=is_master,
        action=action,
    )

    # Set logging file.
    init_logging_file(append=C['append'])

    # Set random seed.
    np.random.seed(C['seed'])

    # Set recursion limit.
    sys.setrecursionlimit(10000)

    message('''\
[Message before run]
Job name: {}
Worker id: {}
Running on node: {}
Start Time: {}
The configuration and hyperparameters are:
'''.format(
        C['job_name'],
        worker_id,
        platform.node(),
        time.ctime(),
    ))

    pprint.pprint(C, stream=sys.stderr)
    logging_file = get_logging_file()
    if logging_file != sys.stderr:
        pprint.pprint(C, stream=logging_file)

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
            train_func = call_table_or_func.get(C['type'].lower(), None)

            if train_func is None:
                raise KeyError('Unknown train type {}'.format(C['type']))

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
