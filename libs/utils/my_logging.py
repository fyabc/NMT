#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from functools import wraps

from .config import Config

__author__ = 'fyabc'


# Logging settings.
_Indent = 2

# Internal variables.
_logging_file = sys.stderr
_depth = 0


def init_logging_file(append=False):
    global _logging_file

    if Config['logging_file'] is None:
        return

    if append:
        _logging_file = open(Config['logging_file'], 'a')
        return

    # Search for new filename
    root, ext = os.path.splitext(Config['logging_file'])
    i = 1

    filename = Config['logging_file']
    while os.path.exists(filename):
        filename = '{}_{}{}'.format(root, i, ext)
        i += 1

    Config['logging_file'] = filename
    _logging_file = open(filename, 'w')


def finalize_logging_file():
    if _logging_file != sys.stderr:
        _logging_file.flush()
        _logging_file.close()


def get_logging_file():
    return _logging_file


def message(*args, **kwargs):
    to_stderr = kwargs.pop('to_stderr', True)

    if _logging_file != sys.stderr:
        print(*args, file=_logging_file, **kwargs)
    if to_stderr:
        print(*args, file=sys.stderr, **kwargs)


def logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _depth

        message(' ' * _Indent * _depth + '[Start function {}...]'.format(func.__name__))
        _depth += 1
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        _depth -= 1
        message(' ' * _Indent * _depth + '[Function {} done, time: {:.3f}s]'.format(
            func.__name__, end_time - start_time))

        return result
    return wrapper


__all__ = [
    'init_logging_file',
    'finalize_logging_file',
    'get_logging_file',
    'message',
    'logging',
]
