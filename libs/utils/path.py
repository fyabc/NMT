#! /usr/bin/python
# -*- encoding: utf-8 -*-

#################
# File and path #
#################

from __future__ import print_function, unicode_literals

import errno
import gzip
import os
import fnmatch

from ._compat import pkl

__author__ = 'fyabc'


def f_open(filename, mode='rb', unpickle=True):
    if filename.endswith('.gz'):
        _open = gzip.open
    else:
        _open = open

    if unpickle:
        with _open(filename, 'rb') as f:
            return pkl.load(f)
    else:
        return open(filename, mode)


def silent_mkdir(*paths):
    """Make directories silently, do not raise error if exists."""

    for path in paths:
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def split_model_name(model_name):
    tmp, ext = os.path.splitext(model_name)
    name, iteration = os.path.splitext(tmp)

    # Remove extra dot
    if iteration:
        iteration = iteration[1:]

    return name, iteration, ext


def find_newest_model(dir_name, raw_name, ext='.npz', ret_filename=False):
    """Find the newest model of current name.

    Model name format: xxx.4.npz
    """

    max_number = -1
    newest_filename = ''

    pattern = '{}.*{}'.format(os.path.basename(raw_name), ext)

    for filename in os.listdir(dir_name):
        if fnmatch.fnmatch(filename, pattern):
            name, iteration, ext = split_model_name(filename)

            try:
                iteration = int(iteration)
            except ValueError:
                continue

            if iteration > max_number:
                max_number = iteration
                newest_filename = filename

    newest_filename = os.path.join(dir_name, newest_filename)

    if ret_filename:
        return max_number, newest_filename
    return max_number


def model_iteration_name(model_name, iteration):
    root, ext = os.path.splitext(model_name)
    return '{}.{}{}'.format(root, iteration, ext)
