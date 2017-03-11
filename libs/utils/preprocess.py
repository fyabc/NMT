#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
from warnings import warn

from .config import Config

__author__ = 'fyabc'


# The escaped string double quote.
_StringDoubleQuote = '@'
_KeyValueSep = '='
_DotSep = '.'
Tilde = '~'


def _strict_update(key_path, value, config):
    """Update the config value strictly (raise error if key not exists)

    :param key_path: The path of attribute key, a list of strings.
    :param value: The attribute value.
    :param config: The config dict to be updated.
    :return: Nothing
    """

    d = config

    len_key_path = len(key_path)
    try:
        for i, k in enumerate(key_path):
            if i == len_key_path - 1:
                if k not in d:
                    raise KeyError()
                else:
                    d[k] = eval(value)
            else:
                d = d[k]
    except (KeyError, TypeError):
        raise KeyError('The key "{}" is not in the parameters.'.format(_DotSep.join(key_path)))


def parse_args(args=None, config=Config):
    args = args or sys.argv

    for i, arg in enumerate(args):
        arg = arg.replace(_StringDoubleQuote, '"')

        if _KeyValueSep in arg:
            key, value = arg.rsplit(_KeyValueSep, 1)
            key_path = key.split(_DotSep)
            _strict_update(key_path, value, config)
        else:
            if i > 0:
                warn('Warning: The argument "{}" is unused'.format(arg))

    return config


def check_config(config=Config):
    pass


def preprocess_config():
    pass
