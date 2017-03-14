#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import json
import re

from .constants import ConfigFilePath

__author__ = 'fyabc'


def load_config(filename):
    """Load JSON config file and remove line comments."""

    with open(filename, 'r') as f:
        _lines = list(f)

        for _i, _line in enumerate(_lines):
            _lines[_i] = re.sub(r'//.*\n', '\n', _line)

        return json.loads(''.join(_lines))


def save_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

Config = load_config(ConfigFilePath)
