#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os
import json
import re

__author__ = 'fyabc'


# Paths
ProjectRootPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DataPath = os.path.join(ProjectRootPath, 'data')
LogPath = os.path.join(ProjectRootPath, 'log')
ModelPath = os.path.join(ProjectRootPath, 'model')


# Load JSON config file and remove line comments
_lines = list(open(os.path.join(ProjectRootPath, 'config.json'), 'r'))

for _i, _line in enumerate(_lines):
    _lines[_i] = re.sub(r'//.*\n', '\n', _line)

Config = json.loads(''.join(_lines))
