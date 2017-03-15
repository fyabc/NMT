#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os as _os

__author__ = 'fyabc'


Tilde = '~'


# Paths
ProjectRootPath = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
DataPath = _os.path.join(ProjectRootPath, 'data')
LogPath = _os.path.join(ProjectRootPath, 'log')
ModelPath = _os.path.join(ProjectRootPath, 'model')

ConfigFileName = 'config.json'
ConfigFilePath = _os.path.join(ProjectRootPath, ConfigFileName)


# Common configuration keys
K_Name = 'name'
K_Logging = 'logging_file'
K_Model = 'model_file'
K_JobType = 'type'
K_ReloadConfig = 'reload_config'
K_StartIteration = 'start_iteration'

K_DataPath = [
    'data_src',
    'data_tgt',
    'vocab_src',
    'vocab_tgt',
]
