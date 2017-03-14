#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os as _os

__author__ = 'fyabc'


# Paths
ProjectRootPath = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
DataPath = _os.path.join(ProjectRootPath, 'data')
LogPath = _os.path.join(ProjectRootPath, 'log')
ModelPath = _os.path.join(ProjectRootPath, 'model')

ConfigFileName = 'config.json'
ConfigFilePath = _os.path.join(ProjectRootPath, ConfigFileName)


# Common configuration names
JobName = 'name'
LoggingFile = 'logging_file'
ModelFile = 'model_file'
JobType = 'type'
ReloadConfig = 'reload_config'
StartIteration = 'start_iteration'
