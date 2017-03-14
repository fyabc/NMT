#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os

from .args import parse_args
from .path import silent_mkdir, find_newest_model
from .config import Config, load_config, save_config
from .constants import *

__author__ = 'fyabc'


def check_config(config=Config):
    pass


def _load_save_config(this_model_path):
    """Load previous saved config."""
    if Config[ReloadConfig]:
        previous_config_filename = os.path.join(this_model_path, ConfigFileName)
        if os.path.exists(previous_config_filename):
            Config.clear()
            Config.update(load_config(previous_config_filename))

    # Save config file
    save_config(Config, os.path.join(this_model_path, ConfigFileName))


def _parse_job_name():
    job_name = Config[JobName]

    if job_name is None:
        raise ValueError('Must set job name')

    # Create directories if not exist.
    this_model_path = os.path.join(ModelPath, job_name)
    this_log_path = os.path.join(LogPath, job_name)

    silent_mkdir(
        this_model_path,
        this_log_path,
    )

    return this_model_path, this_log_path


def _replace_path(this_model_path, this_log_path):
    Config[LoggingFile] = os.path.join(this_log_path, Config[LoggingFile])

    # todo: reload (train from newest model) or restart
    start_iteration = Config[StartIteration]

    if start_iteration is None:
        # Load newest model
        pass
    elif start_iteration < 0:
        # Restart model
        pass
    else:
        pass

    Config[ModelFile] = os.path.join(this_model_path, Config[ModelFile])


def preprocess_config(args=None):
    if '-h' in args or '--help' in args:
        print('See comments of file "config.json" to know how to set arguments.')
        exit(0)

    parse_args(args)

    # todo: modify some config

    # Set job name.
    this_model_path, this_log_path = _parse_job_name()

    # Reload config if needed, then save current config
    _load_save_config(this_model_path)

    # Replace path.
    _replace_path(this_model_path, this_log_path)
