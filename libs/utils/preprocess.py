#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os

from .args import parse_args
from .path import silent_mkdir, find_newest_model, split_model_name
from .config import Config, load_config, save_config
from .constants import *

__author__ = 'fyabc'


def _check_config():
    pass


def _load_save_config(this_model_path, save=True):
    """Load previous saved config."""
    if Config[K_ReloadConfig]:
        previous_config_filename = os.path.join(this_model_path, ConfigFileName)
        if os.path.exists(previous_config_filename):
            # Do not clear new config, just overwrite it.
            # Config.clear()
            Config.update(load_config(previous_config_filename))

    if save:
        save_config(Config, os.path.join(this_model_path, ConfigFileName))


def _parse_job_name():
    job_name = Config[K_Name]

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
    Config[K_Logging] = os.path.join(this_log_path, Config[K_Logging])

    if Config[K_StartIteration] is None:
        # Load newest model
        name, _, ext = split_model_name(Config[K_Model])

        # If there is not any models, this will be -1
        Config[K_StartIteration] = find_newest_model(this_model_path, name, ext, ret_filename=False)

    if Config[K_StartIteration] <= 0:
        # Restart model
        Config[K_StartIteration] = 0

    Config[K_Model] = os.path.join(this_model_path, Config[K_Model])

    for key in K_DataPath:
        Config[key] = Config[key].replace(Tilde, DataPath)


def preprocess_config(args=None, **kwargs):
    if '-h' in args or '--help' in args:
        print('See comments of file "config.json" to know how to set arguments.')
        exit(0)

    parse_args(args)

    # Set job name.
    this_model_path, this_log_path = _parse_job_name()

    # Reload config if needed, then save current config
    _load_save_config(this_model_path, save=kwargs.pop('save', True))

    # Check config.
    _check_config()

    # Replace path.
    _replace_path(this_model_path, this_log_path)
