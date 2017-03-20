#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os

from .args import parse_args
from .path import silent_mkdir, find_newest_model, split_model_name
from .config import C, load_config, save_config
from .constants import *

__author__ = 'fyabc'


def _add_generated_config(**kwargs):
    """Add generated config.

    Generated config are start with `GeneratedPrefix`.
    """

    for k, v in kwargs.iteritems():
        C[Gen_(k)] = v


def _check_config():
    """Check for conflict config."""

    pass


def _load_save_config(this_model_path):
    """Load previous saved config."""
    if C[K_ReloadConfig]:
        previous_config_filename = os.path.join(this_model_path, ConfigFileName)
        if os.path.exists(previous_config_filename):
            # Do not clear new config, just overwrite it.
            # C.clear()
            C.update(load_config(previous_config_filename))

    if C[Gen_('save')]:
        save_config(C, os.path.join(this_model_path, ConfigFileName))


def _parse_job_name():
    job_name = C[K_Name]

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
    """Replace some paths.

    [NOTE]: This will modify the config.
    """

    C[K_Logging] = os.path.join(this_log_path, C[K_Logging])

    worker_id = C[Gen_('worker_id')]
    # todo: Parse worker id to replace data path and logging path.

    if C[K_StartIteration] is None:
        # Load newest model
        name, _, ext = split_model_name(C[K_Model])

        # If there is not any models, this will be -1
        C[K_StartIteration] = find_newest_model(this_model_path, name, ext, ret_filename=False)

    if C[K_StartIteration] <= 0:
        # Restart model
        C[K_StartIteration] = 0

    C[K_Model] = os.path.join(this_model_path, C[K_Model])

    for key in K_DataPath:
        C[key] = C[key].replace(Tilde, DataPath)


def preprocess_config(args=None, **kwargs):
    """Preprocess the args to config dict, and do some other actions, such as replace path.

    :param args: arguments to be parsed.
    :param kwargs: Generated config.
    """

    if '-h' in args or '--help' in args:
        print('See comments of file "config.json" to know how to set arguments.')
        exit(0)

    parse_args(args)

    # Add generated config.
    _add_generated_config(**kwargs)

    # Set job name.
    this_model_path, this_log_path = _parse_job_name()

    # Reload config if needed, then save current config
    # [NOTE] This must before the call of `_replace_path`.
    _load_save_config(this_model_path)

    # Check config.
    _check_config()

    # Replace path.
    _replace_path(this_model_path, this_log_path)
