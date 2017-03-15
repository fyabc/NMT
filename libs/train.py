#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from .data_tool.text_iterator import TextIterator
from .model_class.baseline import BaselineModel

from .utils.config import C
from .utils.constants import *
from .utils.path import model_iteration_name
from .utils.my_logging import message

__author__ = 'fyabc'


def train_baseline():
    print('Training baseline')

    start_iteration = C[K_StartIteration]

    print(start_iteration, C['max_iteration'])
    print(model_iteration_name(C[K_Model], start_iteration))

    text_iterator = TextIterator(
        C['data_src'],
        C['data_tgt'],
        C['vocab_src'],
        C['vocab_tgt'],
        C['batch_size'],
        C['maxlen'],
        C['n_words_src'],
        C['n_words_tgt'],
    )

    if start_iteration <= 0:
        load = None
    else:
        load = {'filename': C[K_Model], 'iteration': start_iteration}

    model = BaselineModel(load=load)
