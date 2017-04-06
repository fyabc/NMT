#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""The real training process here."""

from __future__ import print_function

import os
import sys
import re
import time
import pprint
import numpy as np

from .data_tool.text_iterator import TextIterator
from .data_tool.prepare_data import prepare_data
from .model_class.LstmWithFastFw import LstmWithFastFwModel

from .utils._compat import pkl
from .utils.config import C
from .utils.constants import *
from .utils.my_logging import message, get_logging_file
from .utils.optimizers import get_stepsgd_lr

from . import multiverso as mv

__author__ = 'fyabc'


def train_main():
    os.system(r'"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"')
    sys.stdout.flush()

    C[Gen_('main_net_structure')], C[Gen_('to_upper_layer')] = re.split(r'with', C['network_style'])
    # Very stupid code
    if C[Gen_('main_net_structure')] == 'LSTM':
        C[Gen_('main_net_structure')] = C['network_style']

    # Reload model options.
    # [NOTE] Old options will be stored into ``C``, and ``TO`` will not be changed.
    TO = C.copy()
    if C['reload_'] and os.path.exists(C[K_Model] + '.pkl'):
        message('Reloading model options')
        with open(C[K_Model] + '.pkl', 'rb') as f:
            old_options = pkl.load(f)
            C.update(old_options)

    message('Top options:')
    pprint.pprint(TO)
    pprint.pprint(TO, stream=get_logging_file())
    message('Reloaded options:')
    pprint.pprint(C)
    pprint.pprint(C, stream=get_logging_file())

    message('\n\n\nStart to prepare data\n@Current Time =', str(time.time()))

    print('Loading data')
    train = TextIterator(
        TO['data_src'],
        TO['data_tgt'],
        TO['vocab_src'],
        TO['vocab_tgt'],
        TO['batch_size'],
        TO['maxlen'],
        TO['n_words_src'],
        TO['n_words'],
    )
    train_size = TrainSize

    valid = TextIterator(
        TO['valid_src'],
        TO['valid_tgt'],
        TO['vocab_src'],
        TO['vocab_tgt'],
        TO['valid_batch_size'],
        TO['maxlen'],
        TO['n_words_src'],
        TO['n_words'],
    )

    model = LstmWithFastFwModel(top_options=TO)

    # Training start here
    print('Optimization')
    message('Preparation Done\n@Current Time =', str(time.time()))
    mv.barrier()

    best_p = None
    bad_counter = 0
    iteration = 0
    estop = False
    history_errs = []

    # reload history
    reload_ = TO['reload_']
    saveto = TO[K_Model]
    if reload_ and os.path.exists(saveto):
        old_model = np.load(saveto)
        history_errs = list(old_model['history_errs'])
        if 'uidx' in old_model:
            iteration = old_model['uidx']

    batch_size = TO['batch_size']
    validFreq, saveFreq, sampleFreq = TO['validFreq'], TO['saveFreq'], TO['sampleFreq']
    if validFreq == -1:
        validFreq = train_size / batch_size
    if saveFreq == -1:
        saveFreq = train_size / batch_size
    if sampleFreq == -1:
        sampleFreq = train_size / batch_size

    maxlen = TO['maxlen']
    lrate = TO['lrate']
    optimizer = TO['optimizer']

    for epoch in xrange(TO['max_epochs']):
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            iteration += 1
            x, x_mask, y, y_mask = prepare_data(
                x, y, maxlen=maxlen, n_words_src=TO['n_words_src'], n_words=TO['n_words'])

            if x is None:
                print('Minibatch with zero sample under length', maxlen)
                iteration -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = model.f_grad_shared(x, x_mask, y, y_mask)

            # do the update on parameters
            cur_lr = lrate
            if 'sgd' in optimizer:
                cur_lr = get_stepsgd_lr(lrate, iteration, batch_size)

            model.f_update(cur_lr)

            ud = time.time() - ud_start

            # todo
