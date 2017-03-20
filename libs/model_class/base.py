#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import theano

from ..utils.path import model_iteration_name
from ..utils.config import C
from ..utils.constants import *
from ..utils.my_logging import logging
from ..utils.name_register import NameRegister
from ..multiverso.theano_ext import sharedvar

__author__ = 'fyabc'


class Model(NameRegister):
    NameTable = {}

    # FIXME: The duplicated shared variable is to fix the bug in multiverso.
    # These variables will cause error when trained on more than one nodes without duplicate.
    # From v-yixia
    DuplicateSharedVarList = ['decoder_Ub_att']
    DuplicateSize = 100

    def __init__(self, load=None):
        """
        :param load: None or a dict that contains the parameters of ``self.load``.
        """

        # Parameters of theano shared variables
        self.parameters = OrderedDict()

        # Duplicate parameters of theano shared variables.
        self.dup_parameters = OrderedDict()

        # Learning rate.
        self.learning_rate = None

        # Theano functions.
        self.f_grad_shared = None
        self.f_update = None

        if load is None:
            self.init_parameters(self.init_np_parameters())
        else:
            self.load(**load)

        self.build_model()

    def f_train(self, x, x_mask, y, y_mask):
        """The train function.

        :param x: source sentence
        :param x_mask: source sentence
        :param y: target sentence
        :param y_mask: target sentence

        Go to ``data_tool.prepare_data.prepare_data`` to see parameter types.

        :return: loss value
        """

        if x.shape[1] == 0:
            return None

        loss = self.f_grad_shared(x, x_mask, y, y_mask)
        self.f_update(self.learning_rate)

        return loss

    def init_np_parameters(self):
        """Initialize numpy parameters (values of self.parameters).

        :return np_params: OrderedDict
            dict of numpy values
        """

        raise NotImplementedError()

    def init_parameters(self, np_parameters):
        """Initialize Theano tensor parameters.

        :param np_parameters: OrderedDict
            dict of numpy values, output of self.init_np_parameters
        """

        for name, value in np_parameters.iteritems():
            if name in self.DuplicateSharedVarList:
                self.parameters[name] = theano.shared(value, name=name)
                self.dup_parameters[name] = sharedvar.mv_shared(
                    value=np.ones(self.DuplicateSize) * value[0],
                    name=name,
                    borrow=False,
                )
            else:
                self.parameters[name] = sharedvar.mv_shared(
                    value=np_parameters[name],
                    name=name,
                    borrow=False,
                )

    def build_model(self):
        """Build a training model."""

        raise NotImplementedError()

    @logging
    def save(self, filename=None, iteration=0):
        """Dump values of self.parameters into a npz file."""

        filename = filename or C[K_Model]

        np.savez(str(model_iteration_name(filename, iteration)),
                 **{name: parameter.get_value() for name, parameter in self.parameters.iteritems()})

    @logging
    def load(self, filename=None, iteration=0):
        """Load values of self.parameters from a npz file."""

        filename = filename or C[K_Model]

        with np.load(str(model_iteration_name(filename, iteration))) as f:
            for name, parameter in f.iteritems():
                self.parameters[name] = theano.shared(parameter, name=name)

