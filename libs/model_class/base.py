#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from ..utils.basic import fX, p_
from ..utils.path import model_iteration_name
from ..utils.config import C
from ..utils.constants import *
from ..utils.my_logging import logging
from ..utils.name_register import NameRegister
from ..multiverso.theano_ext import sharedvar

__author__ = 'fyabc'


class Model(NameRegister):
    """The model class.
    
    Shapes of some inputs, outputs and intermediate results:
        [W]: dim_word, word vector dim
        [H]: dim, hidden size
        [BS]: batch_size or n_samples
        [Ts]: n_timestep_src
        [Tt]: n_timestep_tgt
        [Ts/t]: [Ts] or [Tt]
        [Hc]: context_dim
    """
    NameTable = {}

    # FIXME: The duplicated shared variable is to fix the bug in multiverso.
    # These variables will cause error when trained on more than one nodes without duplicate.
    # From v-yixia
    DuplicateSharedVarList = ['decoder_Ub_att']
    DuplicateSize = 100

    def __init__(self, load=None, options=None):
        """
        :param load: None or a dict that contains the parameters of ``self.load``.
        """

        # Options.
        self.O = C.copy() if options is None else options

        # Parameters of theano shared variables
        self.P = OrderedDict()

        # Duplicate parameters of theano shared variables.
        self.dupP = OrderedDict()

        # Reserved for future use. the alignment matrix is put into this dictionary in Cho's tutorial code
        self.opt_ret = {}

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
                self.P[name] = theano.shared(value, name=name)
                self.dupP[name] = sharedvar.mv_shared(
                    value=np.ones(self.DuplicateSize) * value[0],
                    name=name,
                    borrow=False,
                )
            else:
                self.P[name] = sharedvar.mv_shared(
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

        filename = filename or self.O[K_Model]

        np.savez(str(model_iteration_name(filename, iteration)),
                 **{name: parameter.get_value() for name, parameter in self.P.iteritems()})

    @logging
    def load(self, filename=None, iteration=0):
        """Load values of self.parameters from a npz file."""

        filename = filename or self.O[K_Model]

        with np.load(str(model_iteration_name(filename, iteration))) as f:
            for name, parameter in f.iteritems():
                self.P[name] = theano.shared(parameter, name=name)

    # Some useful utilities and layers for building models.

    @staticmethod
    def get_input():
        """Get tensor of input variables.
        
        Input shape: ([Ts/t], [BS])
        """

        x = T.matrix('x', dtype='int64')
        x_mask = T.matrix('x_mask', dtype=fX)
        y = T.matrix('y', dtype='int64')
        y_mask = T.matrix('y_mask', dtype=fX)

        return x, x_mask, y, y_mask

    @staticmethod
    def input_dimensions(x, y):
        """Get input dimensions.

        :param x: input x
        :param y: input y
        :return: 3 Theano variables:
            n_timestep, n_timestep_tgt, n_samples
        """

        n_timestep = x.shape[0]
        n_timestep_tgt = y.shape[0]
        n_samples = x.shape[1]

        return n_timestep, n_timestep_tgt, n_samples

    @staticmethod
    def reverse_input(x, x_mask):
        return x[::-1], x_mask[::-1]

    def embedding(self, input_, n_timestep, n_samples, emb_name='Wemb'):
        """Embedding."""

        emb = self.P[emb_name][input_.flatten()]
        emb = emb.reshape([n_timestep, n_samples, self.O['dim_word']])

        return emb

    @staticmethod
    def dropout(input_, use_noise, rand, dropout_rate=0.5):
        """Dropout layer."""

        kept = rand.binomial(input_.shape, p=(1. - dropout_rate), n=1, dtype=fX)
        ratio = 1. - kept * dropout_rate
        inv_ratio = kept / ratio

        return T.switch(
            use_noise,
            input_ * inv_ratio,
            input_,
        )

    def fully_connect(self, input_, prefix='rconv', activation=T.tanh, **kwargs):
        if isinstance(activation, (str, unicode)):
            activation = eval(activation)
        return activation(T.dot(input_, self.P[p_(prefix, 'W')]) + self.P[p_(prefix, 'b')])

    @staticmethod
    def get_context_mean(context, x_mask):
        """Get mean of context (across time) as initial state of decoder RNN

        Or you can use the last state of forward + backward encoder RNNs
            # return concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)
        """

        return (context * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    @staticmethod
    def attention(context_mask, et, ht_1, We_att, Wh_att, Wb_att, U_att, Ub_att):
        """Attention layer."""

        a_network = T.tanh(T.dot(et, We_att) + T.dot(ht_1, Wh_att) + Wb_att)
        alpha = T.dot(a_network, U_att) + Ub_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = T.exp(alpha)
        if context_mask:
            alpha *= context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        # if Wp_compress_e:
        #    ctx_t = (T.dot(et, Wp_compress_e) * alpha[:,:,None]).sum(0) # This is the c_t in Baidu's paper
        # else:
        #    ctx_t = (et * alpha[:,:,None]).sum(0)
        ctx_t = (et * alpha[:, :, None]).sum(0)
        return ctx_t

    def get_cost(self, y, y_mask, probs):
        y_flat = y.flatten()
        y_flat_idx = T.arange(y_flat.shape[0]) * self.O['n_words'] + y_flat
        cost = -T.log(probs.flatten()[y_flat_idx])
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)

        return cost
