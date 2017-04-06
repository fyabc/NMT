#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict
import warnings
import os

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from ..utils.basic import *
from ..utils.path import model_iteration_name
from ..utils.config import C
from ..utils.constants import *
from ..utils.my_logging import logging, message
from ..utils.name_register import NameRegister
from ..utils.optimizers import Optimizer
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

    def __init__(self, options=None, top_options=None):
        """
        :param options: the model options (maybe updated by old options).
            default to global config ``C``.
        :param top_options: the top-level model options (will not be updated by old options).
            default to ``options``.
        """

        # Options.
        self.O = C if options is None else options
        self.TO = self.O if top_options is None else top_options

        # Parameters of theano shared variables
        self.P = OrderedDict()

        # Duplicate parameters of theano shared variables.
        self.dupP = OrderedDict()

        # Reserved for future use. the alignment matrix is put into this dictionary in Cho's tutorial code
        self.opt_ret = {}

        # Theano functions.
        self.f_log_probs = None
        self.f_grad_shared = None
        self.f_update = None

        print('Building model')

        np_parameters = self.init_np_parameters()
        if self.TO['reload_'] and os.path.exists(self.TO[K_Model]):
            print('Reloading model parameters...')
            self.load_params(np_parameters)

        self.message_params()

        self.init_parameters(np_parameters)

        # Some shared variables.
        self.trng = RandomStreams(self.TO['seed'])
        self.use_noise = theano.shared(np.float32(1.0))

        self.x, self.x_mask, self.y, self.y_mask, self.cost = self.build_model(
            trng=self.trng,
            use_noise=self.use_noise,
        )
        self.inputs = [self.x, self.x_mask, self.y, self.y_mask]

        self.build_loss_and_optimizer()

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

    def build_model(self, **kwargs):
        """Build a training model."""

        raise NotImplementedError()

    def build_loss_and_optimizer(self):
        # Before any regularizer
        print('Building f_log_probs...', end='')
        self.f_log_probs = theano.function(self.inputs, self.cost, profile=self.TO['profile'])
        print('Done')

        self.cost = self.cost.mean()

        # Apply L2 regularization on weights
        decay_c = self.TO['decay_c']
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for key, value in self.P.iteritems():
                if 'Wemb' not in key:
                    weight_decay += (value ** 2).sum()
            weight_decay *= decay_c
            self.cost += weight_decay

        # # regularize the alpha weights
        # alpha_c = self.TO['alpha_c']
        # if alpha_c > 0. and not self.O['decoder'].endswith('simple'):
        #     alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
        #     alpha_reg = alpha_c * (
        #         (T.cast(self.y_mask.sum(0) // self.x_mask.sum(0), 'float32')[:, None] -
        #          self.opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        #     self.cost += alpha_reg

        print('Computing gradient...', end='')
        grads = T.grad(self.cost, wrt=itemlist(self.P))
        print('Done')

        # Apply gradient clipping here
        clip_c = self.TO['clip_c']
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g ** 2).sum()
            new_grads = [
                T.switch(
                    g2 > (clip_c ** 2),
                    g / T.sqrt(g2) * clip_c,
                    g
                ) for g in grads
            ]
            grads = new_grads

        # Compile the optimizer, the actual computational graph is compiled here
        lr = T.scalar(name='lr')
        print('Building optimizers...', end='')
        self.f_grad_shared, self.f_update = Optimizer.get_by_name(self.TO['optimizer']).apply(
            lr, self.P, grads, self.inputs, self.cost)
        print('Done')

    @logging
    def save(self, filename=None, iteration=0, history_errs=None):
        """Dump values of self.parameters into a npz file."""

        filename = filename or self.TO[K_Model]

        np.savez(str(model_iteration_name(filename, iteration)),
                 history_errs=history_errs,
                 uidx=iteration,
                 **{name: parameter.get_value() for name, parameter in self.P.iteritems()})

    def load_params(self, np_parameters, filename=None, iteration=None):
        """Load values of np_parameters from a npz file."""

        filename = filename or self.TO[K_Model]

        # If given iteration, replace xxx.npz -> xxx.iter1000.npz
        if iteration is not None:
            filename = model_iteration_name(filename, iteration)

        with np.load(filename) as f:
            for name, parameter in f.iteritems():
                if name not in np_parameters:
                    warnings.warn('{} is not in the archive'.format(name))
                np_parameters[name] = theano.shared(parameter, name=name)

        return np_parameters

    def message_params(self, exit_=False):
        total_parameters = 0

        message('Model Parameters:')
        for k, v in self.P.iteritems():
            message('  >', k, v.shape, v.dtype)
            total_parameters += v.size
        message('Total parameters of the network: {}'.format(total_parameters))
        message('Model Parameters Done')

        if exit_:
            exit(0)

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
