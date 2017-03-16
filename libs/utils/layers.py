#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""Function of layers and their initializers.

1.  Parameters of layers:

input_: a Theano tensor that represent the input.
params: the dict of (tensor) parameters.
prefix: string, the prefix layer name.

return: a Theano tensor that represent the output.

2.  Parameters of initializers:

params: the dict of (numpy) parameters.
prefix: the prefix layer name.

Optional parameter n_in: input size.
Optional parameter n_out: output size.
Optional parameter dim: dimension size (hidden size?).

return: params
"""

from __future__ import print_function, unicode_literals

import theano
import theano.tensor as T

from ..utils.basic import fX
from ..utils.my_math import concatenate
from ..utils.config import C

__author__ = 'fyabc'


# Some utilities.

def p_(*args):
    """Get the name of tensor with the prefix (layer name) and variable name(s)."""
    return '_'.join(str(arg) for arg in args)


def _slice(_x, n, _dim):
    """Utility function to slice a tensor."""

    if _x.ndim == 3:
        return _x[:, :, n * _dim:(n + 1) * _dim]
    return _x[:, n * _dim:(n + 1) * _dim]


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


def FullyConnect(input_, params, prefix='rconv', activation=T.tanh, **kwargs):
    return activation(T.dot(input_, params[p_(prefix, 'W')]) + params[p_(prefix, 'b')])


# LSTM helper functions.

def _attention_layer(context_mask, et, ht_1, We_att, Wh_att, Wb_att, U_att, Ub_att):
    a_network = T.tanh(T.dot(et, We_att) + T.dot(ht_1, Wh_att) + Wb_att)

    alpha = T.dot(a_network, U_att) + Ub_att
    alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
    alpha = T.exp(alpha)

    if context_mask:
        alpha *= context_mask

    alpha /= alpha.sum(0, keepdims=True)

    ctx_t = (et * alpha[:, :, None]).sum(0)
    return ctx_t


def _lstm_step_slice(src_mask, ft, ht_1, st_1, Wr, br):
    z_vector = ft + T.dot(ht_1, Wr) + br
    _dim = Wr.shape[1] // 4
    z = _slice(z_vector, 0 , _dim)
    z_rho = _slice(z_vector, 1 , _dim)
    z_phi = _slice(z_vector, 2 , _dim)
    z_pi = _slice(z_vector, 3 , _dim)
    st = T.tanh(z) * T.nnet.sigmoid(z_rho) + T.nnet.sigmoid(z_phi) * st_1
    ht = T.tanh(st) * T.nnet.sigmoid(z_pi)

    st = src_mask[:, None] * st + (1. - src_mask)[:, None] * st_1
    ht = src_mask[:, None] * ht + (1. - src_mask)[:, None] * ht_1

    return ht, st


def _lstm_stem_slice_with_theta(src_mask, ft, ht_1, st_1, Wr, br, theta_rho, theta_phi, theta_pi):
    """Internal function: step forward, for LSTM + Fast-Forward [Baidu's paper]"""

    z_vector = ft + T.dot(ht_1, Wr) + br
    _dim = Wr.shape[1] // 4
    z = _slice(z_vector, 0, _dim)
    z_rho = _slice(z_vector, 1, _dim)
    z_phi = _slice(z_vector, 2, _dim)
    z_pi = _slice(z_vector, 3, _dim)
    st = T.tanh(z) * T.nnet.sigmoid(z_rho + st_1 * theta_rho[None, :]) + \
        T.nnet.sigmoid(z_phi + st_1 * theta_phi[None, :]) * st_1
    ht = T.tanh(st) * T.nnet.sigmoid(z_pi + st * theta_pi[None, :])

    st = src_mask[:, None] * st + (1. - src_mask)[:, None] * st_1
    ht = src_mask[:, None] * ht + (1. - src_mask)[:, None] * ht_1

    return ht, st


def LstmFastFwEncoder(input_, params, prefix='encoder', mask=None, dropout_param=None):
    """LSTM Fast-Forward Encoder."""

    n_layer = C['n_encoder_layer']
    use_zigzag = C['use_zigzag']

    assert n_layer >= 1, '#layer must >= 1'

    _lstm_step_func = _lstm_stem_slice_with_theta if C['use_theta'] else _lstm_step_slice

    n_steps = input_.shape[0]
    n_samples = input_.shape[1] if input_.ndim == 3 else 1

    if mask is None:
        mask = T.alloc(1., input_.shape[0], 1)

    global_f = input_
    _hidden_state_last_layer = None

    for layer_id in xrange(n_layer):
        Wf = params[p_(prefix, 'Wf', layer_id)]

        if layer_id > 0:
            if C['use_half']:
                t_dim = global_f.shape[2] // 2
                global_f = concatenate([global_f[:, :, 0:t_dim], _hidden_state_last_layer[0]], axis=1)
            else:
                global_f = concatenate([global_f, _hidden_state_last_layer[0]], axis=-1)

        global_f = T.dot(global_f, Wf)
        dim = Wf.shape[0] / 4
        init_states = [T.alloc(0., n_samples, dim), T.alloc(0., n_samples, dim)]

        if C['use_theta']:
            shared_vars = [params[p_(prefix, 'Wr', layer_id)],
                           params[p_(prefix, 'br', layer_id)],
                           params[p_(prefix, 'theta_rho', layer_id)],
                           params[p_(prefix, 'theta_phi', layer_id)],
                           params[p_(prefix, 'theta_pi', layer_id)]]
        else:
            shared_vars = [params[p_(prefix, 'Wr', layer_id)],
                           params[p_(prefix, 'br', layer_id)]]

        seqs = [mask, global_f]

        _hidden_state_last_layer, _ = theano.scan(
            fn=_lstm_step_func,
            sequences=seqs,
            outputs_info=init_states,
            non_sequences=shared_vars,
            name=p_(prefix, 'layer', layer_id),
            n_steps=n_steps,
            profile=C['profile'],
            strict=True,
        )

        if layer_id < n_layer - 1 and use_zigzag:
            _hidden_state_last_layer[0] = _hidden_state_last_layer[0][::-1]
            mask = mask[::-1]
            global_f = global_f[::-1]

        if dropout_param:
            _hidden_state_last_layer[0] = dropout(_hidden_state_last_layer[0], *dropout_param)

    if use_zigzag and n_layer % 2 == 0:
        return [global_f[::-1], _hidden_state_last_layer[0][::-1]]
    else:
        return [global_f, _hidden_state_last_layer[0]]
