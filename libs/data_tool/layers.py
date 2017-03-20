#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""Function of layers and their initializers.

1.  Parameters of layers:

input_: a Theano tensor that represent the input.
params: the dict of (tensor) parameters.
prefix: string, the prefix layer name.

return: a Theano tensor that represent the output.

2.  Parameters of initializers:

No parameters

return: params: the dict of (np) parameters.
"""

from __future__ import print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from .my_math import concatenate, normal_weight, orthogonal_weight_1xb, uniform_weight, orthogonal_weight
from ..utils.basic import fX
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


def _lstm_step_slice(src_mask, ft,
                     ht_1, st_1,
                     Wr, br):
    z_vector = ft + T.dot(ht_1, Wr) + br
    _dim = Wr.shape[1] // 4
    z = _slice(z_vector, 0, _dim)
    z_rho = _slice(z_vector, 1, _dim)
    z_phi = _slice(z_vector, 2, _dim)
    z_pi = _slice(z_vector, 3, _dim)
    st = T.tanh(z) * T.nnet.sigmoid(z_rho) + T.nnet.sigmoid(z_phi) * st_1
    ht = T.tanh(st) * T.nnet.sigmoid(z_pi)

    st = src_mask[:, None] * st + (1. - src_mask)[:, None] * st_1
    ht = src_mask[:, None] * ht + (1. - src_mask)[:, None] * ht_1

    return ht, st


def _lstm_step_slice_with_theta(src_mask, ft,
                                ht_1, st_1,
                                Wr, br, theta_rho, theta_phi, theta_pi):
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
    """LSTM Fast-Forward Encoder.

    input_: source word embedding
    """

    n_layer = C['n_encoder_layer']
    use_zigzag = C['use_zigzag']

    assert n_layer >= 1, '#layer must >= 1'

    _lstm_step_func = _lstm_step_slice_with_theta if C['use_theta'] else _lstm_step_slice

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


# LSTM Attention model helper functions.

def _lstm_att_step_1st_layer_with_theta(
        yemb, ymask,
        ht_1, st_1, global_f, ctx_t,
        et, context_mask, Wf, Wr, br, theta_rho, theta_phi, theta_pi,
        Wp_compress_e, bp_compress_e, We_att, Wh_att, Wb_att, U_att, Ub_att,
):
    ctx_t = _attention_layer(context_mask, et, ht_1, We_att, Wh_att, Wb_att, U_att, Ub_att)
    ctx_t = T.dot(ctx_t, Wp_compress_e) + bp_compress_e
    global_f = T.dot(concatenate([ctx_t, yemb], axis=-1), Wf)
    ht, st = _lstm_step_slice_with_theta(ymask, global_f, ht_1, st_1, Wr, br, theta_rho, theta_phi, theta_pi)
    return ht, st, global_f, ctx_t


def _lstm_att_step_1stlayer(
        yemb, ymask,
        ht_1, st_1, global_f, ctx_t,
        et, context_mask, Wf, Wr, br, Wp_compress_e, bp_compress_e, We_att, Wh_att, Wb_att, U_att,
        Ub_att
):
    ctx_t = _attention_layer(context_mask, et, ht_1, We_att, Wh_att, Wb_att, U_att, Ub_att)
    ctx_t = T.dot(ctx_t, Wp_compress_e) + bp_compress_e
    global_f = T.dot(concatenate([ctx_t, yemb], axis=-1), Wf)
    ht, st = _lstm_step_slice(ymask, global_f, ht_1, st_1, Wr, br)
    return ht, st, global_f, ctx_t


def LstmFastFwDecoder(input_, params, mask, context, context_mask, previous_h,
                      prefix='decoder', one_step=False, previous_s=None, dropout_param=None):
    """LSTM Fast-Forward Decoder.

    input_: target word embedding
    mask: target mask
    """

    # TODO: Currently, I do not return the alpha for further alignment.
    # You can implement it easily bu inserting a dummy position in _lstm_att_step_1st_layer and get it.
    # TODO: I still write an "if ... else..." for one_step

    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #time_step * #sample * #dim'
    if one_step:
        assert previous_h, 'Previous_h must be provided'
        assert previous_s, 'Previous_s must be provided'
        assert previous_h.ndim == 3, 'Previous_h must be 3-d'
        assert previous_s.ndim == 3, 'Previous_s must be 3-d'

    n_layer = C['n_decoder_layer']
    _lstm_step_func = _lstm_step_slice_with_theta if C['use_theta'] else _lstm_step_slice

    # First, we deal with the first layer with attention model
    n_steps = input_.shape[0]
    n_samples = input_.shape[1] if input_.ndim == 3 else 1

    if mask is None:
        mask = T.alloc(1., input_.shape[0], )

    hidden_dim = params[p_(prefix, 'Wf', 0)].shape[-1] // 4
    non_sequence_vars = [
        context,
        context_mask,
        params[p_(prefix, 'Wf', 0)],
        params[p_(prefix, 'Wr', 0)],
        params[p_(prefix, 'br', 0)],
        params[p_(prefix, 'Wp_compress_e')],
        params[p_(prefix, 'bp_compress_e')],
        params[p_(prefix, 'We_att')],
        params[p_(prefix, 'Wh_att')],
        params[p_(prefix, 'Wb_att')],
        params[p_(prefix, 'U_att')],
        params[p_(prefix, 'Ub_att')],
    ]
    _lstm_first_step_func = _lstm_att_step_1stlayer

    if C['use_theta']:
        non_sequence_vars[5:5] = [
            params[p_(prefix, 'theta_rho', 0)],
            params[p_(prefix, 'theta_phi', 0)],
            params[p_(prefix, 'theta_pi', 0)],
        ]
        _lstm_first_step_func = _lstm_att_step_1st_layer_with_theta

    input_seqs = [input_, mask]

    if one_step:
        _hidden_state_last_layer = list(_lstm_first_step_func(
            *(input_seqs + [previous_h[0], previous_s[0], None, None] + non_sequence_vars)
        ))
    else:
        output_seqs = [
            previous_h,
            # Please note that this is the initial st_1 with all zeros; this is for training purpose only
            T.alloc(0., n_samples, hidden_dim),
            T.alloc(0., n_samples, 4 * hidden_dim),
            T.alloc(0., n_samples, 10 * hidden_dim / 4),
        ]
        _hidden_state_last_layer, _ = theano.scan(
            fn=_lstm_first_step_func,
            sequences=input_seqs,
            outputs_info=output_seqs,
            non_sequences=non_sequence_vars,
            name=p_(prefix, 'layer', 0),
            n_steps=n_steps,
            profile=C['profile'],
            strict=True,
        )

    if dropout_param:
        _hidden_state_last_layer[0] = dropout(_hidden_state_last_layer[0], *dropout_param)

    # Next, we deal with layer > 1, which has no attention model
    global_f = _hidden_state_last_layer[2]
    ctx_from_1st_layer = _hidden_state_last_layer[3]

    if one_step:
        stack_h = _hidden_state_last_layer[0].dimshuffle('x', 0, 1)
        stack_s = _hidden_state_last_layer[1].dimshuffle('x', 0, 1)

        for layer_id in xrange(1, n_layer):
            Wf = params[p_(prefix, 'Wf', layer_id)]
            if C['use_half']:
                t_dim = global_f.shape[1] // 2
                global_f = concatenate([global_f[:, 0:t_dim], stack_h[-1]], axis=-1)
            else:
                global_f = concatenate([global_f, stack_h[-1]], axis=-1)
            global_f = T.dot(global_f, Wf)

            init_states = [previous_h[layer_id], previous_s[layer_id]]
            if C['use_theta']:
                shared_vars = [
                    params[p_(prefix, 'Wr', layer_id)],
                    params[p_(prefix, 'br', layer_id)],
                    params[p_(prefix, 'theta_rho', layer_id)],
                    params[p_(prefix, 'theta_phi', layer_id)],
                    params[p_(prefix, 'theta_pi', layer_id)],
                ]
            else:
                shared_vars = [
                    params[p_(prefix, 'Wr', layer_id)],
                    params[p_(prefix, 'br', layer_id)],
                ]
            seqs = [mask, global_f]
            _hidden_state_last_layer = list(_lstm_step_func(*(seqs + init_states + shared_vars)))
            if dropout_param:
                _hidden_state_last_layer[0] = dropout(_hidden_state_last_layer[0], *dropout_param)
            stack_h = concatenate([stack_h, _hidden_state_last_layer[0].dimshuffle('x', 0, 1)])
            stack_s = concatenate([stack_s, _hidden_state_last_layer[1].dimshuffle('x', 0, 1)])

        return stack_h, stack_s, ctx_from_1st_layer
    else:
        for layer_id in xrange(n_layer):
            Wf = params[p_(prefix, 'Wf', layer_id)]
            if C['use_half']:
                t_dim = global_f.shape[1] // 2
                global_f = concatenate([global_f[:, :, 0:t_dim], _hidden_state_last_layer[0]], axis=-1)
            else:
                global_f = concatenate([global_f, _hidden_state_last_layer[0]], axis=-1)
            global_f = T.dot(global_f, Wf)

            init_states = [previous_h, T.alloc(0., n_samples, Wf.shape[1] / 4)]
            if C['use_theta']:
                shared_vars = [
                    params[p_(prefix, 'Wr', layer_id)],
                    params[p_(prefix, 'br', layer_id)],
                    params[p_(prefix, 'theta_rho', layer_id)],
                    params[p_(prefix, 'theta_phi', layer_id)],
                    params[p_(prefix, 'theta_pi', layer_id)],
                ]
            else:
                shared_vars = [
                    params[p_(prefix, 'Wr', layer_id)],
                    params[p_(prefix, 'br', layer_id)],
                ]
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

            if dropout_param:
                _hidden_state_last_layer[0] = dropout(_hidden_state_last_layer[0], *dropout_param)

    return _hidden_state_last_layer[0], ctx_from_1st_layer


def _gru_step_slice(src_mask, ft,
                    ht_1,
                    Uz, Ur, Uh, bz, br, bh):
    """Internal function: step forward, for GRU + Raw (Raw means that I do not cut the f into half)"""

    _dim = Uz.shape[1]
    f0 = _slice(ft, 0, _dim)
    f1 = _slice(ft, 1, _dim)
    f2 = _slice(ft, 2, _dim)
    zt = T.nnet.sigmoid(f0 + T.dot(ht_1, Uz) + bz)
    rt = T.nnet.sigmoid(f1 + T.dot(ht_1, Ur) + br)
    ht_tilde = T.tanh(f2 + T.dot(rt * ht_1, Uh) + bh)
    ht = (1. - zt) * ht_1 + zt * ht_tilde
    ht = src_mask[:, None] * ht + (1. - src_mask)[:, None] * ht_1
    return ht


def GruEncoder(input_, params, prefix='encoder', mask=None, dropout_param=None):
    """GRU Encoder.

    input_: source word embedding.
    """

    n_layer = C['n_encoder_layer']
    use_zigzag = C['use_zigzag']
    upload_emb = C.get('upload_emb', False)
    use_final_residual = C.get('use_final_residual', False)

    n_steps = input_.shape[0]
    n_samples = input_.shape[1] if input_.ndim == 3 else 1

    global_f = input_
    if upload_emb or (use_final_residual and n_layer == 1):
        xi_1 = T.dot(input_, params[p_(prefix, 'emb2first_W')]) + params[p_(prefix, 'emb2first_b')]
    else:
        xi_1 = None

    _hidden_state_last_layer = None

    for layer_id in xrange(n_layer):
        Wf = params[p_(prefix, 'Wf', layer_id)]

        if layer_id == 0:
            pass
        elif layer_id == 1:
            if not upload_emb:
                global_f = _hidden_state_last_layer
            else:
                global_f = _hidden_state_last_layer + (xi_1[::-1] if use_zigzag else xi_1)
            xi_1 = _hidden_state_last_layer[::-1] if use_zigzag else _hidden_state_last_layer
        else:
            global_f = _hidden_state_last_layer + xi_1
            xi_1 = _hidden_state_last_layer[::-1] if use_zigzag else _hidden_state_last_layer

        if layer_id > 0 and dropout_param:
            global_f = dropout(global_f, *dropout_param)

        global_f = T.dot(global_f, Wf)

        init_states = [T.alloc(0., n_samples, Wf.shape[1] / 3)]
        shared_vars = [
            params[p_(prefix, 'Uz', layer_id)],
            params[p_(prefix, 'Ur', layer_id)],
            params[p_(prefix, 'Uh', layer_id)],
            params[p_(prefix, 'bz', layer_id)],
            params[p_(prefix, 'br', layer_id)],
            params[p_(prefix, 'bh', layer_id)]
        ]

        seqs = [mask, global_f]

        _hidden_state_last_layer, _ = theano.scan(
            fn=_gru_step_slice,
            sequences=seqs,
            outputs_info=init_states,
            non_sequences=shared_vars,
            name=p_(prefix, 'layer', layer_id),
            n_steps=n_steps,
            profile=C['profile'],
            strict=True,
        )

        if layer_id < n_layer - 1 and use_zigzag:
            _hidden_state_last_layer = _hidden_state_last_layer[::-1]
            mask = mask[::-1]
            global_f = global_f[::-1]

            # if dropout_param:
            #     _hidden_state_last_layer = dropout(_hidden_state_last_layer, *dropout_param)

    if n_layer % 2 == 0 and use_zigzag:
        global_f = global_f[::-1]
        _hidden_state_last_layer = _hidden_state_last_layer[::-1]

    if use_final_residual:
        if n_layer % 2 == 0 and use_zigzag:
            _hidden_state_last_layer += xi_1[::-1]
        else:
            _hidden_state_last_layer += xi_1

    return [global_f, _hidden_state_last_layer]


# GRU Attention model helper functions.

def _gru_att_step_1st_layer(
        yemb, ymask,
        ht_1, global_f, ctx_t,
        et, context_mask, Wf, We_att, Wh_att, Wb_att, U_att, Ub_att, Uz, Ur, Uh, bz, br, bh
):
    ctx_t = _attention_layer(context_mask, et, ht_1, We_att, Wh_att, Wb_att, U_att, Ub_att)
    global_f = T.dot(concatenate([ctx_t, yemb], axis=-1), Wf)
    ht = _gru_step_slice(ymask, global_f, ht_1, Uz, Ur, Uh, bz, br, bh)
    return ht, global_f, ctx_t


def _gru_att_step_1st_layer_compress(
        yemb, ymask,
        ht_1, global_f, ctx_t,
        et, context_mask, Wf, We_att, Wh_att, Wb_att, U_att, Ub_att, Uz, Ur, Uh, bz, br,
        bh, Wp_compress_e, bp_compress_e
):
    ctx_t = _attention_layer(context_mask, et, ht_1, We_att, Wh_att, Wb_att, U_att, Ub_att)
    ctx_t = T.dot(ctx_t, Wp_compress_e) + bp_compress_e
    global_f = T.dot(concatenate([ctx_t, yemb], axis=-1), Wf)
    ht = _gru_step_slice(ymask, global_f, ht_1, Uz, Ur, Uh, bz, br, bh)
    return ht, global_f, ctx_t


def GruDecoder(input_, params, mask, context, context_mask, previous_h,
               prefix='decoder', one_step=False, dropout_param=None):
    """GRU Decoder.

    input_: target word embedding
    mask: target mask
    """

    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #time_step * #sample * dim while {} provided'.format(context.ndim)
    if one_step:
        assert previous_h, 'Previous_h state must be provided'
        assert previous_h.ndim == 3, 'In sample_mode, previous_h should by (#layer * #sample * #feature)'

    n_layer = C['n_encoder_layer']
    upload_emb = C.get('upload_emb', False)
    use_final_residual = C.get('use_final_residual', False)

    # First, we deal with the first layer with attention model
    n_steps = input_.shape[0]
    n_samples = input_.shape[1] if input_.ndim == 3 else 1

    if mask is None:
        mask = T.alloc(1., input_.shape[0], )

    hidden_dim = params[p_(prefix, 'Uz', 0)].shape[1]
    non_sequence_vars = [
        context,
        context_mask,
        params[p_(prefix, 'Wf', 0)],
        params[p_(prefix, 'We_att')],
        params[p_(prefix, 'Wh_att')],
        params[p_(prefix, 'Wb_att')],
        params[p_(prefix, 'U_att')],
        params[p_(prefix, 'Ub_att')],
        params[p_(prefix, 'Uz', 0)],
        params[p_(prefix, 'Ur', 0)],
        params[p_(prefix, 'Uh', 0)],
        params[p_(prefix, 'bz', 0)],
        params[p_(prefix, 'br', 0)],
        params[p_(prefix, 'bh', 0)],
    ]

    # [NOTE] Hard code for model structure
    if C['to_upper_layer'].lower() == 'fastfw':
        non_sequence_vars.extend([
            params[p_(prefix, 'Wp_compress_e')],
            params[p_(prefix, 'bp_compress_e')],
        ])
        _first_step_func = _gru_att_step_1st_layer_compress
    else:
        _first_step_func = _gru_att_step_1st_layer

    input_seqs = [input_, mask]

    if one_step:
        output_seqs = [previous_h[0], None, None]
        # To be continued
        _hidden_state_last_layer = list(_first_step_func(*(input_seqs + output_seqs + non_sequence_vars)))
    else:
        output_seqs = [
            previous_h,
            T.alloc(0., n_samples, 3 * hidden_dim),
            T.alloc(0., n_samples, 2 * hidden_dim),
        ]
        _hidden_state_last_layer, _ = theano.scan(
            fn=_first_step_func,
            sequences=input_seqs,
            outputs_info=output_seqs,
            non_sequences=non_sequence_vars,
            name=p_(prefix, 'layer', 0),
            n_steps=n_steps,
            profile=C['profile'],
            strict=True,
        )

    if dropout_param:
        _hidden_state_last_layer[0] = dropout(_hidden_state_last_layer[0], *dropout_param)

    # Next, we deal with layer > 1, which has no attention model
    if upload_emb and n_layer > 1:
        xi_1 = T.dot(input_, params[p_(prefix, 'emb2first_W')]) + params[p_(prefix, 'emb2first_b')]

    global_f = _hidden_state_last_layer[1]
    ctx_from_1st_layer = _hidden_state_last_layer[2]
    _hidden_state_last_layer = _hidden_state_last_layer[0]

    if one_step:
        _hidden_state_last_layer = _hidden_state_last_layer.dimshuffle('x', 0, 1)
        for layer_id in xrange(1, n_layer):
            Wf = params[p_(prefix, 'Wf', layer_id)]
            if layer_id > 1:
                global_f = _hidden_state_last_layer[-1] + _hidden_state_last_layer[-2]
            else:
                if upload_emb:
                    global_f = _hidden_state_last_layer[-1] + xi_1
                else:
                    global_f = _hidden_state_last_layer[-1]

            if dropout_param:
                global_f = dropout(global_f, *dropout_param)
            global_f = T.dot(global_f, Wf)

            shared_vars = [
                params[p_(prefix, 'Uz', layer_id)],
                params[p_(prefix, 'Ur', layer_id)],
                params[p_(prefix, 'Uh', layer_id)],
                params[p_(prefix, 'bz', layer_id)],
                params[p_(prefix, 'br', layer_id)],
                params[p_(prefix, 'bh', layer_id)],
            ]

            hidden_state_2D = _gru_step_slice(*([mask, global_f, previous_h[layer_id]] + shared_vars))
            # if dropout_param:
            #     hidden_state_2D = dropout(hidden_state_2D, *dropout_param)
            _hidden_state_last_layer = concatenate([_hidden_state_last_layer, hidden_state_2D.dimshuffle('x', 0, 1)])
    else:
        for layer_id in xrange(1, n_layer):
            Wf = params[p_(prefix, 'Wf', layer_id)]
            if layer_id > 1:
                global_f = xi_1 + _hidden_state_last_layer
            else:
                if upload_emb:
                    global_f = _hidden_state_last_layer + xi_1
                else:
                    global_f = _hidden_state_last_layer
            xi_1 = _hidden_state_last_layer
            if dropout_param:
                global_f = dropout(global_f, *dropout_param)
            global_f = T.dot(global_f, Wf)

            shared_vars = [
                params[p_(prefix, 'Uz', layer_id)],
                params[p_(prefix, 'Ur', layer_id)],
                params[p_(prefix, 'Uh', layer_id)],
                params[p_(prefix, 'bz', layer_id)],
                params[p_(prefix, 'br', layer_id)],
                params[p_(prefix, 'bh', layer_id)],
            ]
            seqs = [mask, global_f]

            _hidden_state_last_layer, _ = theano.scan(
                fn=_gru_step_slice,
                sequences=seqs,
                outputs_info=[previous_h],
                non_sequences=shared_vars,
                name=p_(prefix, 'layer', layer_id),
                n_steps=n_steps,
                profile=C['profile'],
                strict=True,
            )

        if use_final_residual and n_layer >= 2:
            _hidden_state_last_layer += xi_1
            # if dropout_param:
            #     _hidden_state_last_layer = dropout(_hidden_state_last_layer, *dropout_param)

    return _hidden_state_last_layer, ctx_from_1st_layer


################
# Initializers #
################


# Initialize LSTM part helper functions.

def _param_init_lstm_part(params, m_layer, lstm_dim, prefix):
    for layer_id in xrange(m_layer):
        params[p_(prefix, 'Wr', layer_id)] = orthogonal_weight_1xb(lstm_dim, 4)
        params[p_(prefix, 'br', layer_id)] = np.zeros((4 * lstm_dim,), dtype=fX)
        if C['use_theta']:
            params[p_(prefix, 'theta_rho', layer_id)] = np.zeros((lstm_dim,), dtype='float32')  # norm_vector(lstm_dim)
            params[p_(prefix, 'theta_phi', layer_id)] = np.zeros((lstm_dim,), dtype='float32')
            params[p_(prefix, 'theta_pi', layer_id)] = np.zeros((lstm_dim,), dtype='float32')


def init_LstmWithFastFw():
    init_affine_weight, n_encoder_layer, n_decoder_layer, embedding_dim, lstm_dim, alignment_dim, voc_size = \
        C['init_affine_weight'], C['n_encoder_layer'], C['n_decoder_layer'], C['dim_word'], C['dim'], C[
            'alignment_dim'], C['n_words']

    assert lstm_dim % 2 == 0
    assert n_encoder_layer >= 1, '#LstmEncoderLayer must >= 1'
    assert n_decoder_layer >= 1, '#LstmDecoderLayer must >= 1'

    params = OrderedDict()

    # embedding
    params['Wemb'] = normal_weight(voc_size, embedding_dim, scale=init_affine_weight)
    params['Wemb_dec'] = normal_weight(voc_size, embedding_dim, scale=init_affine_weight)

    # Encoder (forward)
    _param_init_lstm_part(params, n_encoder_layer, lstm_dim, 'encoder')
    params[p_('encoder', 'Wf', 0)] = normal_weight(embedding_dim, 4 * lstm_dim, scale=init_affine_weight)
    for layer_id in xrange(1, n_encoder_layer):
        params[p_('encoder', 'Wf', layer_id)] = normal_weight(
            (1 + 4 / (1 + C['use_half'])) * lstm_dim,
            4 * lstm_dim,
            scale=init_affine_weight,
        )

    # Encoder (reverse)
    _param_init_lstm_part(params, n_encoder_layer, lstm_dim, 'encoder_r')
    params[p_('encoder_r', 'Wf', 0)] = normal_weight(embedding_dim, 4 * lstm_dim, scale=init_affine_weight)
    for layer_id in xrange(1, n_encoder_layer):
        params[p_('encoder_r', 'Wf', layer_id)] = normal_weight(
            (1 + 4 / (1 + C['use_half'])) * lstm_dim,
            4 * lstm_dim,
            scale=0.01,
        )

    # Decoder
    _param_init_lstm_part(params, n_decoder_layer, lstm_dim, 'decoder')
    params[p_('decoder', 'Wf', 0)] = normal_weight(embedding_dim, 4 * lstm_dim, scale=init_affine_weight)
    for layer_id in xrange(1, n_encoder_layer):
        params[p_('decoder', 'Wf', layer_id)] = normal_weight(
            (1 + 4 / (1 + C['use_half'])) * lstm_dim,
            4 * lstm_dim,
            scale=init_affine_weight,
        )

    # Compress the output of the encoder to 1/4
    params[p_('decoder', 'Wp_compress_e')] = normal_weight(10 * lstm_dim, 10 * lstm_dim / 4, scale=init_affine_weight)
    params[p_('decoder', 'bp_compress_e')] = np.zeros((10 * lstm_dim / 4,), dtype=fX)

    # For the initial state of decoder
    params[p_('initDecoder'), 'W'] = normal_weight(10 * lstm_dim, lstm_dim, scale=init_affine_weight)
    params[p_('initDecoder'), 'b'], np.zeros((lstm_dim,), dtype=fX)

    # For attention model
    params[p_('decoder', 'We_att')] = normal_weight(10 * lstm_dim, alignment_dim, scale=0.01)
    params[p_('decoder', 'Wh_att')] = normal_weight(lstm_dim, alignment_dim, scale=0.01)
    params[p_('decoder', 'Wb_att')] = np.zeros((alignment_dim,), dtype=fX)
    params[p_('decoder', 'U_att')] = normal_weight(alignment_dim, 1, scale=0.01)
    params[p_('decoder', 'Ub_att')] = np.zeros((1,), dtype=fX)

    # Map the output from decoder to the softmax layer
    params[p_('fc_compress_lastHiddenState', 'W')] = normal_weight(lstm_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_lastHiddenState', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_compress_emb', 'W')] = normal_weight(embedding_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_emb', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_compress_ctx', 'W')] = normal_weight(10 * lstm_dim / 4, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_ctx', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_to_softmax', 'W')] = normal_weight(embedding_dim, voc_size, scale=init_affine_weight)
    params[p_('fc_to_softmax', 'b')] = np.zeros((voc_size,), dtype=fX)

    return params


def _param_init_gru_part(params, m_layer, gru_dim, prefix):
    for layer_id in xrange(m_layer):
        params[p_(prefix, 'Ur', layer_id)] = orthogonal_weight(gru_dim)
        params[p_(prefix, 'Uz', layer_id)] = orthogonal_weight(gru_dim)
        params[p_(prefix, 'Uh', layer_id)] = orthogonal_weight(gru_dim)
        params[p_(prefix, 'br', layer_id)] = np.zeros((gru_dim,), dtype='float32')
        params[p_(prefix, 'bz', layer_id)] = np.zeros((gru_dim,), dtype='float32')
        params[p_(prefix, 'bh', layer_id)] = np.zeros((gru_dim,), dtype='float32')


def init_GruWithHighway():
    init_affine_weight, n_encoder_layer, n_decoder_layer, embedding_dim, gru_dim, alignment_dim, voc_size = \
        C['init_affine_weight'], C['n_encoder_layer'], C['n_decoder_layer'], C['dim_word'], C['dim'], C[
            'alignment_dim'], C['n_words']

    assert gru_dim % 2 == 0
    assert n_encoder_layer >= 1, '#LstmEncoderLayer must >= 1'
    assert n_decoder_layer >= 1, '#LstmDecoderLayer must >= 1'

    # [NOTE] Hard code for model structure
    use_src4all_layers = C['to_upper_layer'].lower() != 'raw'

    if C['init_distribution'] == 'norm':
        weight_sampler = normal_weight
    elif C['init_distribution'] == 'uniform':
        weight_sampler = uniform_weight
    else:
        raise Exception('Only support Normal or Uniform distribution while {} Given'.format(C['init_distribution']))

    params = OrderedDict()

    # Embedding
    params['Wemb'] = weight_sampler(voc_size, embedding_dim, scale=init_affine_weight)
    params['Wemb_dec'] = weight_sampler(voc_size, embedding_dim, scale=init_affine_weight)

    # Encoder (forward and reverse)
    _param_init_gru_part(params, n_encoder_layer, gru_dim, 'encoder')
    _param_init_gru_part(params, n_encoder_layer, gru_dim, 'encoder_r')
    params[p_('encoder', 'Wf', 0)] = weight_sampler(embedding_dim, 3 * gru_dim, scale=init_affine_weight)
    params[p_('encoder_r', 'Wf', 0)] = weight_sampler(embedding_dim, 3 * gru_dim, scale=init_affine_weight)
    for layer_id in xrange(1, n_encoder_layer):
        params[p_('encoder', 'Wf', layer_id)] = weight_sampler(
            embedding_dim * use_src4all_layers + gru_dim, 3 * gru_dim, scale=init_affine_weight)
        params[p_('encoder_r', 'Wf', layer_id)] = weight_sampler(
            embedding_dim * use_src4all_layers + gru_dim, 3 * gru_dim, scale=init_affine_weight)

    # Decoder
    _param_init_gru_part(params, n_decoder_layer, gru_dim, 'decoder')
    params[p_('decoder', 'Wf', 0)] = weight_sampler(2 * gru_dim + embedding_dim, 3 * gru_dim, scale=init_affine_weight)
    for layer_id in xrange(1, n_decoder_layer):
        params[p_('decoder', 'Wf', layer_id)] = weight_sampler(
            3 * gru_dim + embedding_dim * use_src4all_layers, 3 * gru_dim, scale=init_affine_weight)

    # For the initial state of decoder
    params[p_('initDecoder', 'W')] = weight_sampler(2 * gru_dim, gru_dim, scale=init_affine_weight)
    params[p_('initDecoder', 'b')] = np.zeros((gru_dim,), dtype='float32')

    # For attention model
    params[p_('decoder', 'We_att')] = normal_weight(2 * gru_dim, alignment_dim, scale=0.01)
    params[p_('decoder', 'Wh_att')] = normal_weight(gru_dim, alignment_dim, scale=0.01)
    params[p_('decoder', 'Wb_att')] = np.zeros((alignment_dim,), dtype=fX)
    params[p_('decoder', 'U_att')] = normal_weight(alignment_dim, 1, scale=0.01)
    params[p_('decoder', 'Ub_att')] = np.zeros((1,), dtype=fX)

    # Map the output from decoder to the softmax layer
    params[p_('fc_compress_lastHiddenState', 'W')] = weight_sampler(gru_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_lastHiddenState', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_compress_emb', 'W')] = weight_sampler(embedding_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_emb', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_compress_ctx', 'W')] = weight_sampler(2 * gru_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_ctx', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_to_softmax', 'W')] = weight_sampler(embedding_dim, voc_size, scale=init_affine_weight)
    params[p_('fc_to_softmax', 'b')] = np.zeros((voc_size,), dtype=fX)

    return params


def init_GruWithFastFw():
    init_affine_weight, n_encoder_layer, n_decoder_layer, embedding_dim, gru_dim, alignment_dim, voc_size = \
        C['init_affine_weight'], C['n_encoder_layer'], C['n_decoder_layer'], C['dim_word'], C['dim'], C[
            'alignment_dim'], C['n_words']

    assert gru_dim % 2 == 0
    assert n_encoder_layer >= 1, '#LstmEncoderLayer must >= 1'
    assert n_decoder_layer >= 1, '#LstmDecoderLayer must >= 1'

    params = OrderedDict()

    # Embedding
    params['Wemb'] = normal_weight(voc_size, embedding_dim, scale=init_affine_weight)
    params['Wemb_dec'] = normal_weight(voc_size, embedding_dim, scale=init_affine_weight)

    # Encoder (forward and reverse)
    _param_init_gru_part(params, n_encoder_layer, gru_dim, 'encoder')
    _param_init_gru_part(params, n_encoder_layer, gru_dim, 'encoder_r')
    params[p_('encoder', 'Wf', 0)] = normal_weight(embedding_dim, 3 * gru_dim, scale=init_affine_weight)
    params[p_('encoder_r', 'Wf', 0)] = normal_weight(embedding_dim, 3 * gru_dim, scale=init_affine_weight)
    for layer_id in xrange(1, n_encoder_layer):
        params[p_('encoder', 'Wf', layer_id)] = normal_weight(
            3 * gru_dim // (1 + C['use_half']) + gru_dim, 3 * gru_dim, scale=init_affine_weight)
        params[p_('encoder_r', 'Wf', layer_id)] = normal_weight(
            3 * gru_dim // (1 + C['use_half']) + gru_dim, 3 * gru_dim, scale=init_affine_weight)

    # Compress the output of the encoder to 1/4
    params[p_('decoder', 'Wp_compress_e')] = normal_weight(8 * gru_dim, 2 * gru_dim, scale=init_affine_weight)
    params[p_('decoder', 'bp_compress_e')] = np.zeros((2 * gru_dim,), dtype='float32')

    # Decoder
    _param_init_gru_part(params, n_decoder_layer, gru_dim, 'decoder')
    params[p_('decoder', 'Wf', 0)] = normal_weight(2 * gru_dim + embedding_dim, 3 * gru_dim, scale=init_affine_weight)
    for layer_id in xrange(1, n_decoder_layer):
        params[p_('decoder', 'Wf', layer_id)] = normal_weight(
            3 * gru_dim // (1 + C['use_half']) + gru_dim, 3 * gru_dim, scale=0.01)

    # For the initial state of decoder
    params[p_('initDecoder', 'W')] = normal_weight(8 * gru_dim, gru_dim, scale=init_affine_weight)
    params[p_('initDecoder', 'b')] = np.zeros((gru_dim,), dtype='float32')

    # For attention model
    params[p_('decoder', 'We_att')] = normal_weight(8 * gru_dim, alignment_dim, scale=0.01)
    params[p_('decoder', 'Wh_att')] = normal_weight(gru_dim, alignment_dim, scale=0.01)
    params[p_('decoder', 'Wb_att')] = np.zeros((alignment_dim,), dtype=fX)
    params[p_('decoder', 'U_att')] = normal_weight(alignment_dim, 1, scale=0.01)
    params[p_('decoder', 'Ub_att')] = np.zeros((1,), dtype=fX)

    # Map the output from decoder to the softmax layer
    params[p_('fc_compress_lastHiddenState', 'W')] = normal_weight(gru_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_lastHiddenState', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_compress_emb', 'W')] = normal_weight(embedding_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_emb', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_compress_ctx', 'W')] = normal_weight(2 * gru_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_ctx', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_to_softmax', 'W')] = normal_weight(embedding_dim, voc_size, scale=init_affine_weight)
    params[p_('fc_to_softmax', 'b')] = np.zeros((voc_size,), dtype=fX)

    return params


def init_GruWithImap():
    init_affine_weight, n_encoder_layer, n_decoder_layer, embedding_dim, gru_dim, alignment_dim, voc_size = \
        C['init_affine_weight'], C['n_encoder_layer'], C['n_decoder_layer'], C['dim_word'], C['dim'], C[
            'alignment_dim'], C['n_words']

    assert gru_dim % 2 == 0
    assert n_encoder_layer >= 1, '#LstmEncoderLayer must >= 1'
    assert n_decoder_layer >= 1, '#LstmDecoderLayer must >= 1'

    params = OrderedDict()

    # Embedding
    params['Wemb'] = normal_weight(voc_size, embedding_dim, scale=init_affine_weight)
    params['Wemb_dec'] = normal_weight(voc_size, embedding_dim, scale=init_affine_weight)

    # Encoder (forward and reverse)
    _param_init_gru_part(params, n_encoder_layer, gru_dim, 'encoder')
    _param_init_gru_part(params, n_encoder_layer, gru_dim, 'encoder_r')
    params[p_('encoder', 'Wf', 0)] = normal_weight(embedding_dim, 3 * gru_dim, scale=init_affine_weight)
    params[p_('encoder_r', 'Wf', 0)] = normal_weight(embedding_dim, 3 * gru_dim, scale=init_affine_weight)

    if C.get('upload_emb', False):
        params[p_('encoder', 'emb2first_W')] = normal_weight(embedding_dim, gru_dim, scale=init_affine_weight)
        params[p_('encoder_r', 'emb2first_W')] = normal_weight(embedding_dim, gru_dim, scale=init_affine_weight)
        params[p_('encoder', 'emb2first_b')] = np.zeros((gru_dim,), dtype=fX)
        params[p_('encoder_r', 'emb2first_b')] = np.zeros((gru_dim,), dtype=fX)
        params[p_('decoder', 'emb2first_W')] = normal_weight(embedding_dim, gru_dim, scale=init_affine_weight)
        params[p_('decoder', 'emb2first_b')] = np.zeros((gru_dim,), dtype=fX)

    for layer_id in xrange(1, n_encoder_layer):
        params[p_('encoder', 'Wf', layer_id)] = normal_weight(gru_dim, 3 * gru_dim, scale=init_affine_weight)
        params[p_('encoder_r', 'Wf', layer_id)] = normal_weight(gru_dim, 3 * gru_dim, scale=init_affine_weight)

    # Decoder
    _param_init_gru_part(params, n_decoder_layer, gru_dim, 'decoder')
    params[p_('decoder', 'Wf', 0)] = normal_weight(2 * gru_dim + embedding_dim, 3 * gru_dim, scale=init_affine_weight)
    for layer_id in xrange(1, n_decoder_layer):
        params[p_('decoder', 'Wf', layer_id)] = normal_weight(gru_dim, 3 * gru_dim, scale=0.01)

    # For the initial state of decoder
    params[p_('initDecoder', 'W')] = normal_weight(2 * gru_dim, gru_dim, scale=init_affine_weight)
    params[p_('initDecoder', 'b')] = np.zeros((gru_dim,), dtype='float32')

    # For attention model
    params[p_('decoder', 'We_att')] = normal_weight(2 * gru_dim, alignment_dim, scale=0.01)
    params[p_('decoder', 'Wh_att')] = normal_weight(gru_dim, alignment_dim, scale=0.01)
    params[p_('decoder', 'Wb_att')] = np.zeros((alignment_dim,), dtype=fX)
    params[p_('decoder', 'U_att')] = normal_weight(alignment_dim, 1, scale=0.01)
    params[p_('decoder', 'Ub_att')] = np.zeros((1,), dtype=fX)

    # Map the output from decoder to the softmax layer
    params[p_('fc_compress_lastHiddenState', 'W')] = normal_weight(gru_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_lastHiddenState', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_compress_emb', 'W')] = normal_weight(embedding_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_emb', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_compress_ctx', 'W')] = normal_weight(2 * gru_dim, embedding_dim, scale=init_affine_weight)
    params[p_('fc_compress_ctx', 'b')] = np.zeros((embedding_dim,), dtype=fX)
    params[p_('fc_to_softmax', 'W')] = normal_weight(embedding_dim, voc_size, scale=init_affine_weight)
    params[p_('fc_to_softmax', 'b')] = np.zeros((voc_size,), dtype=fX)

    return params
