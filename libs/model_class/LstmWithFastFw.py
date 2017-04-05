#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict

import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T

from .base import Model
from ..utils.basic import p_, fX, slice_
from ..utils.my_math import *

__author__ = 'fyabc'


# LSTM init helper functions.
def _param_init_lstm_part(params, m_layer, lstm_dim, prefix, O):
    for layer_id in xrange(m_layer):
        params[p_(prefix, 'Wr', layer_id)] = orthogonal_weight_1xb(lstm_dim, 4)
        params[p_(prefix, 'br', layer_id)] = np.zeros((4 * lstm_dim,), dtype=fX)
        if O['use_theta']:
            params[p_(prefix, 'theta_rho', layer_id)] = np.zeros((lstm_dim,), dtype=fX)  # norm_vector(lstm_dim)
            params[p_(prefix, 'theta_phi', layer_id)] = np.zeros((lstm_dim,), dtype=fX)
            params[p_(prefix, 'theta_pi', layer_id)] = np.zeros((lstm_dim,), dtype=fX)


# LSTM step functions.

# Internal function: step forward, for LSTM + Fast-Forward [Baidu's paper]
def _lstm_step_slice_withTheta(src_mask, ft, ht_1, st_1, Wr, br, theta_rho, theta_phi, theta_pi):
    z_vector = ft + T.dot(ht_1, Wr) + br
    _dim = Wr.shape[1] // 4
    z = slice_(z_vector, 0 , _dim)
    z_rho = slice_(z_vector, 1 , _dim)
    z_phi = slice_(z_vector, 2 , _dim)
    z_pi = slice_(z_vector, 3 , _dim)
    st = T.tanh(z) * T.nnet.sigmoid(z_rho + st_1 * theta_rho[None, :]) + \
        T.nnet.sigmoid(z_phi +  st_1 * theta_phi[None, :]) * st_1
    ht = T.tanh(st) * T.nnet.sigmoid(z_pi + st * theta_pi[None, :])

    st = src_mask[:, None] * st + (1. - src_mask)[:, None] * st_1
    ht = src_mask[:, None] * ht + (1. - src_mask)[:, None] * ht_1

    return ht, st


def _lstm_step_slice(src_mask, ft, ht_1, st_1, Wr, br):
    z_vector = ft + T.dot(ht_1, Wr) + br
    _dim = Wr.shape[1] // 4
    z = slice_(z_vector, 0 , _dim)
    z_rho = slice_(z_vector, 1 , _dim)
    z_phi = slice_(z_vector, 2 , _dim)
    z_pi = slice_(z_vector, 3 , _dim)
    st = T.tanh(z) * T.nnet.sigmoid(z_rho) + T.nnet.sigmoid(z_phi) * st_1
    ht = T.tanh(st) * T.nnet.sigmoid(z_pi)

    st = src_mask[:, None] * st + (1. - src_mask)[:, None] * st_1
    ht = src_mask[:, None] * ht + (1. - src_mask)[:, None] * ht_1

    return ht, st


def _lstm_att_step_1stlayer_withTheta(yemb, ymask,
                                      ht_1, st_1, global_f, ctx_t,
                                      et, context_mask, Wf, Wr, br, theta_rho, theta_phi, theta_pi,
                                      Wp_compress_e, bp_compress_e, We_att, Wh_att, Wb_att, U_att, Ub_att):
    ctx_t = Model.attention(context_mask, et, ht_1, We_att, Wh_att, Wb_att, U_att, Ub_att)
    ctx_t = T.dot(ctx_t, Wp_compress_e) + bp_compress_e
    global_f = T.dot(concatenate([ctx_t, yemb], axis=-1), Wf)
    ht, st = _lstm_step_slice_withTheta(ymask, global_f, ht_1, st_1, Wr, br, theta_rho, theta_phi, theta_pi)
    return ht, st, global_f, ctx_t


def _lstm_att_step_1stlayer(yemb, ymask,
                            ht_1, st_1, global_f, ctx_t,
                            et, context_mask, Wf, Wr, br, Wp_compress_e, bp_compress_e, We_att, Wh_att, Wb_att,
                            U_att, Ub_att):
    ctx_t = Model.attention(context_mask, et, ht_1, We_att, Wh_att, Wb_att, U_att, Ub_att)
    ctx_t = T.dot(ctx_t, Wp_compress_e) + bp_compress_e
    global_f = T.dot(concatenate([ctx_t, yemb], axis=-1), Wf)
    ht, st = _lstm_step_slice(ymask, global_f, ht_1, st_1, Wr, br)
    return ht, st, global_f, ctx_t


class LstmWithFastFwModel(Model):
    def init_np_parameters(self):
        np_params = OrderedDict()

        init_affine_weight = self.O['init_affine_weight']
        n_encoder_layer, n_decoder_layer, embedding_dim, lstm_dim, alignment_dim, voc_size = \
            self.O['n_encoder_layer'], self.O['n_decoder_layer'], self.O['dim_word'], self.O['dim'], \
            self.O['alignment_dim'], self.O['n_words_tgt']

        assert lstm_dim % 2 == 0
        assert n_encoder_layer >= 1, '#LstmEncoderLayer must >= 1'
        assert n_decoder_layer >= 1, '#LstmDecoderLayer must >= 1'

        # Embedding
        np_params['Wemb'] = normal_weight(voc_size, embedding_dim, scale=init_affine_weight)
        np_params['Wemb_dec'] = normal_weight(voc_size, embedding_dim, scale=init_affine_weight)

        # Encoder (forward)
        _param_init_lstm_part(np_params, n_encoder_layer, lstm_dim, 'encoder', self.O)
        np_params[p_('encoder', 'Wf', 0)] = normal_weight(embedding_dim, 4 * lstm_dim, scale=init_affine_weight)
        for layer_id in xrange(1, n_encoder_layer):
            np_params[p_('encoder', 'Wf', layer_id)] = normal_weight(
                (1 + 4 / (1 + self.O['use_half'])) * lstm_dim, 4 * lstm_dim, scale=init_affine_weight)

        # Encoder (Reverse)
        _param_init_lstm_part(np_params, n_encoder_layer, lstm_dim, 'encoder_r', self.O)
        np_params[p_('encoder_r', 'Wf', 0)] = normal_weight(embedding_dim, 4 * lstm_dim, scale=init_affine_weight)
        for layer_id in xrange(1, n_encoder_layer):
            np_params[p_('encoder_r', 'Wf', layer_id)] = normal_weight(
                (1 + 4 / (1 + self.O['use_half'])) * lstm_dim, 4 * lstm_dim, scale=0.01)

        # Decoder
        _param_init_lstm_part(np_params, n_decoder_layer, lstm_dim, 'decoder', self.O)
        np_params[p_('decoder', 'Wf', 0)] = normal_weight(embedding_dim + lstm_dim * 10 / 4, 4 * lstm_dim,
                                                          scale=init_affine_weight)
        for layer_id in xrange(1, n_decoder_layer):
            np_params[p_('decoder', 'Wf', layer_id)] = normal_weight(
                (1 + 4 / (1 + self.O['use_half'])) * lstm_dim, 4 * lstm_dim, scale=init_affine_weight)

        # self.Oompress the output of the encoder to 1/4
        np_params[p_('decoder', 'Wp_compress_e')] = normal_weight(10 * lstm_dim, 10 * lstm_dim / 4,
                                                                  scale=init_affine_weight)
        np_params[p_('decoder', 'bp_compress_e')] = np.zeros((10 * lstm_dim / 4,), dtype=fX)

        # For the initial state of decoder
        np_params[p_('initDecoder', 'W')] = normal_weight(10 * lstm_dim, lstm_dim, scale=init_affine_weight)
        np_params[p_('initDecoder', 'b')] = np.zeros((lstm_dim,), dtype=fX)

        # For attention model
        np_params[p_('decoder', 'We_att')] = normal_weight(10 * lstm_dim, alignment_dim, scale=0.01)
        np_params[p_('decoder', 'Wh_att')] = normal_weight(lstm_dim, alignment_dim, scale=0.01)
        np_params[p_('decoder', 'Wb_att')] = np.zeros((alignment_dim,), dtype=fX)
        np_params[p_('decoder', 'U_att')] = normal_weight(alignment_dim, 1, scale=0.01)
        np_params[p_('decoder', 'Ub_att')] = np.zeros((1,), dtype=fX)

        # Map the output from decoder to the softmax layer
        np_params[p_('fc_compress_lastHiddenState', 'W')] = normal_weight(lstm_dim, embedding_dim,
                                                                          scale=init_affine_weight)
        np_params[p_('fc_compress_lastHiddenState', 'b')] = np.zeros((embedding_dim,), dtype=fX)
        np_params[p_('fc_compress_emb', 'W')] = normal_weight(embedding_dim, embedding_dim, scale=init_affine_weight)
        np_params[p_('fc_compress_emb', 'b')] = np.zeros((embedding_dim,), dtype=fX)
        np_params[p_('fc_compress_ctx', 'W')] = normal_weight(10 * lstm_dim / 4, embedding_dim,
                                                              scale=init_affine_weight)
        np_params[p_('fc_compress_ctx', 'b')] = np.zeros((embedding_dim,), dtype=fX)
        np_params[p_('fc_to_softmax', 'W')] = normal_weight(embedding_dim, voc_size, scale=init_affine_weight)
        np_params[p_('fc_to_softmax', 'b')] = np.zeros((voc_size,), dtype=fX)

        return np_params

    def build_model(self, **kwargs):
        self.opt_ret.clear()

        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(1.)))
        trng = kwargs.pop('trng', RandomStreams(self.O['seed']))

        dropout_param = None
        if self.O['use_dropout'][0]:
            dropout_param = [use_noise, trng, self.O['use_dropout'][1]]

        x, x_mask, y, y_mask = self.get_input()
        xr, xr_mask = self.reverse_input(x, x_mask)

        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        # Word embedding for forward rnn (source)
        emb = self.embedding(x, n_timestep, n_samples)
        proj_f = self._encoder(emb, 'encoder', mask=x_mask, dropout_param=dropout_param)

        # Word embedding for backward rnn (source)
        embr = self.embedding(xr, n_timestep, n_samples)
        proj_r = self._encoder(embr, 'encoder', mask=xr_mask, dropout_param=dropout_param)

        # Context will be the concatenation of forward and backward RNNs
        ctx = concatenate([proj_f[0], proj_r[0][::-1], proj_f[1], proj_r[1][::-1]], axis=proj_f[0].ndim - 1)

        # Mean of the context across time, which will be used to initialize decoder LSTM. This is the original code
        ctx_mean = self.get_context_mean(ctx, x_mask)

        # Initial decoder state
        initial_decoder_h = self.fully_connect(ctx_mean, 'initDecoder', T.tanh)

        # Word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        emb = self.embedding(y, n_timestep_tgt, n_samples, 'Wemb_dec')
        emb_shifted = T.zeros_like(emb)
        emb_shifted = T.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        hidden_from_last_layer, ctx_from_1st_layer = self._decoder(
            emb, y_mask, ctx, x_mask, initial_decoder_h,
            prefix='decoder', one_step=False, dropout_param=dropout_param,
        )

        # As suggested in Page 14 of the NMT + Attention model paper, let us implement the equation above section A.2.3
        fc_hidden = self.fully_connect(hidden_from_last_layer, prefix='fc_compress_lastHiddenState', activ='linear')
        fc_emb = self.fully_connect(emb, prefix='fc_compress_emb', activ='linear')
        fc_ctx = self.fully_connect(ctx_from_1st_layer, prefix='fc_compress_ctx', activ='linear')

        fc_sum = T.tanh(fc_hidden + fc_emb + fc_ctx)

        # According to Baidu's paper, dropout is only used in LSTM. So I drop the following two lines out (v-yixia)
        # if self.O['use_dropout'][0]:
        #    fc_sum = self.dropout(fc_sum, use_noise, trng, self.O['use_dropout'][1])

        softmax_output = self.fully_connect(fc_sum, prefix='fc_to_softmax', activ='linear')
        softmax_output_shp = softmax_output.shape
        probs = T.nnet.softmax(softmax_output.reshape([softmax_output_shp[0] * softmax_output_shp[1],
                                                       softmax_output_shp[2]]))

        cost = self.get_cost(y, y_mask, probs)

        return x, x_mask, y, y_mask, cost

    def _encoder(self, src_word_embedding, prefix='encoder', mask=None, dropout_param=None):
        m_layer = self.O['n_encoder_layer']
        use_zigzag = self.O['use_zigzag']
        assert m_layer >= 1, '#layer should greater than 1'
        if self.O['use_theta']:
            _lstm_step_f = _lstm_step_slice_withTheta
        else:
            _lstm_step_f = _lstm_step_slice

        nsteps = src_word_embedding.shape[0]
        if src_word_embedding.ndim == 3:
            n_samples = src_word_embedding.shape[1]
        else:
            n_samples = 1
        if mask is None:
            mask = T.alloc(1., src_word_embedding.shape[0], 1)

        global_f = src_word_embedding
        _hidden_state_lastLayer = None

        for layer_id in xrange(m_layer):
            Wf = self.P[p_(prefix, 'Wf', layer_id)]
            if layer_id > 0:
                if self.O['use_half']:
                    tdim = global_f.shape[2] // 2
                    global_f = concatenate([global_f[:, :, 0: tdim], _hidden_state_lastLayer[0]], axis=-1)
                else:
                    global_f = concatenate([global_f, _hidden_state_lastLayer[0]], axis=-1)
            global_f = T.dot(global_f, Wf)
            dim = Wf.shape[1] / 4
            init_states = [T.alloc(0., n_samples, dim), T.alloc(0., n_samples, dim)]
            if self.O['use_theta']:
                shared_vars = [self.P[p_(prefix, 'Wr', layer_id)],
                               self.P[p_(prefix, 'br', layer_id)],
                               self.P[p_(prefix, 'theta_rho', layer_id)],
                               self.P[p_(prefix, 'theta_phi', layer_id)],
                               self.P[p_(prefix, 'theta_pi', layer_id)]]
            else:
                shared_vars = [self.P[p_(prefix, 'Wr', layer_id)],
                               self.P[p_(prefix, 'br', layer_id)]]

            seqs = [mask, global_f]

            _hidden_state_lastLayer, _ = theano.scan(
                fn=_lstm_step_f,
                sequences=seqs,
                outputs_info=init_states,
                non_sequences=shared_vars,
                name=p_(prefix, 'layer', layer_id),
                n_steps=nsteps,
                profile=self.O['profile'],
                strict=True,
            )

            if layer_id < m_layer - 1 and use_zigzag:
                _hidden_state_lastLayer[0] = _hidden_state_lastLayer[0][::-1]
                mask = mask[::-1]
                global_f = global_f[::-1]

            if dropout_param:
                _hidden_state_lastLayer[0] = self.dropout(_hidden_state_lastLayer[0], dropout_param[0],
                                                          dropout_param[1], dropout_param[2])

        if use_zigzag and m_layer % 2 == 0:
            return [global_f[::-1], _hidden_state_lastLayer[0][::-1]]
        else:
            return [global_f, _hidden_state_lastLayer[0]]

    def _decoder(self, dst_word_embedding, dst_mask, context, context_mask, previous_h,
                 prefix='decoder', one_step=False, previous_s=None, dropout_param=None):
        # From v-yixia
        # TODO: Currently, I do not return the alpha for further alignment.
        # You can implement it easily bu inserting a dummy position in _lstm_att_step_1step_layer and get it.
        # TODO: I still write an "if ... else..." for one_step
        assert context, 'Context must be provided'

        assert context.ndim == 3, 'Context must be 3-d: #time_step x #sample x dim'
        if one_step:
            assert previous_h, 'previous_h must be provided'
            assert previous_s, 'previous_s must be provided'
            assert previous_h.ndim == 3, 'previous_h.ndim == 3'
            assert previous_s.ndim == 3, 'previous_s.ndim == 3'

        m_layer = self.O['n_decoder_layer']
        if self.O['use_theta']:
            _lstm_step_f = _lstm_step_slice_withTheta
        else:
            _lstm_step_f = _lstm_step_slice

        # First, we deal with the first layer with attention model
        nsteps = dst_word_embedding.shape[0]
        if dst_word_embedding.ndim == 3:
            n_samples = dst_word_embedding.shape[1]
        else:
            n_samples = 1

        if dst_mask is None:
            dst_mask = T.alloc(1., dst_word_embedding.shape[0], )

        # First, let us deal with the first layer
        hidden_dim = self.P[p_(prefix, 'Wf', 0)].shape[
                         -1] // 4  # self.P[p_(prefix, 'theta_rho', 0)].shape[0]
        non_sequence_vars = [context, context_mask,
                              self.P[p_(prefix, 'Wf', 0)],
                              self.P[p_(prefix, 'Wr', 0)],
                              self.P[p_(prefix, 'br', 0)],
                              self.P[p_(prefix, 'Wp_compress_e')],
                              self.P[p_(prefix, 'bp_compress_e')],
                              self.P[p_(prefix, 'We_att')],
                              self.P[p_(prefix, 'Wh_att')],
                              self.P[p_(prefix, 'Wb_att')],
                              self.P[p_(prefix, 'U_att')],
                              self.P[p_(prefix, 'Ub_att')]]
        _lstm_first_step = _lstm_att_step_1stlayer
        if self.O['use_theta']:
            non_sequence_vars[5:5] = [self.P[p_(prefix, 'theta_rho', 0)],
                                       self.P[p_(prefix, 'theta_phi', 0)],
                                       self.P[p_(prefix, 'theta_pi', 0)]]
            _lstm_first_step = _lstm_att_step_1stlayer_withTheta

        input_seqs = [dst_word_embedding, dst_mask]

        if one_step:
            _hidden_state_lastLayer = list(
                _lstm_first_step(*(input_seqs + [previous_h[0], previous_s[0], None, None] + non_sequence_vars)))
        else:
            output_seqs = [previous_h,
                           T.alloc(0., n_samples, hidden_dim),
                           # Please note that this is the initial st_1 with all zeros; this is for training purpose only
                           T.alloc(0., n_samples, 4 * hidden_dim),
                           T.alloc(0., n_samples, 10 * hidden_dim / 4)]
            _hidden_state_lastLayer, _ = theano.scan(fn=_lstm_first_step,
                                                     sequences=input_seqs,
                                                     outputs_info=output_seqs,
                                                     non_sequences=non_sequence_vars,
                                                     name=p_(prefix, 'layer', 0),
                                                     n_steps=nsteps,
                                                     profile=self.O['profile'],
                                                     strict=True)

        if dropout_param:
            _hidden_state_lastLayer[0] = self.dropout(_hidden_state_lastLayer[0], dropout_param[0], dropout_param[1],
                                                      dropout_param[2])

        # Next, we deal with layer > 1, which has no attention model
        global_f = _hidden_state_lastLayer[2]
        ctx_from_1stlayer = _hidden_state_lastLayer[3]

        if one_step:
            stack_h = _hidden_state_lastLayer[0].dimshuffle('x', 0, 1)
            stack_s = _hidden_state_lastLayer[1].dimshuffle('x', 0, 1)
            for layer_id in xrange(1, m_layer):
                Wf = self.P[p_(prefix, 'Wf', layer_id)]
                if self.O['use_half']:
                    tdim = global_f.shape[1] // 2
                    global_f = concatenate([global_f[:, 0: tdim], stack_h[-1]], axis=-1)
                else:
                    global_f = concatenate([global_f, stack_h[-1]], axis=-1)
                global_f = T.dot(global_f, Wf)
                init_states = [previous_h[layer_id], previous_s[layer_id]]
                if self.O['use_theta']:
                    shared_vars = [self.P[p_(prefix, 'Wr', layer_id)],
                                   self.P[p_(prefix, 'br', layer_id)],
                                   self.P[p_(prefix, 'theta_rho', layer_id)],
                                   self.P[p_(prefix, 'theta_phi', layer_id)],
                                   self.P[p_(prefix, 'theta_pi', layer_id)]]
                else:
                    shared_vars = [self.P[p_(prefix, 'Wr', layer_id)],
                                   self.P[p_(prefix, 'br', layer_id)]]
                seqs = [dst_mask, global_f]
                _hidden_state_lastLayer = list(_lstm_step_f(*(seqs + init_states + shared_vars)))
                if dropout_param:
                    _hidden_state_lastLayer[0] = self.dropout(_hidden_state_lastLayer[0], dropout_param[0],
                                                              dropout_param[1], dropout_param[2])
                stack_h = concatenate([stack_h, _hidden_state_lastLayer[0].dimshuffle('x', 0, 1)])
                stack_s = concatenate([stack_s, _hidden_state_lastLayer[1].dimshuffle('x', 0, 1)])
            return stack_h, stack_s, ctx_from_1stlayer
        else:
            for layer_id in xrange(1, m_layer):
                Wf = self.P[p_(prefix, 'Wf', layer_id)]
                if self.O['use_half']:
                    tdim = global_f.shape[2] // 2
                    global_f = concatenate([global_f[:, :, 0: tdim], _hidden_state_lastLayer[0]], axis=-1)
                else:
                    global_f = concatenate([global_f, _hidden_state_lastLayer[0]], axis=-1)
                global_f = T.dot(global_f, Wf)
                dim = Wf.shape[1] / 4
                init_states = [previous_h, T.alloc(0., n_samples, dim)]
                if self.O['use_theta']:
                    shared_vars = [self.P[p_(prefix, 'Wr', layer_id)],
                                   self.P[p_(prefix, 'br', layer_id)],
                                   self.P[p_(prefix, 'theta_rho', layer_id)],
                                   self.P[p_(prefix, 'theta_phi', layer_id)],
                                   self.P[p_(prefix, 'theta_pi', layer_id)]]
                else:
                    shared_vars = [self.P[p_(prefix, 'Wr', layer_id)],
                                   self.P[p_(prefix, 'br', layer_id)]]

                seqs = [dst_mask, global_f]
                _hidden_state_lastLayer, _ = theano.scan(fn=_lstm_step_f,
                                                         sequences=seqs,
                                                         outputs_info=init_states,
                                                         non_sequences=shared_vars,
                                                         name=p_(prefix, 'layer', layer_id),
                                                         n_steps=nsteps,
                                                         profile=self.O['profile'],
                                                         strict=True)

                if dropout_param:
                    _hidden_state_lastLayer[0] = self.dropout(_hidden_state_lastLayer[0], dropout_param[0],
                                                              dropout_param[1], dropout_param[2])

        return _hidden_state_lastLayer[0], ctx_from_1stlayer


LstmWithFastFwModel.register_class(['LstmWithFastFw'])
