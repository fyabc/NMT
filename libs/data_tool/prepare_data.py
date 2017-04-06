#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import numpy as np

from ..utils.basic import fX

__author__ = 'fyabc'


def prepare_data(xs, ys, maxlen=None, n_words_src=30000, n_words=30000):
    """Batch preparation of NMT data.

    This swap the axis!

    Parameters
    ----------
    xs: a list of source sentences
    ys: a list of target sentences
    maxlen: max length of sentences.

    Returns
    -------
    x, x_mask, y, y_mask: numpy arrays (maxlen * n_samples)
    """

    x_lens = [len(s) for s in xs]
    y_lens = [len(s) for s in ys]

    # Filter long sentences.
    if maxlen is not None:
        xs_new, ys_new = [], []
        x_lens_new, y_lens_new = [], []

        for lx, sx, ly, sy in zip(x_lens, xs, y_lens, ys):
            if lx < maxlen and ly < maxlen:
                xs_new.append(sx)
                x_lens_new.append(lx)
                ys_new.append(sy)
                y_lens_new.append(ly)

        xs, x_lens, ys, y_lens = xs_new, x_lens_new, ys_new, y_lens_new

        if not x_lens or not y_lens:
            return None, None, None, None

    n_samples = len(xs)
    maxlen_x = np.max(x_lens) + 1
    maxlen_y = np.max(y_lens) + 1

    x = np.zeros((maxlen_x, n_samples), dtype='int64')
    y = np.zeros((maxlen_y, n_samples), dtype='int64')
    x_mask = np.zeros((maxlen_x, n_samples), dtype=fX)
    y_mask = np.zeros((maxlen_y, n_samples), dtype=fX)

    for i, (sx, sy) in enumerate(zip(xs, ys)):
        x[:x_lens[i], i] = sx
        x_mask[:x_lens[i] + 1, i] = 1.
        y[:y_lens[i], i] = sy
        y_mask[:y_lens[i] + 1, i] = 1.

    return x, x_mask, y, y_mask
