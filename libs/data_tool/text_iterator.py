#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import numpy as np

from ..utils.path import f_open

__author__ = 'fyabc'


class TextIterator(object):
    """The text iterator of NMT input data."""

    UNK = 1

    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128, maxlen=100,
                 n_words_source=-1, n_words_target=-1):
        self.source = f_open(source, mode='r', unpickle=False)
        self.target = f_open(target, mode='r', unpickle=False)
        self.source_dict = f_open(source_dict, mode='rb', unpickle=True)
        self.target_dict = f_open(target_dict, mode='rb', unpickle=True)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * 40

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.end_of_data = False

    def next(self):
        if self.end_of_data:
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if not self.source_buffer:
            for k_ in xrange(self.k):
                s = self.source.readline()
                if s == '':
                    break
                t = self.target.readline()
                if t == '':
                    break

                self.source_buffer.append(s.strip().split())
                self.target_buffer.append(t.strip().split())

            # sort by target buffer
            t_len = np.array([len(t) for t in self.target_buffer])
            t_idx = t_len.argsort()

            self.source_buffer = [self.source_buffer[i] for i in t_idx]
            self.target_buffer = [self.target_buffer[i] for i in t_idx]

        if not self.source_buffer or not self.target_buffer:
            self.reset()
            raise StopIteration

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                if not self.source_buffer:
                    break

                s = self.source_buffer.pop()
                s = (self.source_dict.get(w, self.UNK) for w in s)

                if self.n_words_source > 0:
                    s = [w if w < self.n_words_source else 1 for w in s]
                else:
                    s = list(s)

                t = self.target_buffer.pop()
                t = (self.target_dict.get(w, self.UNK) for w in t)

                if self.n_words_target > 0:
                    t = [w if w < self.n_words_target else 1 for w in t]
                else:
                    s = list(t)

                if len(s) > self.maxlen and len(t) > self.maxlen:
                    continue

                source.append(s)
                target.append(t)

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break

        except IOError:
            self.end_of_data = True

        if not source or not target:
            self.reset()
            raise StopIteration

        return np.array(source, dtype='int64'), np.array(target, dtype='int64')
