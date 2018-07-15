# -*- coding: utf-8 -*-
"""Unit tests for the training function."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

from justice.similarity_model import training_data


def test_prefix_function():
    time_samples = np.array([0, 1, 2, 3, 4, 10, 11, 12, 12, 13, 14], dtype=np.float64)
    diffs = np.array([1, 1, 1, 1, 6, 1, 1, 0, 1, 1], dtype=np.float64)
    assert (diffs == (time_samples[1:] - time_samples[:-1])).all()

    def get_windowed_samples(window_size):
        start_indices = training_data.get_window_valid_indices(diffs, 0.8, 1.2, window_size)
        results = []
        for start_idx in start_indices:
            results.append(map(int, time_samples[start_idx:(start_idx + window_size)]))
        return results

    assert get_windowed_samples(1) == [[0], [1], [2], [3], [4], [10], [11], [12], [12], [13], [14]]
    assert get_windowed_samples(3) == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [10, 11, 12], [12, 13, 14]]
    assert get_windowed_samples(4) == [[0, 1, 2, 3], [1, 2, 3, 4]]


def test_positive_negative_mix():
    with tf.Graph().as_default() as g:
        dataset = training_data.sample_data_input_fn({
            'window_size': 5,
            'batch_size': 8,
        })
        all_counts = set()
        with tf.Session(graph=g) as sess:
            tensors = dataset.make_one_shot_iterator().get_next()
            for _ in xrange(10):
                results = sess.run(tensors)
                counts = tuple(sorted(collections.Counter(results['goal']).items()))
                all_counts.add(counts)

        # Each batch is either all positives or negatives.
        assert all_counts == {((0.0, 8),), ((1.0, 8),)}
