# -*- coding: utf-8 -*-
"""Unit tests for max model kernel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from justice.align_model import max_model_kernel


def test_per_band_model_fn_not_max():
    band_features = {
        "before_padding": 3,
        "before_flux": [0, 0, 0, 1, 2, 1],
        "after_flux": [0.5, 0.1, 0, 0, 0, 0],
        "after_padding": 4,
        "closest_flux": 0.75,
    }
    params = {
        "batch_size": 1,
        "window_size": 6,
        "flux_scale_epsilon": 0.01,
    }
    as_tensors = {
        key: tf.constant([value], dtype=tf.float32)
        for key, value in band_features.items()
    }
    max_value = max_model_kernel.per_band_model_fn(as_tensors, params)
    with tf.Session() as sess:
        value, = sess.run(max_value)
    assert abs(value) < 1e-10  # Should be close to zero.


def test_per_band_model_fn_is_max():
    band_features = {
        "before_padding": 3,
        "before_flux": [0, 0, 0, 1, 2, 1],
        "after_flux": [0.5, 0.1, 0, 0, 0, 0],
        "after_padding": 4,
        "closest_flux": 2.75,
    }
    params = {
        "batch_size": 1,
        "window_size": 6,
        "flux_scale_epsilon": 0.01,
    }
    as_tensors = {
        key: tf.constant([value], dtype=tf.float32)
        for key, value in band_features.items()
    }
    max_value = max_model_kernel.per_band_model_fn(as_tensors, params)
    with tf.Session() as sess:
        value, = sess.run(max_value)
    assert 0.99 < abs(value) < 1.01  # should be close to 1
