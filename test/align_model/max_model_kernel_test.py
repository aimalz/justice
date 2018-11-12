# -*- coding: utf-8 -*-
"""Unit tests for max model kernel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf

from justice.align_model import max_model_kernel


@pytest.fixture(autouse=True)
def tf_graph():
    with tf.Graph().as_default():
        yield


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


def test_feature_model_fn():
    def get_band_features(band_name):
        return {
            "before_padding": 3,
            "before_flux": [0, 0, 0, 1, 2, 1],
            "after_flux": [0.5, 0.1, 0, 0, 0, 0],
            "after_padding": 4,
            "closest_flux": 0.75 if band_name == 'a' else 2.75
        }

    params = {
        "batch_size": 3,
        "window_size": 6,
        "flux_scale_epsilon": 0.01,
        "lc_bands": ['a', 'b'],
    }
    as_tensors = {
        f"band_{band_name}.{key}": tf.constant([value, value, value], dtype=tf.float32)
        for band_name in ['a', 'b'] for key, value in get_band_features(band_name).items()
    }
    max_kernel_features = max_model_kernel.feature_model_fn(
        features=as_tensors, params=params
    )
    with tf.Session() as sess:
        max_kernel_features = sess.run(max_kernel_features)
    assert np.round(max_kernel_features["is_max_soft"]).tolist() == [[0.0, 1.0]] * 3
