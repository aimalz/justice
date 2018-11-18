# -*- coding: utf-8 -*-
"""Simple model that just returns a vector indicating whether this point is the max (or near it).

This is mostly for unit testing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from justice.align_model import graph_typecheck
from justice.features import band_settings_params


def _left_mask(before_padding, window_size):
    """Generates a mask for left-padded vectors.

    e.g. suppose left features are [0, 0, 0, x], where the "0" values are padding,
    so before_padding = 3. This function will return a mask [False, False, False, True].
    (in reality everything is vectorized by batch dimension.

    :param before_padding: [batch_size] tensor of before_padding values.
    :param window_size: scalar window size.
    :return: [batch_size, window_size] boolean tensor mask.
    """
    return tf.logical_not(tf.sequence_mask(before_padding, maxlen=window_size))


def _right_mask(after_padding, window_size):
    """Same as above, but for right-padded vectors."""
    return tf.sequence_mask(window_size - after_padding, maxlen=window_size)


def masked_sigmoid(sigmoid_input, mask):
    assert len(sigmoid_input.shape) == 2
    sigmoid_values = tf.sigmoid(sigmoid_input)
    assert list(map(int, sigmoid_input.shape)) == list(map(int, mask.shape))
    return tf.where(mask, sigmoid_values, tf.ones_like(sigmoid_values))


def per_band_model_fn(band_features, params, debug_print=False):
    # NOTE(gatoatigrado): dense_extracted_features.WindowFeatures provides a convenient
    # API for many calculations here, but right now the unit test data for max_model_kernel_test
    # does not provide time tensors, making it inconvenient to modernize this code.
    batch_size = params["batch_size"]
    window_size = params["window_size"]
    inv_eps = 1.0 / params["flux_scale_epsilon"]
    graph_typecheck.assert_shape(band_features["before_padding"], [batch_size])
    graph_typecheck.assert_shape(band_features["after_padding"], [batch_size])
    graph_typecheck.assert_shape(band_features["closest_flux"], [batch_size])

    before_flux = graph_typecheck.assert_shape(
        band_features["before_flux"], [batch_size, window_size]
    )
    after_flux = graph_typecheck.assert_shape(
        band_features["after_flux"], [batch_size, window_size]
    )

    # Return a soft-greater-than operator, product that all scores are greater.
    is_greater_than_before = masked_sigmoid(
        inv_eps * (tf.expand_dims(band_features["closest_flux"], axis=1) - before_flux),
        _left_mask(band_features["before_padding"], window_size)
    )
    is_greater_than_after = masked_sigmoid(
        inv_eps * (tf.expand_dims(band_features["closest_flux"], axis=1) - after_flux),
        _right_mask(band_features["after_padding"], window_size)
    )
    graph_typecheck.assert_shape(is_greater_than_before, [batch_size, window_size])
    graph_typecheck.assert_shape(is_greater_than_after, [batch_size, window_size])
    if debug_print:
        is_greater_than_before = graph_typecheck.print_single(
            is_greater_than_before, "is_greater_than_before:"
        )
        is_greater_than_after = graph_typecheck.print_single(
            is_greater_than_after, "is_greater_than_after:"
        )
    return tf.reduce_prod(
        is_greater_than_before, axis=1
    ) * tf.reduce_prod(
        is_greater_than_after, axis=1
    )


def feature_model_fn(features, params):
    band_settings = band_settings_params.BandSettings.from_params(params)
    per_band_data = band_settings.per_band_sub_model_fn(
        per_band_model_fn, features, params=params
    )
    result = tf.stack(per_band_data, axis=1)
    graph_typecheck.assert_shape(result, [params["batch_size"], band_settings.nbands])
    return {"is_max_soft": result}
