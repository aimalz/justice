# -*- coding: utf-8 -*-
"""Extracts dense features with linear transformations."""
import typing

import tensorflow as tf

from justice.align_model import graph_typecheck


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


class WindowFeatures(object):
    """Helper for dealing with window-like raw features.

    In particular, this class generates concatenated "dflux_dt" values, and has a masking
    helper, which will probably be applied after doing some non-linear transformations to
    dflux_dt (or possibly raw self.dtime, self.dflux) values.
    """

    def __init__(self, band_features: dict, batch_size: int, window_size: int):
        def batch_shaped(t):
            return graph_typecheck.assert_shape(t, [batch_size])

        def batch_win_shaped(t):
            return graph_typecheck.assert_shape(t, [batch_size, window_size])

        def batch_2win_shaped(t):
            return graph_typecheck.assert_shape(t, [batch_size, 2 * window_size])

        def tile_to_2win(t):
            return tf.tile(tf.expand_dims(t, 1), [1, 2 * window_size])

        closest_time = batch_shaped(band_features['closest_time'])
        closest_flux = batch_shaped(band_features['closest_flux'])

        # Before and after flux.
        before_flux = batch_win_shaped(band_features["before_flux"])
        after_flux = batch_win_shaped(band_features["after_flux"])
        before_time = batch_win_shaped(band_features["before_time"])
        after_time = batch_win_shaped(band_features["after_time"])

        self.dtime = batch_2win_shaped(
            tf.concat([before_time, after_time], axis=1) - tile_to_2win(closest_time),
        )
        self.dflux = batch_2win_shaped(
            tf.concat([before_flux, after_flux], axis=1) - tile_to_2win(closest_flux),
        )

        # Masking tensor.
        left_mask = _left_mask(
            batch_shaped(
                band_features["before_padding"]),
            window_size)
        right_mask = _right_mask(
            batch_shaped(band_features["after_padding"]), window_size
        )
        self.mask = batch_2win_shaped(tf.concat([left_mask, right_mask], axis=1))

    def dflux_dt(self, clip_magnitude: typing.Optional[float]) -> tf.Tensor:
        """Computes dflux/dt.

        :param clip_magnitude: Option for clipping the magnitude, if dt might be very small.
        :return: <float>[batch_size, 2 * window_size] dflux/dt tensor.
        """
        result = self.dflux / self.dtime
        if clip_magnitude is not None:
            result = tf.clip_by_value(
                result, clip_value_min=-clip_magnitude, clip_value_max=clip_magnitude
            )
        return result

    def masked(
        self, expanded_tensor: tf.Tensor, value_if_masked: float,
        expected_extra_dims: typing.List[int]
    ):
        """Masks a tensor which was calculated from dflux_dt.

        :param expanded_tensor: <float>[batch_size, window_size, ...] Tensor with first
            dimensions being batch_size and window_size.
        :param value_if_masked: Value to fill for masked positions.
        :param expected_extra_dims: Expected extra dimensions.
        :returns: Tensor of same shape as expanded_tensor, but with `value_if_masked` filled
            in masked dimensions.
        """
        mask_shape = list(map(int, self.mask.shape))
        graph_typecheck.assert_shape(expanded_tensor, mask_shape + expected_extra_dims)

        value_if_masked = expanded_tensor.dtype.as_numpy_dtype(value_if_masked)
        if_masked_tensor = tf.fill(expanded_tensor.shape, value_if_masked)
        mask = self.mask
        for i in range(2, 2 + len(expected_extra_dims)):
            mask = tf.expand_dims(mask, axis=i)
        mask = tf.tile(mask, [1, 1] + expected_extra_dims)
        return tf.where(mask, expanded_tensor, if_masked_tensor)
