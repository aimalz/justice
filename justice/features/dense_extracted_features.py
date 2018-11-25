# -*- coding: utf-8 -*-
"""Extracts dense features with linear transformations."""
import json
import pathlib
import typing

import numpy as np
import tensorflow as tf

from justice import path_util
from justice.align_model import graph_typecheck
from justice.features import band_settings_params


def _left_mask(before_padding, window_size):
    """Generates a mask for left-padded vectors.

    Mask elements are True if a valid element is present.

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

    def __init__(
        self,
        band_features: dict,
        batch_size: int,
        window_size: int,
        band_time_diff: float = 4.0
    ):
        """Initializes a windowed feature extractor.

        :param band_features: Band features, generated by raw_value_features.
        :param batch_size: Outer batch size.
        :param window_size: Number of points in 'before' and 'after' windows.
        :param band_time_diff: Maximum difference between requested time and actual time
            in the window.
        """

        def batch_shaped(t):
            return graph_typecheck.assert_shape(t, [batch_size])

        def batch_win_shaped(t):
            return graph_typecheck.assert_shape(t, [batch_size, window_size])

        def batch_2win_shaped(t):
            return graph_typecheck.assert_shape(t, [batch_size, 2 * window_size])

        def tile_to_2win(t):
            return tf.tile(tf.expand_dims(t, 1), [1, 2 * window_size])

        closest_time = batch_shaped(band_features['closest_time_in_band'])
        closest_flux = batch_shaped(band_features['closest_flux_in_band'])
        self.in_window = tf.less(
            batch_shaped(band_features["closest_time_diff"]), band_time_diff
        )

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

        self.mask = batch_2win_shaped(
            tf.logical_and(
                tf.concat([left_mask, right_mask], axis=1), tile_to_2win(self.in_window)
            )
        )

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


def initial_layer(
    window_feature: WindowFeatures, *, clip_magnitude=10.0, include_flux_and_time=False
) -> tf.Tensor:
    features = tf.expand_dims(window_feature.dflux_dt(clip_magnitude=clip_magnitude), 2)
    if include_flux_and_time:
        dflux = tf.expand_dims(window_feature.dflux, 2)
        dtime = tf.expand_dims(window_feature.dtime, 2)
        features = tf.concat([features, dflux, dtime],
                             axis=2,
                             name="initial_layer_concat")
    return features


class CutoffData:
    def __init__(self, config_json: dict):
        self.window_size: int = config_json["window_size"]
        self.band_time_diff: int = config_json["band_time_diff"]
        self.embedding_size: int = config_json["desired_num_cutoffs"]
        self.models_by_band = {}
        for solution in config_json["solutions"]:
            band = solution["band"]
            self.models_by_band.setdefault(band, {})
            self.models_by_band[band][solution["column"]] = (
                solution["median_scale"],
                solution["cutoffs"],
            )

    def dflux_dt_dflux_dtime_scales(self, band: str, dtype=np.float32):
        """Generates a vector of scalar offsets.

        :param band: Band name.
        :param dtype: Data type of output array.
        :return: <dtype>[3] matrix of scales per channel.
        """
        return np.array([
            self.models_by_band[band]["dflux_dt"][0],
            self.models_by_band[band]["dflux"][0],
            self.models_by_band[band]["dtime"][0],
        ],
            dtype=dtype)

    def dflux_dt_dflux_dtime_cutoffs(self, band: str, dtype=np.float32):
        """Generates a matrix of [dflux_dt, dflux, dtime].

        :param band: Band name.
        :param dtype: Data type of output array.
        :return: <dtype>[3, self.embedding_size] matrix of cutoffs.
        """
        return np.array([
            self.models_by_band[band]["dflux_dt"][1],
            self.models_by_band[band]["dflux"][1],
            self.models_by_band[band]["dtime"][1],
        ],
            dtype=dtype)

    @classmethod
    def from_file(cls, filename: pathlib.Path):
        if not filename.is_file():
            raise EnvironmentError(
                "Please generate tf_align_model data using the tf_align_model_input_"
                "feature_percentiles.ipynb notebook or clone https://github.com/"
                "gatoatigrado/plasticc-generated-data into data/tf_align_model."
            )
        with open(str(filename)) as f:
            return cls(json.load(f))


def initial_layer_binned(
    initial_layer_features: tf.Tensor,
    cutoff_data: CutoffData,
    band: str,
    nonlinearity=tf.sigmoid
):
    batch_size, twice_window_size, channels = map(int, initial_layer_features.shape)
    if channels == 3:
        scales = cutoff_data.dflux_dt_dflux_dtime_scales(band)
        cutoffs = cutoff_data.dflux_dt_dflux_dtime_cutoffs(band)

        cutoffs_batch_window = tf.expand_dims(tf.expand_dims(cutoffs, 0), 0)
        scales_batch_window = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(scales, 0), 0), -1
        )
        init_layer_per_cutoff = tf.expand_dims(initial_layer_features, -1)
        graph_typecheck.assert_shape(
            cutoffs_batch_window, [1, 1, channels, cutoff_data.embedding_size]
        )
        graph_typecheck.assert_shape(scales_batch_window, [1, 1, channels, 1])
        graph_typecheck.assert_shape(
            init_layer_per_cutoff, [batch_size, twice_window_size, channels, 1]
        )
        result = nonlinearity(
            (init_layer_per_cutoff - cutoffs_batch_window) / scales_batch_window
        )
        return graph_typecheck.assert_shape(
            result, [batch_size, twice_window_size, channels, cutoff_data.embedding_size]
        )
    else:
        raise NotImplementedError(f"{channels}-size data not implemented.")


def cutoff_data_for_window_size(window_size):
    if window_size == 10:
        cutoff_data = CutoffData.from_file(
            path_util.tf_align_data / 'feature_extraction' /
            'cutoffs__window_sz-10__2018-11-23.json'
        )
    else:
        raise ValueError("No supported cutoff data for window size")
    return cutoff_data


def initial_layer_binned_defaults(
    band_features: dict,
    band: str,
    batch_size: int,
    window_size: int,
    value_if_masked: float = 0.0
):
    cutoff_data = cutoff_data_for_window_size(window_size)
    wf = WindowFeatures(band_features, batch_size=batch_size, window_size=window_size)
    init_layer = initial_layer(wf, include_flux_and_time=True)
    binned = initial_layer_binned(init_layer, cutoff_data=cutoff_data, band=band)
    masked = wf.masked(
        binned,
        value_if_masked=value_if_masked,
        expected_extra_dims=[3, cutoff_data.embedding_size]
    )
    return masked


def per_band_model_fn(band_features, band_name, params, value_if_masked: float = 0.0):
    batch_size = params["batch_size"]
    window_size = params["window_size"]
    return initial_layer_binned_defaults(
        band_features,
        band=band_name,
        batch_size=batch_size,
        window_size=window_size,
        value_if_masked=value_if_masked
    )


def feature_model_fn(features, params):
    band_settings = band_settings_params.BandSettings.from_params(params)
    per_band_data = band_settings.per_band_sub_model_fn_with_band_name(
        per_band_model_fn, features, params=params
    )
    return graph_typecheck.assert_shape(
        tf.stack(per_band_data, axis=4), [
            params["batch_size"],
            2 * params["window_size"],
            3,
            cutoff_data_for_window_size(params["window_size"]).embedding_size,
            band_settings.nbands,
        ]
    )
