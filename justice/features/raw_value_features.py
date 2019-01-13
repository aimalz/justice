# -*- coding: utf-8 -*-
"""Extracts raw values around a point.

There's some complexity in what should happen when the requested time is different from
the actual times available in the desired band: do we generate a pseudo-point or re-center
around a nearby one? For now we'll do the latter, and possibly throw away all data for
this band if it's simply too far away, but we should revisit this decision.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from justice import lightcurve
from justice.features import band_settings_params, pointwise_feature_extractor


class RawValueExtractor(pointwise_feature_extractor.PointwiseFeatureExtractor):
    band_settings: band_settings_params.BandSettings
    window_size: int

    def __init__(self, window_size: int, band_settings, window_bias: float = 1e-8):
        self.window_size = window_size
        self.band_settings = band_settings
        self.window_bias = window_bias

    def _window_pad_left(self, array: np.ndarray, num_pad, axis=0):
        assert axis == 0, "Other modes not implemented yet."
        array = array[-self.window_size:]
        return np.concatenate(
            (np.zeros(shape=(num_pad, ) + array.shape[1:], dtype=array.dtype), array),
            axis=axis
        )

    def _window_pad_right(self, array: np.ndarray, num_pad, axis=0):
        assert axis == 0, "Other modes not implemented yet."
        array = array[:self.window_size]
        return np.concatenate(
            (array, np.zeros(shape=(num_pad, ) + array.shape[1:], dtype=array.dtype)),
            axis=axis
        )

    def _get_num_pad(self, array: np.ndarray, axis=0):
        return max(0, self.window_size - array.shape[axis])

    def _extract_per_band(self, band: lightcurve.BandData, time: float):
        closest = band.closest_point(time)
        before = band.before_time(closest.time, bias=self.window_bias)
        before_pad = self._get_num_pad(before.time)
        after = band.after_time(closest.time, bias=self.window_bias)
        after_pad = self._get_num_pad(after.time)
        return {
            "before_time": self._window_pad_left(before.time, before_pad),
            "before_flux": self._window_pad_left(before.flux, before_pad),
            "before_padding": before_pad,
            "after_time": self._window_pad_right(after.time, after_pad),
            "after_flux": self._window_pad_right(after.flux, after_pad),
            "after_padding": after_pad,
            "requested_time": time,
            "closest_time_in_band": closest.time,
            "closest_flux_in_band": closest.flux,
            "closest_time_diff": abs(closest.time - time),
        }

    def extract(self, lc: lightcurve._LC, time: float):
        return self.band_settings.generate_per_band_features(
            functools.partial(self._extract_per_band, time=time), lc
        )
