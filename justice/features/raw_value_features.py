# -*- coding: utf-8 -*-
"""Extracts raw values around a point."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from justice import lightcurve
from justice.features import band_settings_params


class RawValueExtractor(object):
    band_settings: band_settings_params.BandSettings
    window_size: int

    def __init__(self, window_size: int, band_settings):
        self.window_size = window_size
        self.band_settings = band_settings

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
        before = band.before_time(time)
        before_pad = self._get_num_pad(before.time)
        after = band.after_time(time)
        after_pad = self._get_num_pad(after.time)
        closest = band.closest_point(time)
        return {
            "before_time": self._window_pad_left(before.time, before_pad),
            "before_flux": self._window_pad_left(before.flux, before_pad),
            "before_padding": before_pad,
            "after_time": self._window_pad_right(after.time, after_pad),
            "after_flux": self._window_pad_right(after.flux, after_pad),
            "after_padding": after_pad,
            "closest_flux": closest.flux,
            "closest_time_diff": abs(closest.time - time),
        }

    def extract(self, lc: lightcurve._LC, time: float):
        return self.band_settings.generate_per_band_features(
            functools.partial(self._extract_per_band, time=time), lc
        )
