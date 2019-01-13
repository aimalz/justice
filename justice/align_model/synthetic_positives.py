# -*- coding: utf-8 -*-
"""Generates synthetic positive data."""
import math
import random
import typing

import numpy as np
import tensorflow as tf
from scipy.stats import truncnorm

from justice import lightcurve
from justice import xform
from justice.features import band_settings_params, example_pair
from justice.features import raw_value_features
from justice.features.example_pair import FullPositivesPair


class BasicPositivesGenerator:
    """Generates positive example pairs by transforming a light curve.

    For non-test data, we could consider estimating the range of shifts/dilations by
    target, for example if we see that variable objects have a period within [x, 3x]
    (for some x) then we could consider dilate_time_stdev_factor=3.
    """

    def __init__(
        self,
        *,
        translate_time_stdev=1.0,
        translate_flux_stdev=0.01,
        dilate_time_stdev_factor=1.1,
        dilate_flux_stdev_factor=1.5,
        rng: random.Random = None,
        max_time_dilation = 6,
        max_flux_dilation = 16
    ):
        self.translate_time_stdev = translate_time_stdev
        self.translate_flux_stdev = translate_flux_stdev

        if dilate_time_stdev_factor < 1.0:
            raise ValueError("dilate_time_stdev_factor must be >= 1.")
        if dilate_flux_stdev_factor < 1.0:
            raise ValueError("dilate_flux_stdev_factor must be >= 1.")

        self.log_dilate_time_stdev = math.log(dilate_time_stdev_factor)
        self.log_dilate_flux_stdev = math.log(dilate_flux_stdev_factor)
        self.rng = random.Random() if rng is None else rng
        self.max_time_dilation = max_time_dilation
        self.max_flux_dilation = max_flux_dilation

    def make_xform(self):
        translate_time = self.rng.normalvariate(0, self.translate_time_stdev)
        translate_flux = self.rng.normalvariate(0, self.translate_flux_stdev)
        time_trunc = math.log(self.max_time_dilation) - self.log_dilate_time_stdev
        flux_trunc = math.log(self.max_flux_dilation) - self.log_dilate_flux_stdev

        dilations = truncnorm.rvs(
            [-time_trunc, -flux_trunc],
            [time_trunc, flux_trunc],
            0,
            [self.log_dilate_time_stdev, self.log_dilate_flux_stdev],
            random_state=self.rng.getrandbits(32))

        return xform.LinearBandDataXform(
            translate_time,
            translate_flux,
            dilate_time=math.exp(dilations[0]),
            dilate_flux=math.exp(dilations[1]),
            check_positive=True,
        )

    def make_positive_pair(self, lc: lightcurve._LC) -> FullPositivesPair:
        time = float(self.rng.choice(lc.all_times_unique()))
        xf = self.make_xform()
        lc_xf = xform.SameLCXform(xf)
        transformed = lc_xf.apply(lc)
        time_transformed: float = xf.apply_time(time)
        return FullPositivesPair(
            lca=lc, time_a=time, lcb=transformed, time_b=time_transformed
        )


class RandomSubsampler(xform.BandDataXform):
    """Band data transformer that samples points within a band.
    """

    def __init__(
        self,
        *,
        min_rate: float,
        max_rate: float,
        preserve_time: float,
        preserve_time_radius: float = 4.0,
        rng: np.random.RandomState = None
    ):
        self.rng = rng if rng is not None else np.random.RandomState()
        self.preserve_time = preserve_time
        self.preserve_time_radius = preserve_time_radius
        self.min_rate = min_rate
        self.max_rate = max_rate

    def apply(self, band):
        indices = np.arange(0, len(band.time))
        preserve_mask = np.abs(band.time -
                               self.preserve_time) < self.preserve_time_radius
        preserved = indices[preserve_mask]
        indices = indices[~preserve_mask]
        num_points = len(indices)
        # usually 2 but allow lower if very small
        hard_min_size = min(num_points, 2)
        min_size = max(hard_min_size, int(self.min_rate * num_points))
        max_size = int(math.ceil(self.max_rate * num_points))
        if min_size >= max_size:
            sample_size = min_size
        else:
            sample_size = self.rng.randint(min_size, max_size)
        sample_indices = np.sort(
            np.concatenate([
                preserved,
                self.rng.choice(indices, size=sample_size, replace=False)
            ])
        )
        return lightcurve.BandData(
            time=band.time[sample_indices],
            flux=band.flux[sample_indices],
            flux_err=band.flux_err[sample_indices],
            detected=band.detected[sample_indices]
        )

    def apply_time(self, time: typing.Union[float, np.ndarray]):
        return time


class PositivePairSubsampler:
    def __init__(
        self,
        *,
        min_rate: float = 0.5,
        max_rate: float = 1.0,
        preserve_time_radius: float = 4.0,
        rng: np.random.RandomState = None
    ):
        self.rng = rng if rng is not None else np.random.RandomState()
        self.preserve_time_radius = preserve_time_radius
        self.min_rate = min_rate
        self.max_rate = max_rate

    def lc_xform(self, time: float) -> xform.SameLCXform:
        band_xform = RandomSubsampler(
            min_rate=self.min_rate,
            max_rate=self.max_rate,
            preserve_time=time,
            preserve_time_radius=self.preserve_time_radius,
            rng=self.rng
        )
        return xform.SameLCXform(band_xform)

    def apply(self, fpp: FullPositivesPair) -> FullPositivesPair:
        lca_xf = self.lc_xform(fpp.time_a)
        lcb_xf = self.lc_xform(fpp.time_b)
        return FullPositivesPair(
            lca=lca_xf.apply(fpp.lca),
            lcb=lcb_xf.apply(fpp.lcb),
            time_a=fpp.time_a,
            time_b=fpp.time_b
        )


class RawValuesFullPositives(example_pair.PairFexFromPointwiseFex):
    def __init__(self, bands, window_size):
        self.band_settings = band_settings_params.BandSettings(bands=bands)
        fex = raw_value_features.RawValueExtractor(
            window_size=window_size, band_settings=self.band_settings
        )
        super(RawValuesFullPositives, self).__init__(fex=fex, label=True)

    @classmethod
    def from_params(cls, params: dict):
        return cls(params["lc_bands"], params["window_size"])
