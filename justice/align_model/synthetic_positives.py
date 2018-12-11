# -*- coding: utf-8 -*-
"""Generates synthetic positive data."""
import math
import random
import typing

import numpy as np
import tensorflow as tf
from scipy.stats import truncnorm

from justice import xform, lightcurve
from justice.align_model import lr_prefixing
from justice.features import band_settings_params, raw_value_features, tf_dataset_builder


class FullPositivesPair:
    """Simple struct for a pair of positive examples.

    The meaning is that `time_a` of light curve `lca` should be aligned with `time_b` of
    light curve `lcb`. For the current alignment model, we do not have other gold/intended
    transformation parameters, we merely hope that points having the best alignment will
    lead to reasonable inference of dilation.
    """

    __slots__ = ("lca", "lcb", "time_a", "time_b")

    def __init__(
        self, lca: lightcurve._LC, lcb: lightcurve._LC, time_a: float, time_b: float
    ):
        self.lca = lca
        self.lcb = lcb
        self.time_a = time_a
        self.time_b = time_b


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
        rng: random.Random = None
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

    def make_xform(self):
        translate_time = self.rng.normalvariate(0, self.translate_time_stdev)
        translate_flux = self.rng.normalvariate(0, self.translate_flux_stdev)

        dilations = truncnorm.rvs(
            math.log(0.04),
            math.log(25),
            0,
            [self.log_dilate_flux_stdev, self.log_dilate_time_stdev],
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


class RawValuesFullPositives:
    def __init__(self, bands, window_size):
        self.band_settings = band_settings_params.BandSettings(bands=bands)
        self.fex = raw_value_features.RawValueExtractor(
            window_size=window_size, band_settings=self.band_settings
        )

    def apply(self, fpp: FullPositivesPair) -> typing.Dict[str, tf.Tensor]:
        first_features = self.fex.extract(fpp.lca, fpp.time_a)
        second_features = self.fex.extract(fpp.lcb, fpp.time_b)
        result = lr_prefixing.prefix_dicts(first_features, second_features)
        result["labels"] = True
        return result

    def make_dataset(
        self, fpp_gen: typing.Iterator[FullPositivesPair]
    ) -> tf.data.Dataset:
        return tf_dataset_builder.dataset_from_generator_auto_dtypes(
            self.apply(fpp) for fpp in fpp_gen
        )

    @classmethod
    def from_params(cls, params: dict):
        return cls(params["lc_bands"], params["window_size"])
