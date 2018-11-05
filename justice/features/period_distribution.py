# -*- coding: utf-8 -*-
"""Runs various Lomb-Scarble methods."""
import abc
import collections
import typing

import numpy as np

from justice import lightcurve
import astropy.stats
from gatspy import periodic


class MultiBandPeriod(collections.OrderedDict):
    __slots__ = (
        "period",
        "best_period",
    )

    def __init__(self, *, period, band_to_power, best_period=None):
        super().__init__(band_to_power)
        # for value in self.values():
        #     assert isinstance(value, np.ndarray)
        #     assert len(value.shape) == 1, "Expected 1D shape per band."
        self.period = period
        self.best_period = best_period

    def plot(self, band_name):
        import matplotlib.pyplot as plt
        plt.plot(self.period, self[band_name])
        plt.xlabel("Period")
        plt.ylabel("Power")

    def period_max(self, band_name):
        return self.period[np.nanargmax(self[band_name])]

    def frequency_max(self, band_name):
        return 1.0 / self.period_max(band_name)


class LsTransformBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, period: typing.Union[str, np.ndarray] = 'default') -> None:
        if period == 'default':
            self.period = np.linspace(2., 500., 10000)
        else:
            assert isinstance(period, np.ndarray)
            self.period = period

    @abc.abstractmethod
    def apply(self, lc: lightcurve._LC) -> MultiBandPeriod:
        raise NotImplementedError()


def _compute_ls(band, period):
    return astropy.stats.LombScargle(band.time, band.flux, band.flux_err).power(
        period
    )  # replace with autopower soon


class IndependentLs(LsTransformBase):
    def apply(self, lc: lightcurve._LC) -> MultiBandPeriod:
        band_to_power = {
            band_name: _compute_ls(band, self.period)
            for band_name, band in lc.bands.items()
        }
        return MultiBandPeriod(period=self.period, band_to_power=band_to_power)


class MultiBandLs(LsTransformBase):
    """Runs gatspy multi-band LS.

    However, this currently seems to give the same period distribution for all bands.
    """

    def apply(self, lc: lightcurve._LC) -> MultiBandPeriod:
        def _concat_for_all_bands(key_fcn):
            return np.concatenate([
                key_fcn(name, band) for name, band in lc.bands.items()
            ])

        times = _concat_for_all_bands(lambda _, band: band.time)
        fluxes = _concat_for_all_bands(lambda _, band: band.flux)
        flux_errs = _concat_for_all_bands(lambda _, band: band.flux_err)
        bands = _concat_for_all_bands(
            lambda name, band: np.full_like(band.time, name, dtype=str)
        )

        model = periodic.LombScargleMultiband(fit_period=True)
        model.optimizer.period_range = (.1, np.max(times) - np.min(times))
        model.fit(times, fluxes, flux_errs, bands)
        power = model.periodogram(self.period)
        return MultiBandPeriod(
            period=self.period,
            band_to_power={name: power
                           for name in lc.bands.keys()},
            best_period=model.best_period
        )
