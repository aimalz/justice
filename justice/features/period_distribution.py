# -*- coding: utf-8 -*-
"""Runs various Lomb-Scargle methods."""
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
        "best_periods",
    )

    def __init__(self, *, period, band_to_power,
                 best_periods=None, scores=None):
        super().__init__(band_to_power)
        self.period = period
        self.best_periods = best_periods
        self.scores = scores

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


class MultiBandLs(LsTransformBase):
    """Runs gatspy multi-band LS.
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

        model = periodic.LombScargleMultiband(fit_period=False)
        model.optimizer.period_range = (.1, np.max(times) - np.min(times))
        model.optimizer.quiet = True
        model.fit(times, fluxes, flux_errs, bands)
        period, power = model.periodogram_auto()
        best_periods, scores = model.find_best_periods(5, return_scores=True)
        return MultiBandPeriod(
            period=period,
            band_to_power={name: power
                           for name in lc.bands.keys()},
            best_periods=best_periods, scores=scores
        )
