# -*- coding: utf-8 -*-
"""Runs various Lomb-Scarble methods."""
import abc
import collections
import typing

import numpy as np

from justice import lightcurve
import astropy.stats
from gatspy import periodic


class MultiBandFrequency(collections.OrderedDict):
    __slots__ = ("frequency", )

    def __init__(self, *, frequency, band_to_power):
        super().__init__(band_to_power)
        for value in self.values():
            assert isinstance(value, np.ndarray)
            assert len(value.shape) == 1, "Expected 1D shape per band."
        self.frequency = frequency

    def plot(self, band_name):
        import matplotlib.pyplot as plt
        plt.plot(self.frequency, self[band_name])

    def freq_max(self, band_name):
        return self.frequency[np.nanargmax(self[band_name])]

    def period_max(self, band_name):
        return 1.0 / self.freq_max(band_name)


class LsTransformBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, frequency: typing.Union[str, np.ndarray] = 'default') -> None:
        if frequency == 'default':
            self.frequency = 1.0 / np.linspace(100, 2, 100)
        else:
            assert isinstance(frequency, np.ndarray)
            self.frequency = frequency

    @abc.abstractmethod
    def transform(self, lc: lightcurve._LC) -> MultiBandFrequency:
        raise NotImplementedError()


def _compute_ls(band, frequency):
    return astropy.stats.LombScargle(
        band.time, band.flux, band.flux_err).power(frequency)


class IndependentLs(LsTransformBase):
    def transform(self, lc: lightcurve._LC) -> MultiBandFrequency:
        band_to_power = {
            band_name: _compute_ls(band, self.frequency)
            for band_name, band in lc.bands.items()
        }
        return MultiBandFrequency(frequency=self.frequency, band_to_power=band_to_power)


class MultiBandLs(LsTransformBase):
    """Runs gatspy multi-band LS.

    However, this currently seems to give the same frequency distribution for all bands.
    """

    def transform(self, lc: lightcurve._LC) -> MultiBandFrequency:
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

        model = periodic.LombScargleMultiband()
        model.fit(times, fluxes, flux_errs, bands)
        power = model.periodogram(1.0 / self.frequency)
        return MultiBandFrequency(
            frequency=self.frequency,
            band_to_power={name: power
                           for name in lc.bands.keys()}
        )
