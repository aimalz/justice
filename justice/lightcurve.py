import abc
import collections
import math
import typing

import numpy as np
import scipy.stats as sps

from justice import xform


class BandData(object):
    """Light curve data for a single band.
    """

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> None:
        """Initializes BandData.

        :param time: Time values, 1-D np float array.
        :param flux: Flux values, 1-D np float array.
        :param flux_err: Flux error values, 1-D np float array.
        """
        assert time.shape == flux.shape == flux_err.shape
        self.time = time
        self.flux = flux
        self.flux_err = flux_err

    def __repr__(self) -> str:
        """Formats light curve to a string for debugging."""
        return 'BandData(time={self.time}, flux={self.flux}, flux_err={self.flux_err})'.format(
            self=self)

    def __add__(self, other: 'BandData') -> 'BandData':
        """Concatenates this light curve with another, and sorts by time.

        :param other: Other light curve band data.
        """
        # this function is a likely culprit for future slowness, given how many
        # times we'll be calling it.
        times = np.concatenate((self.time, other.time))
        fluxes = np.concatenate((self.flux, other.flux))
        flux_errs = np.concatenate((self.flux_err, other.flux_err))

        # tried kind='mergesort', but it wasn't any faster with 1e7 points
        ordinals = np.argsort(times)
        return BandData(times[ordinals], fluxes[ordinals], flux_errs[ordinals])

    @classmethod
    def from_cadence_shape_and_errfracs(cls, cadence, shape, errfracs):
        true_fluxes = shape(cadence)
        error_bars = errfracs * true_fluxes
        errors = sps.norm(0, error_bars).rvs(true_fluxes.shape)
        observed_fluxes = true_fluxes + errors
        return BandData(cadence, observed_fluxes, error_bars)

    def connect_the_dots(self) -> float:
        """Returns the arc length of the light curve.

        Sensitive to the magnitude of flux.

        :return: Arc length measurement.
        """
        # ignores errorbars
        time_diffs = self.time[1:] - self.time[:-1]
        flux_diffs = self.flux[1:] - self.flux[:-1]
        return float(np.sum(np.sqrt(time_diffs**2 + flux_diffs**2)))


class _LC:
    """Abstract base light curve class. Subclasses should provide a list of bands.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **bands: BandData) -> None:
        """Initializes a light curve.

        :param bands: Dictionary of bands.
        """
        if frozenset(bands.keys()) != frozenset(self._expected_bands):
            raise ValueError(
                "Expected bands {} but got {}".format(self._expected_bands, bands.keys())
            )

        d: collections.OrderedDict[str, BandData] = collections.OrderedDict()
        for b in self._expected_bands:
            d[b] = bands[b]
        for k in bands:
            assert k in self._expected_bands
        self.bands = d

    @property
    def nbands(self) -> int:
        """Returns the number of bands."""
        return len(self.bands)

    @property
    @abc.abstractmethod
    def _expected_bands(self) -> typing.List[str]:
        """Returns list of expected bands.

        :return: List of expected bands
        :rtype: list[str]
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        kwargs = ', '.join([
            '{}={}'.format(band, data) for band, data in self.bands.items()
        ])
        return '{dataset}({kwargs})'.format(
            dataset=self.__class__.__name__, kwargs=kwargs
        )

    def __add__(self, other: '_LC') -> '_LC':
        """Concatenates all bands of two light curves together.

        :param other: Other light curve.
        :return: New merged light curve.
        """
        assert self._expected_bands == other._expected_bands
        bands = {band: self.bands[band] + other.bands[band] for band in self.bands}
        return self.__class__(**bands)

    def to_arrays(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Formats this LC to a tuple of arrays, suitable for GPy.

        Pads with repeats with the flux_errs much bigger.

        :return: np.array, np.array, np.array
        """
        max_size = max(bd.time.shape[0] for bd in self.bands.values())
        out_time = np.zeros((max_size, len(self.bands)))
        out_flux = np.zeros((max_size, len(self.bands)))
        out_flux_err = np.zeros((max_size, len(self.bands)))
        for i, b in enumerate(self._expected_bands):
            band = self.bands[b]
            band_len = band.time.shape[0]
            n_copies = math.ceil(max_size / band_len)
            out_time[:, i] = np.concatenate((band.time, ) * n_copies)[:max_size]
            out_flux[:, i] = np.concatenate((band.flux, ) * n_copies)[:max_size]
            if n_copies == 1:
                out_flux_err[:, i] = band.flux_err
            else:
                chunks = (band.flux_err, ) + (band.flux_err * 100, ) * (n_copies - 1)
                out_flux_err[:, i] = np.concatenate(chunks)[:max_size]
        ordinals = np.argsort(out_time)

        return np.squeeze(
            out_time[ordinals], axis=2
        ), np.squeeze(
            out_flux[ordinals], axis=2
        ), np.squeeze(
            out_flux_err[ordinals], axis=2
        )

    def get_xform(self, vals: np.ndarray = None) -> xform.Xform:
        if vals is None:
            vals = [0., 0., 1., 1.]
            for _ in self._expected_bands:
                vals.append(1.)
        tx = vals[0]
        ty = vals[1]
        dx = vals[2]
        dy = vals[3]
        bc: collections.OrderedDict[str, float] = collections.OrderedDict()
        for b, val in zip(self._expected_bands, vals[4:]):
            bc[b] = val
        return xform.Xform(tx, ty, dx, dy, bc)

    def connect_the_dots(self) -> float:
        """Returns the sum of the arc length of all bands.

        :return: Arclength, summed over bands.
        """
        # ignores errorbars
        arclen = 0.
        for b in self._expected_bands:
            arclen += self.bands[b].connect_the_dots()
        return arclen


class SNDatasetLC(_LC):
    """Supernova dataset light curve."""

    @property
    def _expected_bands(self):
        return ['g', 'r', 'i', 'z']


class OGLEDatasetLC(_LC):
    """OGLE dataset light curve."""

    @property
    def _expected_bands(self):
        return ['I', 'V']
