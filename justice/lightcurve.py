import numpy as np
from math import ceil
import scipy.stats as sps
from collections import OrderedDict
from justice.xform import Xform

class BandData(object):
    def __init__(self, time, flux, flux_err):
        assert time.shape == flux.shape == flux_err.shape
        self.time = time
        self.flux = flux
        self.flux_err = flux_err

    def __repr__(self):
        return 'BandData(<size {size}>)'.format(size=len(self.time))

    def __add__(self, other):
        # this function is a likely culprit for future slowness, given how many times we'll be calling it.
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

    def connect_the_dots(self):
        #ignores errorbars
        time_difs = time[1:] - time[:-1]
        flux_difs = flux[1:] - flux[:-1]
        return np.sum(np.sqrt(time_diffs**2 + flux_diffs**2))


class _LC:
    def __init__(self, **bands):
        d = OrderedDict()
        for b in self._expected_bands:
            d[b] = bands[b]
        for k in bands:
            assert k in self._expected_bands
        self.bands = bands

    @property
    def nbands(self):
        return len(self.bands)

    def _expected_bands(self):
        return ''

    def __repr__(self):
        kwargs = ', '.join(['{}={}'.format(band, data) for band, data in self.bands.items()])
        return '{dataset}({kwargs})'.format(dataset=self.__class__.__name__, kwargs=kwargs)

    def __add__(self, other):
        assert self._expected_bands == other._expected_bands
        bands = dict((band, self.bands[band] + other.bands[band]) for band in self.bands)
        return self.__class__(**bands)

    def to_arrays(self):
        """
        Formats this LC to a tuple of arrays, suitable for GPy
        Pads with repeats with the flux_errs much bigger

        :param band_order: Order of expected bands.
        :return: np.array, np.array, np.array
        """
        max_size = max(bd.time.shape[0] for bd in self.bands.values())
        out_time = np.zeros((max_size, len(self.bands)))
        out_flux = np.zeros((max_size, len(self.bands)))
        out_flux_err = np.zeros((max_size, len(self.bands)))
        for i, b in enumerate(self._expected_bands):
            band = self.bands[b]
            band_len = band.time.shape[0]
            n_copies = ceil(max_size / band_len)
            out_time[:, i] = np.concatenate((band.time,) * n_copies)[:max_size]
            out_flux[:, i] = np.concatenate((band.flux,) * n_copies)[:max_size]
            if n_copies == 1:
                out_flux_err[:, i] = band.flux_err
            else:
                chunks = (band.flux_err,) + (band.flux_err * 100,) * (n_copies - 1)
                out_flux_err[:, i] = np.concatenate(chunks)[:max_size]
        ordinals = np.argsort(out_time)

        return out_time[ordinals], out_flux[ordinals], out_flux_err[ordinals]

    def get_xform(self, vals=None):
        if vals is None:
            vals[0]=0.
            vals[1]=0.
            vals[2]=1.
            vals[3]=1.
            for b in self._expected_bands:
                vals.append(1.)
        tx = vals[0]
        ty = vals[1]
        dx = vals[2]
        dy = vals[3]
        bc = OrderedDict()
        for b, val in zip(self._expected_bands, vals[4:]):
            bc[b]=val
        return Xform(tx, ty, dx, dy, bc)

    def connect_the_dots(self):
        # ignores errorbars
        arclen = 0.
        for b in self._expected_bands:
            arclen += self.bands[b].connect_the_dots()
        return arclen


class SNDatasetLC(_LC):
    @property
    def _expected_bands(self):
        return 'griz'


class OGLEDatasetLC(_LC):
    @property
    def _expected_bands(self):
        return 'IV'

