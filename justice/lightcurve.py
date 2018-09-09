import numpy as np


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


class _LC:
    def __init__(self, **bands):
        assert set(bands.keys()) == self._expected_bands
        self.bands = bands

    @property
    def _expected_bands(self):
        return set()

    def __repr__(self):
        kwargs = ', '.join(['{}={}'.format(band, data) for band, data in self.bands.items()])
        return '{dataset}({kwargs})'.format(dataset=self.__class__.__name__, kwargs=kwargs)

    def __add__(self, other):
        assert self._expected_bands == other._expected_bands
        bands = dict((band, self.bands[band] + other.bands[band]) for band in self.bands)
        return self.__class__(**bands)


class SNDatasetLC(_LC):
    @property
    def _expected_bands(self):
        return {'g', 'r', 'i', 'z'}
