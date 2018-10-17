import numpy as np
from justice import lightcurve
import abc


class BandNameMapper:
    """
    Our observation space is indexed by discrete BandNames
    but our model space is indexed by continuous pseudo-wavelength (pwav)
    An instance of `BandNameMapper` represents the injection of BandNames into points
    on the pwav space. Here, the mapping is represented as a dictionary of pwavs
    so that BandName b -> pwavs[b]
    The numbers you supply here should already have redshift incorporated
    """

    def __init__(self, **pwavs):
        # some sloppiness here: these are not ordered.
        self.pwavs = pwavs

    def make_lc2d(self, lc: lightcurve._LC) -> lightcurve.LC2D:
        assert frozenset(lc.expected_bands) == frozenset(self.pwavs.keys())
        for i, e in enumerate(lc.expected_bands)[:-1]:
            next_e = lc.expected_bands[i + 1]
            assert self.pwavs[e] < self.pwavs[next_e], 'Bands must be mapped to increasing pwavs'
        pwav = np.concat([
            np.ones_like(lc.bands[b].time) * self.pwavs[b]
            for b in lc.expected_bands
        ])

        time = np.concat([band.time for name, band in lc.bands.items()])
        flux = np.concat([band.flux for name, band in lc.bands.items()])
        flux_err = np.concat([band.flux_err for name, band in lc.bands.items()])
        detected = np.concat([band.detected for name, band in lc.bands.items()])
        return lightcurve.LC2D(pwav, time, flux, flux_err, detected)


class BandDataXform:
    def __init__(self, time_fn, flux_fn, flux_err_fn):
        self._time_fn = time_fn
        self._flux_fn = flux_fn
        self._flux_err_fn = flux_err_fn

    def apply(self, bd) -> lightcurve.BandData:
        return lightcurve.BandData(
            self._time_fn(bd.time),
            self._flux_fn(bd.flux),
            self._flux_err_fn(bd.flux_err),
            bd.detected,
        )

    @classmethod
    def identity(cls):
        return cls(lambda x: x, lambda x: x, lambda x: x)


class LinearBandDataXform(BandDataXform):
    def __init__(self, translate_time, translate_flux, dilate_time, dilate_flux):
        self._translate_time = translate_time
        self._translate_flux = translate_flux
        self._dilate_time = dilate_time
        self._dilate_flux = dilate_flux

    def apply(self, bd) -> lightcurve.BandData:
        new_x = self._dilate_time * (bd.time + self._translate_time)
        new_y = self._dilate_flux * (bd.flux + self._translate_flux)
        # TODO: does error really behave this way?
        new_yerr = np.sqrt(self._dilate_flux) * bd.flux_err
        return lightcurve.BandData(new_x, new_y, new_yerr)

    @classmethod
    def identity(cls):
        return cls(0, 0, 1, 1)


class LCXform:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, lc) -> lightcurve._LC:
        pass


class IndependentLCXform(LCXform):
    def __init__(self, **band_xforms):
        self._band_xforms: collections.OrderedDict[str, BandDataXform] = collections.OrderedDict()
        for b in sorted(band_xforms.keys()):
            self._band_xforms[b] = band_xforms[b]

    def apply(self, lc) -> lightcurve._LC:
        # want to allow easy creation of this class with just one BandDataXform
        assert self._band_xforms.keys().issubset(lc.bands.keys())
        new_bands = {}
        for name in lc.bands:
            new_bands[name] = self._band_xforms(lc.bands[name])
        return lc.__class__(**new_bands)

    @classmethod
    def identity(cls):
        return cls()


class LC2DXform:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, lc2d) -> lightcurve.LC2D:
        pass
