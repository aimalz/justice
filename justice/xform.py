import numpy as np
from justice import lightcurve
import abc
import collections


class BandNameMapper:
    """
    ("pwav" means pseudo-wavelength. "pseudo" because the filters are aggregating
    across jagged absorption spectra, making it infeasible to model true wavelengths)

    An observation is six discrete bands.
    But a proper model is more like a 2D surface in a space of (pwav, time, flux)
    We need some object to represent the conversion from independent-seeming bands
    to stripes on that 2D surface
    A `BandNameMapper` has a dictionary of pwavs so that BandName b -> pwavs[b],
    putting each observed band at a *constant* pwav
    The numbers you supply here should already have redshift incorporated
    """

    def __init__(self, **pwavs):
        # some sloppiness here: these are not ordered.
        self.pwavs = pwavs

    def make_lc2d(self, lc: lightcurve._LC) -> lightcurve.LC2D:
        assert frozenset(lc.expected_bands) == frozenset(self.pwavs.keys())
        for i in range(len(lc.expected_bands) - 1):
            this_e = lc.expected_bands[i]
            next_e = lc.expected_bands[i + 1]
            assert self.pwavs[this_e] < self.pwavs[next_e], 'Bands must be mapped to increasing pwavs'
        pwav = np.concat([
            np.ones_like(lc.bands[b].time) * self.pwavs[b]
            for b in lc.expected_bands
        ])

        time = np.concat([band.time for name, band in lc.bands.items()])
        flux = np.concat([band.flux for name, band in lc.bands.items()])
        flux_err = np.concat([band.flux_err for name, band in lc.bands.items()])
        detected = np.concat([band.detected for name, band in lc.bands.items()])
        return lightcurve.LC2D(pwav, time, flux, flux_err, detected)

    def transform_band(self, bd, ty, dy):
        # currently ignoring rs
        # check that error really does behave this way
        new_x = self.dx * (bd.time + self.tx)
        new_y = dy * (bd.flux + ty)
        new_yerr = np.sqrt(dy) * bd.flux_err
        return bd.__class__(new_x, new_y, new_yerr, bd.detected)

    def transform(self, lc):
        bands = {
            b: self.transform_band(lc.bands[b], self.ty[b], self.dy[b])
            for b in lc.bands
        }
        return lc.__class__(**bands)

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


class LCXform:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, lc) -> lightcurve._LC:
        pass


class IndependentLCXform(LCXform):
    """Every band gets a different transform"""

    def __init__(self, **band_xforms):
        self._band_xforms: collections.OrderedDict[str, BandDataXform] = collections.OrderedDict()
        for b in sorted(band_xforms.keys()):
            self._band_xforms[b] = band_xforms[b]

    def apply(self, lc) -> lightcurve._LC:
        # want to allow easy creation of this class with just one BandDataXform
        assert set(self._band_xforms.keys()).issubset(lc.bands.keys())
        new_bands = {}
        for name in lc.bands:
            new_bands[name] = self._band_xforms[name].apply(lc.bands[name])
        return lc.__class__(**new_bands)


class SameLCXform(LCXform):
    """All bands get the same transform"""

    def __init__(self, band_xform):
        self._band_xform = band_xform

    def apply(self, lc) -> lightcurve._LC:
        new_bands = {}
        for name in lc.bands:
            new_bands[name] = self._band_xform.apply(lc.bands[name])
        return lc.__class__(**new_bands)


class LC2DXform:
    """Just a stub for now. We don't know what these functions might look like yet
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, lc2d) -> lightcurve.LC2D:
        pass
