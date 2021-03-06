import typing

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
            assert self.pwavs[this_e] < self.pwavs[
                next_e], 'Bands must be mapped to increasing pwavs'
        pwav = np.concatenate([
            np.ones_like(lc.bands[b].time) * self.pwavs[b] for b in lc.expected_bands
        ])

        time = np.concatenate([band.time for name, band in lc.bands.items()])
        flux = np.concatenate([band.flux for name, band in lc.bands.items()])
        flux_err = np.concatenate([band.flux_err for name, band in lc.bands.items()])
        detected = np.concatenate([band.detected for name, band in lc.bands.items()])
        return lightcurve.LC2D(pwav, time, flux, flux_err, detected)


class BandDataXform:
    def __init__(self, time_fn, flux_fn, flux_err_fn):
        self._time_fn = time_fn
        self._flux_fn = flux_fn
        self._flux_err_fn = flux_err_fn

    def __str__(self):
        return "BandDataXform[]"

    def apply(self, bd) -> lightcurve.BandData:
        return lightcurve.BandData(
            self._time_fn(bd.time),
            self._flux_fn(bd.flux),
            self._flux_err_fn(bd.flux_err),
            bd.detected,
        )

    def apply_time(self, time: typing.Union[float, np.ndarray]):
        return self._time_fn(time)


class LinearBandDataXform(BandDataXform):
    def __init__(
        self,
        translate_time,
        translate_flux,
        dilate_time,
        dilate_flux,
        check_positive=True
    ):
        self._translate_time = translate_time
        self._translate_flux = translate_flux
        self._dilate_time = dilate_time
        self._dilate_flux = dilate_flux
        if check_positive:
            if dilate_time <= 0:
                raise ValueError("Expected dilate_time to be positive.")
            if dilate_flux <= 0:
                raise ValueError("Expected dilate_flux to be positive.")

    def __str__(self):
        return (
            "LinearBandDataXform["
            f"t → {self._dilate_time:.2f} (t + {self._translate_time:.2f}), "
            f"flux → {self._dilate_flux:.2f} (flux + {self._translate_flux:.2f})]"
        )

    def apply(self, bd) -> lightcurve.BandData:
        new_x = self._dilate_time * (bd.time + self._translate_time)
        new_y = self._dilate_flux * (bd.flux + self._translate_flux)
        # TODO: does error really behave this way?
        new_yerr = np.sqrt(self._dilate_flux) * bd.flux_err
        return lightcurve.BandData(new_x, new_y, new_yerr, detected=bd.detected)

    def apply_time(self, time: typing.Union[float, np.ndarray]):
        return self._dilate_time * (time + self._translate_time)


class LCXform:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, lc) -> lightcurve._LC:
        pass


class IndependentLCXform(LCXform):
    """Every band gets a different transform"""

    def __init__(self, **band_xforms):
        self._band_xforms: collections.OrderedDict[str, BandDataXform
                                                   ] = collections.OrderedDict()
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

    def __init__(self, band_xform: BandDataXform) -> None:
        self._band_xform = band_xform

    def apply(self, lc) -> lightcurve._LC:
        new_bands = {}
        for name in lc.bands:
            new_bands[name] = self._band_xform.apply(lc.bands[name])
        result = lc.__class__(**new_bands)
        result.meta.update(lc.meta)
        result.meta["last_transform"] = str(self._band_xform)
        return result

    def apply_time(self, time: typing.Union[float, np.ndarray]):
        return self._band_xform.apply_time(time)


class LC2DXform:
    """Just a stub for now. We don't know what these functions might look like yet
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, lc2d) -> lightcurve.LC2D:
        pass
