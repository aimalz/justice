import numpy as np
from justice import lightcurve


class BandNameMapper:
    """
    Our observation space is indexed by discrete BandNames
    but our model space is indexed by continuous pseudo-wavelength (pwav)
    An instance of `BandNameMapper` represents the injection of BandNames into points
    on the pwav space. Here, the mapping is represented as a sorted list of wavelengths
    so that BandName[i] -> target_wavelengths[i]
    The numbers you supply here must already have redshift incorporated
    """

    def __init__(self, target_wavelengths):
        for i in range(len(target_wavelengths) - 1):
            assert target_wavelengths[i] < target_wavelengths[i + 1]
        self.wavelengths = target_wavelengths

    @property
    def nbands(self):
        return len(self.wavelengths)


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
        # TODO: check that error really does behave this way
        new_yerr = np.sqrt(self._dilate_flux) * bd.flux_err
        return bd.__class__(new_x, new_y, new_yerr)


class LCXform:
    # TODO
    def apply(self, lc) -> lightcurve._LC:
        pass


class LC2DXform:
    # TODO
    def apply(self, lc2d) -> lightcurve.LC2D:
        pass
