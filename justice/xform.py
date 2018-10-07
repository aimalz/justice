import typing
from collections import namedtuple

import numpy as np
from tensorflow.contrib.framework import nest

if typing.TYPE_CHECKING:
    from justice import lightcurve


class Xform(namedtuple('Xform', ('tx', 'ty', 'dx', 'dy', 'rs'))):
    """
    translation, dilation in x, y, plus redshift

    """
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if kwargs or not args:  # Using kwargs is discouraged as of right now
            assert not args
            kwargs.setdefault("tx", 0.0)
            kwargs.setdefault("ty", [0.0] * len(lc.bands))
            kwargs.setdefault("dx", 1.0)
            kwargs.setdefault("dy", 1.0)
            kwargs.setdefault("rs", 0.0)
            return super(cls, Xform).__new__(cls, **kwargs)
        else:
            return super(cls, Xform).__new__(cls, *args)

    def as_array(self):
        return np.array(nest.flatten(self), dtype=np.float64)

    def transform_band(self, bd, ty_ind):
        # check that error really does behave this way
        new_x = self.dx * (bd.time + self.tx)
        new_y = self.dy * (bd.flux + self.ty[ty_ind])
        new_yerr = np.sqrt(self.dy) * bd.flux_err
        return bd.__class__(new_x, new_y, new_yerr)

    def transform(self, lc):
        bands = {b: self.transform_band(lc.bands[b], self.bc[b]) for b in lc.bands}
        return lc.__class__(**bands)


class PerBandTransforms(dict):
    def transform(self, lc: 'lightcurve._LC'):
        if frozenset(self.keys()) != frozenset(lc.expected_bands):
            raise ValueError(
                "Expected bands {} but got {}".format(self.keys(), lc.expected_bands)
            )

        return lc.__class__(
            **{
                b: self[b].transform_band(band_data, 1.0)
                for b, band_data in lc.bands.items()
            }
        )


def make_xform(lst):
    return Xform(lst[0], lst[1], lst[2], lst[3], lst[4])


def transform(lc, xform):
    return xform.transform(lc)
