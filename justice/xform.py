import typing
from collections import namedtuple

import numpy as np
from tensorflow.contrib.framework import nest

if typing.TYPE_CHECKING:
    from justice import lightcurve


class Xform(namedtuple('Xform', ('tx', 'ty', 'dx', 'dy', 'rs'))):
    """
    translation, dilation in x, y, plus band coupling

    """
    __slots__ = ()

    def __new__(cls, tx, ty, dx, dy, rs):
        # if kwargs or not args:
        # Using kwargs is discouraged as of right now
        #     assert not args
        # kwargs.setdefault("tx", 0.0)
        # kwargs.setdefault("ty", {b: 0.0})
        # kwargs.setdefault("dx", 1.0)
        # kwargs.setdefault("dy", {b: 1.0})
        # kwargs.setdefault("rs", 0.0)
        # noinspection PyTypeChecker
        #     return super(cls, Xform).__new__(cls, **kwargs)
        # else:
        #     # noinspection PyTypeChecker
        return super(cls, Xform).__new__(cls, tx, ty, dx, dy, rs)

    def as_array(self):
        return np.array(nest.flatten(self), dtype=np.float64)

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


class PerBandTransforms(dict):
    # noinspection PyProtectedMember
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
