import numpy as np
from collections import namedtuple
from tensorflow.contrib.framework import nest


class Xform(namedtuple('Xform', ('tx', 'ty', 'dx', 'dy', 'bc'))):
    """
    translation, dilation in x, y, plus band coupling

    """
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if kwargs or not args: #Using kwargs is discouraged as of right now
            assert not args
            kwargs.setdefault("tx", 0.0)
            kwargs.setdefault("ty", 0.0)
            kwargs.setdefault("dx", 1.0)
            kwargs.setdefault("dy", 1.0)
            kwargs.setdefault("bc", {'b': 0.0}) 
            return super(cls, Xform).__new__(cls, **kwargs)
        else:
            return super(cls, Xform).__new__(cls, *args)

    def as_array(self):
        return np.array(nest.flatten(self), dtype=np.float64)

    def transform_band(self, bd, bc):
        # check that error really does behave this way
        new_x = self.dx * (bd.time + self.tx)
        new_y = self.dy * (bd.flux + self.ty)
        new_yerr = np.sqrt(self.dy) * bd.flux_err
        return bd.__class__(new_x, new_y, new_yerr)

    def transform(self, lc):
        bands = {b: self.transform_band(lc.bands[b], self.bc[b]) for b in lc.bands}
        return lc.__class__(**bands)


def make_xform(list):
    xform = Xform(list[0], list[1], list[2], list[3], list[4])
    return xform


def transform(lc, xform):
    return xform.transform(lc)
