from justice.lightcurve import LC
import numpy as np
from collections import namedtuple
from tensorflow.contrib.framework import nest


class Xform(namedtuple('Xform', ('tx', 'ty', 'dx', 'dy', 'bc'))):
    """
    translation, dilation in x, y, plus band coupling

    """
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if kwargs or not args:
            assert not args
            kwargs.setdefault("tx", 0.0)
            kwargs.setdefault("ty", 0.0)
            kwargs.setdefault("dx", 1.0)
            kwargs.setdefault("dy", 1.0)
            kwargs.setdefault("bc", [1.0,])
            return super(cls, Xform).__new__(cls, **kwargs)
        else:
            return super(cls, Xform).__new__(cls, *args)

    def as_array(self):
        return np.array(nest.flatten(self), dtype=np.float64)

    def transform(self, lc):
        # check that error really does behave this way
        # unsure if we should separate xform into bands as well, or have a global xform?
        new_x = []
        new_y = []
        new_yerr = []
        for x, y, yerr, b in zip(lc.x.T, lc.y.T, lc.yerr.T, self.bc):
            new_x.append(self.dx * (x + self.tx))
            new_y.append(self.dy * b * (y + self.ty))
            new_yerr.append(np.sqrt(self.dy) * yerr)
        return LC(np.array(new_x).T, np.array(new_y).T, np.array(new_yerr).T)


def make_xform(list):
    xform = Xform(list[0], list[1], list[2], list[3], list[4])
    return xform


def transform(lc, xform):
    return xform.transform(lc)
