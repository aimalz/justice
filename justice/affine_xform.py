from justice.lightcurve import LC
import numpy as np
from collections import namedtuple


class Aff(namedtuple('Aff', ('tx', 'ty', 'dx', 'dy'))):
    """
    translation, dilation in x, y
    currently this doesn't handle multiple filters/colors

    """
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if kwargs or not args:
            assert not args
            kwargs.setdefault("tx", 0.0)
            kwargs.setdefault("ty", 0.0)
            kwargs.setdefault("dx", 1.0)
            kwargs.setdefault("dy", 1.0)
            return super(cls, Aff).__new__(cls, **kwargs)
        else:
            return super(cls, Aff).__new__(cls, *args)

    def as_array(self):
        return np.array(self, dtype=np.float64)

    def transform(self, lc):
        # check that error really does behave this way
        # unsure if we should separate aff into bands as well, or have a global aff?
        new_x = []
        new_y = []
        new_yerr = []
        for x, y, yerr in zip(lc.x.T, lc.y.T, lc.yerr.T):
            new_x.append(self.dx * (x + self.tx))
            new_y.append(self.dy * (y + self.ty))
            new_yerr.append(np.sqrt(self.dy) * yerr)
        return LC(np.array(new_x).T, np.array(new_y).T, np.array(new_yerr).T)


def make_aff(list):
    aff = Aff(list[0], list[1], list[2], list[3])
    return aff


def transform(lc, aff):
    return aff.transform(lc)
