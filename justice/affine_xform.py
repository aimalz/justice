from justice.lightcurve import LC
import numpy as np
from collections import namedtuple

# translation, dilation in x, y
# currently this doesn't handle multiple filters/colors
Aff = namedtuple('Aff', ('tx', 'ty', 'dx', 'dy'))


def make_aff(list):
    aff = Aff(list[0], list[1], list[2], list[3])
    return aff


def transform(lc, aff):
    # check that error really does behave this way
    #unsure if we should separate aff into bands as well, or have a global aff?
    new_x = []
    new_y = []
    new_yerr = []
    for x, y, yerr in zip(lc.x, lc.y, lc.yerr):
        new_x.append((aff.dx * x) + aff.tx)
        new_y.append((aff.dy * y) + aff.ty)
        new_yerr.append(np.sqrt(aff.dy) * yerr)
    return LC(new_x, new_y, new_yerr)
