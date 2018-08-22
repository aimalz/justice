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
    new_x = (aff.dx * lc.x) + aff.tx
    new_y = (aff.dy * lc.y) + aff.ty
    new_yerr = np.sqrt(aff.dy) * lc.yerr
    return LC(new_x, new_y, new_yerr)
