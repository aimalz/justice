"""Utility functions"""

from collections import namedtuple
import numpy as np

LC = namedtuple('LC', ('x', 'y', 'yerr'))# time, flux/mag, flux/mag err

Aff = namedtuple('Aff', ('tx', 'ty', 'dx', 'dy'))# translation, dilation in x, y

def make_aff(list):
    aff = Aff(list[0], list[1], list[2], list[3])
    return aff

def transform(lc, aff):
    # check that error really does behave this way
    new_x = (aff.dx * lc.x) + aff.tx
    new_y = (aff.dy * lc.y) + aff.ty
    new_yerr = np.sqrt(aff.dy) * lc.yerr
    return LC(new_x, new_y, new_yerr)

def merge(lca, lcb):
    new_x = np.concatenate((lca.x, lcb.x))
    new_y = np.concatenate((lca.y, lcb.y))
    new_yerr = np.concatenate((lca.yerr, lcb.yerr))
    order = np.argsort(new_x)
    ord_x = new_x[order]
    ord_y = new_y[order]
    ord_yerr = new_yerr[order]
    return LC(ord_x, ord_y, ord_yerr)
