from collections import namedtuple
import numpy as np

LC = namedtuple('LC', ('x', 'y', 'yerr'))  # time, flux/mag, flux/mag err


def merge(lca, lcb):
    new_x = np.concatenate((lca.x, lcb.x))
    new_y = np.concatenate((lca.y, lcb.y))
    new_yerr = np.concatenate((lca.yerr, lcb.yerr))
    order = np.argsort(new_x)
    ord_x = new_x[order]
    ord_y = new_y[order]
    ord_yerr = new_yerr[order]
    return LC(ord_x, ord_y, ord_yerr)
