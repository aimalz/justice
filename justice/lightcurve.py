from collections import namedtuple
import numpy as np
"""
Note: x, y, yerr all need to be arrays with shape (N, band)
"""
LC = namedtuple('LC', ('x', 'y', 'yerr'))  # time, flux/mag, flux/mag err


def merge(lca, lcb):
    ord_x = []
    ord_y = []
    ord_yerr = []
    for i in range(lca.x.shape[1]):
        new_x = np.concatenate((lca.x[:, i], lcb.x[:, i]))
        new_y = np.concatenate((lca.y[:, i], lcb.y[:, i]))
        new_yerr = np.concatenate((lca.yerr[:, i], lcb.yerr[:, i]))
        order = np.argsort(new_x)
        ord_x.append(new_x[order])
        ord_y.append(new_y[order])
        ord_yerr.append(new_yerr[order])
    return LC(np.array(ord_x).T, np.array(ord_y).T, np.array(ord_yerr).T)
