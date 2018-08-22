from collections import namedtuple
import numpy as np

"""
Note: x, y, yerr all need to be __lists__ of numpy arrays, bc number of obvs might not be the same.
"""

LC = namedtuple('LC', ('x', 'y', 'yerr'))  # time, flux/mag, flux/mag err


def merge(lca, lcb):
    ord_x = []
    ord_y = []
    ord_yerr = []
    print (lca, lcb)
    for i,j in enumerate(lca.x):
        print (lca.x[i], lcb.x[i], lca.y[i], lcb.y[i])
        new_x = np.concatenate((lca.x[i], lcb.x[i]))
        new_y = np.concatenate((lca.y[i], lcb.y[i]))
        new_yerr = np.concatenate((lca.yerr[i], lcb.yerr[i]))
        order = np.argsort(new_x)
        print (len(new_x), len(new_y), len(new_yerr))
        ord_x.append(new_x[order])
        ord_y.append(new_y[order])
        ord_yerr.append(new_yerr[order])
    return LC(ord_x, ord_y, ord_yerr)
