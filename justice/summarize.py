"""Tools for summarizing lightcurve data into statistics"""

import george
from george import kernels
import numpy as np

from util import LC, Aff

def transform(lc, aff):
    new_x = (aff.dx * lc.x) + aff.tx
    new_y = (aff.dy * lc.y) + aff.ty
    new_yerr = np.sqrt(aff.dy) * lc.yerr)
    return LC(new_x, new_y, new_yerr)

def lineup(lca, lcb):
    # do this on coarse grid, then refine
    pass

def merge(lca, lcb):
    new_x = np.concatenate((lca.x, lcb.x))
    new_y = np.concatenate((lca.y, lcb.y))
    new_yerr = np.concatenate((lca.yerr, lcb.yerr))
    order = np.argsort(new_x)
    ord_x = new_x[order]
    ord_y = new_y[order]
    ord_yerr = new_yerr[order]
    return LC(ord_x, ord_y, ord_yerr)

def connect_the_dots(lc):
    # ignores errors
    x_difs = (lc.x[1:] - lc.x[:-1])
    y_difs = lc.y[1:] - lc.y[:-1]
    sol = np.sqrt(x_difs ** 2 + y_difs ** 2)
    return np.sum(sol)

def gp_action(kernel, lctrain, xpred):
    gp = george.GP(kernel)
    gp.compute(lctrain.x, lctrain.yerr**2)
    ypred, yerrpred = gp.predict(lctrain.y, xpred, return_var=True)
    return LC(xpred, ypred, yerrpred)

def opt_arclen(lca, lcb, ivals=Aff(0., 0., 1., 1.), method='Nelder-Mead', options={'maxiter':10000}, vb=True):
    def _helper(aff):
        # (deltax, deltay, stretchx, stretchy) = aff
        lc = transform(lcb, aff)#deltax, deltay, stretchx, stretchy)
#         new_len = connect_the_dots(lc)
        new_lc = merge(lca, lc)
        length = connect_the_dots(new_lc)
        to_min = length
        return(to_min)
    res = spo.minimize(_helper, ivals, method=method, options=options)
    tmp = transform(lcb, res.x)
    fin = merge(lca, tmp)
    if vb:
        debug = connect_the_dots(fin)
        return(res, debug)
    return(res)
