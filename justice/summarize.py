"""Tools for summarizing lightcurve data into statistics"""

import george
from george import kernels
import numpy as np
import scipy.optimize as spo

from util import LC, Aff, transform, merge, make_aff

def lineup(lca, lcb):
    # optimize the lining up for GP and arclen
    # do this on coarse grid, then refine
    pass

def connect_the_dots(lc):
    # ignores errors
    x_difs = (lc.x[1:] - lc.x[:-1])
    y_difs = lc.y[1:] - lc.y[:-1]
    sol = np.sqrt(x_difs ** 2 + y_difs ** 2)
    return np.sum(sol)

def opt_arclen(lca, lcb, ivals=np.array([0., 0., 1., 1.]), constraints=[], method='Nelder-Mead', options={'maxiter':10000}, vb=True):
    if method != 'Nelder-Mead':
        def pos_dil(aff):
            return(min(aff.dx, aff.dy))
        constraints += [{'type': 'ineq', 'fun': pos_dil}]
    else:
        constraints = None
    def _helper(vals):
        aff = make_aff(vals)
        lc = transform(lcb, aff)
        new_lc = merge(lca, lc)
        length = connect_the_dots(new_lc)
        return(length)
    res = spo.minimize(_helper, ivals, constraints=constraints, method=method, options=options)
    if vb:
        # tmp = transform(lcb, res.x)
        # fin = merge(lca, tmp)
        # debug = connect_the_dots(fin)
        return(res)
    res_aff = make_aff(res.x)
    return(res_aff)

def gp_action(kernel, lctrain, xpred):
    gp = george.GP(kernel)
    gp.compute(lctrain.x, lctrain.yerr**2)
    ypred, yprederr = gp.predict(lctrain.y, xpred, return_var=True)
    lctest = LC(xpred, ypred, np.sqrt(yprederr))
    return lctest

def fit_gp(lc):
    pass
