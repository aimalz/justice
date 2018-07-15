"""Tools for summarizing lightcurve data into statistics"""

import george
from george import kernels
import numpy as np
import scipy.optimize as spo

from simulate import LC, Aff, transform

def lineup(lca, lcb):
    # optimize the lining up for GP and arclen
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
    # ignores errorbars
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
    # don't know if this way of handling constraints actually works -- untested!
    def _helper(vals):
        aff = Aff(*vals)
        lc = transform(lcb, aff)
        new_lc = merge(lca, lc)
        length = connect_the_dots(new_lc)
        return(length)
    # could make this a probability by taking chi^2 error relative to connect_the_dots original, but it didn't work better in the sandbox notebook
    res = spo.minimize(_helper, ivals, constraints=constraints, method=method, options=options)
    if vb:
        print(res)
    res_aff = Aff(*res.x)
    return(res_aff)

def gp_train(kernel, lctrain):
    gp = george.GP(kernel)
    gp.compute(lctrain.x, lctrain.yerr**2)
    return gp

def gp_pred(kernel, lctrain, xpred):
    gp = gp_train(kernel, lctrain)
    ypred, yprederr = gp.predict(lctrain.y, xpred, return_var=True)
    lcpred = LC(xpred, ypred, np.sqrt(yprederr))
    return lcpred

def fit_gp(kernel, lctrain, xpred):
    gp = gp_train(kernel, lctrain)
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -1. * gp.log_likelihood(lctrain.y)
    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -1. * gp.grad_log_likelihood(lctrain.y)
    result = spo.minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    # print(result.x)
    gp.set_parameter_vector(result.x)
    # print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(lctrain.y)))
    ypred, yvarpred = gp.predict(lctrain.y, xpred, return_var=True)
    lcpred = LC(xpred, ypred, np.sqrt(yvarpred))
    fin_like = gp.log_likelihood(lctrain.y)
    return(lcpred, fin_like)
