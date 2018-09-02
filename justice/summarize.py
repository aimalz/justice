"""Tools for summarizing lightcurve data into statistics"""

import GPy
import numpy as np
import scipy.optimize as spo

from justice.affine_xform import Aff, transform
from justice.lightcurve import LC, merge


def lineup(lca, lcb):
    # optimize the lining up for GP and arclen
    # do this on coarse grid, then refine
    pass


def connect_the_dots(lc):
    # ignores errorbars
    sol = 0.
    for x, y, yerr in zip(lc.x.T, lc.y.T, lc.yerr.T):
        x_difs = (x[1:] - x[:-1])
        y_difs = y[1:] - y[:-1]
        sol += np.sum(np.sqrt(x_difs**2 + y_difs**2))
    return sol


def opt_arclen(
    lca,
    lcb,
    ivals=np.array([0., 0., 1., 1.]),
    constraints=[],
    method='Nelder-Mead',
    options={'maxiter': 10000},
    vb=True
):
    if method != 'Nelder-Mead':

        def pos_dil(aff):
            return (min(aff.dx, aff.dy))

        constraints += [{'type': 'ineq', 'fun': pos_dil}]
    else:
        constraints = None

    # don't know if this way of handling constraints actually works -- untested!
    def _helper(vals):
        aff = Aff(*vals)
        lc = transform(lcb, aff)
        new_lc = merge(lca, lc)
        length = connect_the_dots(new_lc)
        return (length)

    # could make this a probability by taking chi^2 error relative to
    # connect_the_dots original, but it didn't work better in the sandbox
    # notebook
    res = spo.minimize(
        _helper, ivals, constraints=constraints, method=method, options=options
    )
    if vb:
        print(res)
    res_aff = Aff(*res.x)
    return (res_aff)


def fit_gp(lctrain, xpred, kernel=None):
    gp = GPy.models.gp_regression.GPRegression(lctrain.x, lctrain.y, normalizer=True)
    gp.optimize()

    ypred, yvarpred = gp.predict(xpred)
    lcpred = LC(xpred, ypred, np.sqrt(yvarpred))
    fin_like = gp.log_likelihood()
    return (lcpred, fin_like)


def opt_gp(
    lca,
    lcb,
    ivals=np.array([0., 0., 1., 1.]),
    constraints=[],
    method='Nelder-Mead',
    options={'maxiter': 10000},
    vb=True
):
    if method != 'Nelder-Mead':

        def pos_dil(aff):
            return (min(aff.dx, aff.dy))

        constraints += [{'type': 'ineq', 'fun': pos_dil}]
    else:
        constraints = None
    # Not general! Need to change.
    def_cadence = np.vstack((np.arange(0., 1000., 10.), np.arange(0., 1000., 10.))).T

    # don't know if this way of handling constraints actually works -- untested!
    def _helper(vals):
        aff = Aff(*vals)
        lc = transform(lcb, aff)
        new_lc = merge(lca, lc)
        pred, fin_like = fit_gp(new_lc, def_cadence)
        return (-fin_like)

    # could make this a probability by taking chi^2 error relative to
    # connect_the_dots original, but it didn't work better in the sandbox
    # notebook
    res = spo.minimize(
        _helper, ivals, constraints=constraints, method=method, options=options
    )
    if vb:
        print(res)
    res_aff = Aff(*res.x)
    return (res_aff)
