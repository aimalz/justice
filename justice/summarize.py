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
    vb=True,
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


def fit_gp(lctrain, kernel=None):
    gp = GPy.models.gp_regression.GPRegression(lctrain.x, lctrain.y, normalizer=True)
    gp.optimize()

    return gp.log_likelihood()


def pred_gp(lctrain, xpred, kernel=None):
    gp = GPy.models.gp_regression.GPRegression(lctrain.x, lctrain.y, normalizer=True)
    gp.optimize()
    ypred, yvarpred = gp.predict(xpred)
    lcpred = LC(xpred, ypred, np.sqrt(yvarpred))
    fin_like = gp.log_likelihood()
    return (lcpred, fin_like)


class OverlapCostComponent(object):
    __slots__ = ("cost_outside", "cost_percentiles", "cost_base")

    def __init__(self, cost_percentiles, cost_outside=None):
        """
        Represents the cost component for overlap.

        :param cost_percentiles: Array with costs by percentile. e.g. if the array
            is [1.0, 0.5, 0.0], and the light curves overlap 40%, then the result will be 0.6.
            Generally the cost should decrease as the light curves overlap more.
        :param cost_outside: Cost outside array; defaults to None. Can be something
            higher than `cost_percentiles` if desired.
        """
        for x, x_next in zip(cost_percentiles, cost_percentiles[1:]):
            if x_next > x:
                raise ValueError("Expected decreasing sequence.")
        self.cost_percentiles = np.array(cost_percentiles, dtype=np.float64)
        self.cost_outside = float(
            cost_outside if cost_outside is not None else cost_percentiles[0]
        )
        self.cost_base = np.linspace(0, 1, len(cost_percentiles))

    def cost(self, lca, lcb):
        min_lca, max_lca = np.min(lca.x), np.max(lca.x)
        min_lcb, max_lcb = np.min(lcb.x), np.max(lcb.x)
        overlap = min(max_lca, max_lcb) - max(min_lca, min_lcb)
        if overlap <= 0:
            return self.cost_outside
        else:
            overlap_percent = (2 * overlap) / (
                (max_lca - min_lca) + (max_lcb - min_lcb)
            )
            assert 0.0 <= overlap_percent <= 1.0
            return np.interp(x=overlap_percent, xp=self.cost_base, fp=self.cost_percentiles)


def opt_gp(
    lca,
    lcb,
    ivals=np.array([0., 0., 1., 1.]),
    constraints=[],
    method='Nelder-Mead',
    options={'maxiter': 10000},
    vb=True,
    overlap_cost_fcn=None,
    component_sensitivity=None,
):
    """Fits two light curves using a Gaussian process.

    :param lca: First light curve.
    :param lcb: Second light curve, which will be transformed.
    :param ivals: Initial values for affine_xform.
    :param constraints: List of constraints, only used for certain methods.
    :param method: Scipy optimize method name.
    :param options: Scipy optimize method options, including maxiter.
    :param vb: Whether to print the result.
    :param overlap_cost_fcn: Optional cost function for favoring overlapping curves.
    :param component_sensitivity: Array to scale parameters by. Doesn't seem to
        be affecting/fixing optimizer that much yet.
    :return: Resulting affine transformation for lcb.
    """
    if component_sensitivity == 'auto':
        component_sensitivity = Aff(
            tx=np.std(lca.x) + np.std(lcb.x),
            ty=1.0,
            dx=1.0,
            dy=1.0,
        ).as_array()

    if component_sensitivity is not None:
        ivals = ivals / component_sensitivity

    if method != 'Nelder-Mead':
        # don't know if this way of handling constraints actually works -- untested!
        def pos_dil(aff):
            return (min(aff.dx, aff.dy))

        constraints += [{'type': 'ineq', 'fun': pos_dil}]
    else:
        constraints = None

    def _helper(vals):
        if component_sensitivity is not None:
            vals = vals * component_sensitivity
        aff = Aff(*vals)
        lc = transform(lcb, aff)
        new_lc = merge(lca, lc)
        fin_like = fit_gp(new_lc)

        overlap_cost = 0.0
        if overlap_cost_fcn is not None:
            overlap_cost = overlap_cost_fcn.cost(lca, lc)
            overlap_cost = np.abs(fin_like) * overlap_cost

        return (-fin_like) + overlap_cost

    res = spo.minimize(
        _helper, ivals, constraints=constraints, method=method, options=options
    )
    if component_sensitivity is not None:
        res.x *= component_sensitivity

    if vb:
        print(res)
    return Aff(*res.x)
