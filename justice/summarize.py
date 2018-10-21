"""Tools for summarizing lightcurve data into statistics"""

import GPy
import numpy as np
import scipy.optimize as spo
from tensorflow.contrib.framework import nest

from justice import lightcurve
from justice import xform


def lineup(lca, lcb):
    # optimize the lining up for GP and arclen
    # do this on coarse grid, then refine
    pass


def generate_ivals(lc: lightcurve._LC) -> np.ndarray:
    """
    Generate starting values for optimization

    :param lc: Starting lightcurve
    :return: flattened list of xform parameters
    """
    numbands = lc.nbands
    return np.array(nest.flatten(lc.get_xform()))


def opt_arclen(
    lca: lightcurve._LC,
    lcb: lightcurve._LC,
    ivals=None,
    constraints=None,
    method='Nelder-Mead',
    options=None,
    vb=False,
) -> xform.Xform:
    """
    Minimizes the arclength between two lightcurves after merging

    :param lca: First lightcurve.
    :param lcb: Lightcurve to try merging in
    :param ivals: initial values to try
    :param constraints: Not sure how these work, feel free to give it a try though!
    :param method: Only Nelder_Mead is tested as of now
    :param options: Only maxiter is included right now
    :param vb: Boolean verbose
    :return: best xform
    """
    if constraints is None:
        constraints = []
    if options is None:
        options = {'maxiter': 10000}
    if ivals is None:
        ivals = generate_ivals(lca)

    if method != 'Nelder-Mead':

        def pos_dil(xf: xform.Xform):
            return min(xf.dx, xf.dy)

        constraints += [{'type': 'ineq', 'fun': pos_dil}]
    else:
        constraints = None

    def _preprocess(lca, lcb):
        ty = {}
        dy = {}  # We may want to line up xs as well
        lcs = []
        for lc in [lca, lcb]:
            for b in 'ugrizY':
                ty[b] = np.min(lc.bands[b].flux)
                dy[b] = 1. / (np.max(lc.bands[b].flux) - ty[b])
            lcs.append(xform.Xform(0., ty, 1., dy, 0.).transform(lc))
        return lcs[0], lcs[1]

    lca, lcb = _preprocess(lca, lcb)

    # don't know if this way of handling constraints actually works -- untested!
    def _helper(vals):
        lca_xform = lca.get_xform(vals=vals)
        lc = xform.transform(lcb, lca_xform)
        new_lc = lca + lc
        length = new_lc.connect_the_dots()
        return length

    # could make this a probability by taking chi^2 error relative to
    # connect_the_dots original, but it didn't work better in the sandbox
    # notebook
    res = spo.minimize(
        _helper, ivals, constraints=constraints, method=method, options=options
    )
    if vb:
        print(res)
    res_xform = lca.get_xform(res.x)
    return res_xform


def fit_gp(lctrain, kernel=None):
    gpy_data = lctrain.to_arrays()
    gp = GPy.models.gp_regression.GPRegression(gpy_data[0], gpy_data[1], normalizer=True)
    gp.optimize()
    return gp.log_likelihood()


def pred_gp(lctrain, xpred, kernel=None):
    gpy_data = lctrain.to_arrays()
    gp = GPy.models.gp_regression.GPRegression(gpy_data[0], gpy_data[1], normalizer=True)
    gp.optimize()
    ypred, yvarpred = gp.predict(xpred)
    lcpred = LC(xpred, ypred, np.sqrt(yvarpred))
    fin_like = gp.log_likelihood()
    return lcpred, fin_like


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

    def cost(self, lca: lightcurve._LC, lcb: lightcurve._LC) -> float:
        """

        :param lca: First light curve.
        :param lcb: Second light curve.
        :return:
        """

        def _time_bounds(lc):
            return (
                min(np.min(band.time) for band in lc.bands.values()),
                max(np.max(band.time) for band in lc.bands.values()),
            )

        min_lca, max_lca = _time_bounds(lca)
        min_lcb, max_lcb = _time_bounds(lcb)
        overlap = min(max_lca, max_lcb) - max(min_lca, min_lcb)
        if overlap <= 0:
            return self.cost_outside
        else:
            overlap_percent = (2 * overlap) / \
                ((max_lca - min_lca) + (max_lcb - min_lcb))
            assert 0.0 <= overlap_percent <= 1.0
            return np.interp(
                x=overlap_percent, xp=self.cost_base, fp=self.cost_percentiles
            )


def opt_gp(
    lca,
    lcb,
    ivals=None,
    constraints=None,
    method='Nelder-Mead',
    options=None,
    vb=True,
    overlap_cost_fcn=None,
    component_sensitivity=None,
):
    """Fits two light curves using a Gaussian process.

    :param lca: First light curve.
    :param lcb: Second light curve, which will be transformed.
    :param ivals: Initial values for xform.
    :param constraints: List of constraints, only used for certain methods.
    :param method: Scipy optimize method name.
    :param options: Scipy optimize method options, including maxiter.
    :param vb: Whether to print the result.
    :param overlap_cost_fcn: Optional cost function for favoring overlapping curves.
    :param component_sensitivity: Array to scale parameters by. Doesn't seem to
        be affecting/fixing optimizer that much yet.
    :return: Resulting transformation for lcb.
    """
    if constraints is None:
        constraints = []
    if options is None:
        options = {'maxiter': 10000}
    if ivals is None:
        ivals = generate_ivals(lca)

    if component_sensitivity == 'auto':
        component_sensitivity = xform.Xform(
            tx=np.std(lca.x) + np.std(lcb.x),
            ty=1.0,
            dx=1.0,
            dy=1.0,
        ).as_array()

    if component_sensitivity is not None:
        ivals = ivals / component_sensitivity

    if method != 'Nelder-Mead':
        # don't know if this way of handling constraints actually works -- untested!
        def pos_dil(xf: xform.Xform):
            return min(xf.dx, xf.dy)

        constraints += [{'type': 'ineq', 'fun': pos_dil}]
    else:
        constraints = None

    def _helper(vals):
        if component_sensitivity is not None:
            vals = vals * component_sensitivity
        lca_xform = lca.get_xform(vals=vals)
        lc = xform.transform(lcb, lca_xform)
        new_lc = lca + lc
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

    resxform = lca.get_xform(res.x)
    return resxform
