"""Tools for summarizing lightcurve data into statistics"""

import GPy
import numpy as np
import scipy.optimize as spo
from tensorflow.contrib.framework import nest

from justice import lightcurve
from justice import xform


def opt_alignment(
    lca: lightcurve._LC,
    lcb: lightcurve._LC,
    ivals=None,
    constraints=None,
    method='Nelder-Mead',
    options=None,
    vb=True,
) -> xform.LCXform:
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
        ivals = xform.LinearBandDataXform.ivals()

    if method != 'Nelder-Mead':

        def pos_dil(xf: xform.LinearBandDataXform):
            return min(xf._dilate_time, xf._dilate_flux)

        constraints += [{'type': 'ineq', 'fun': pos_dil}]
    else:
        constraints = None

    # don't know if this way of handling constraints actually works -- untested!
    def _helper(vals):
        bd_xform = xform.LinearBandDataXform(*vals)
        lca_xform = xform.SimultaneousLCXform(bd_xform)
        lc = lca_xform.apply(lcb)
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
    res_xform = xform.SimultaneousLCXform(xform.LinearBandDataXform(*res.x))
    return res_xform


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
            overlap_percent = (2 * overlap) / ((max_lca - min_lca) + (max_lcb - min_lcb))
            assert 0.0 <= overlap_percent <= 1.0
            return np.interp(
                x=overlap_percent, xp=self.cost_base, fp=self.cost_percentiles
            )
