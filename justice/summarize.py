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
    vb=False,
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
        ivals = np.array([0, 0, 1, 1])

    if method != 'Nelder-Mead':

        def pos_dil(xf: xform.LinearBandDataXform):
            return min(xf._dilate_time, xf._dilate_flux)

        constraints += [{'type': 'ineq', 'fun': pos_dil}]
    else:
        constraints = None

    def _preprocess(lca, lcb):
        lcs = []
        for lc in [lca, lcb]:
            band_xforms = {}
            for b in lc.bands:  # 'ugrizY':
                ty = np.min(lc.bands[b].flux)
                band_xforms[b] = xform.LinearBandDataXform(
                    translate_time=0,
                    translate_flux=ty,
                    dilate_time=0,
                    dilate_flux=1. / (np.max(lc.bands[b].flux) - ty),
                )
            lc_xform = xform.IndependentLCXform(**band_xforms)
            lcs.append(lc_xform.apply(lc))
        return lcs[0], lcs[1]

    lca, lcb = _preprocess(lca, lcb)

    # don't know if this way of handling constraints actually works -- untested!
    def _helper(vals):
        bd_xform = xform.LinearBandDataXform(*vals)
        lca_xform = xform.SameLCXform(bd_xform)
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
    res_xform = xform.SameLCXform(xform.LinearBandDataXform(*res.x))
    return res_xform
