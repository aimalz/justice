"""Simulates mock data"""

import numpy as np
import scipy.stats as sps
from justice.lightcurve import LC


def make_gauss(scale, loc=0., amp=1., const=0.):
    func = sps.norm(loc, scale)
    peakval = func.pdf(loc)
    ampfact = amp / peakval

    def out(x):
        return ampfact * func.pdf(x) + const
    return out


def make_sine(period, phase=0., amp=1., const=0.):
    # returns a sinusoid function that is always non-negative

    def func(x):
        return amp * (np.sin(period * x + phase)) + (const + amp)
    return func


def make_cadence(x, xerr):
    assert (np.all((x[1:] - x[:-1]) > xerr))
    jitter = (np.random.uniform(np.shape(x)) - 0.5) * xerr
    new_x = x + jitter
    return new_x


def apply_err(y, yerrfrac):
    # only uniform errors, can't differ at each point
    yerr = yerrfrac * y
    new_y = y + sps.norm(0., yerr).rvs(np.shape(y))
    return (new_y, yerr)


def make_dataset(num_obj, def_cadence, cls_models, cls_params, cls_wts=None):
    num_cls = len(cls_models)
    lcs = []
    true_cls = np.random.choice(range(num_cls), num_obj, p=cls_wts)
    for cls_id in true_cls:
        x = make_cadence(def_cadence, 0.5)
        model = cls_models[cls_id](**cls_params[cls_id])
        y, yerr = apply_err(model(x), 0.1)
        lcs.append(LC(x, y, yerr))
    return lcs
