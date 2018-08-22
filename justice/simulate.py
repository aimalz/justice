"""Simulates mock data"""

import numpy as np
import scipy.stats as sps
from justice.lightcurve import LC


def make_gauss(scales, locs=[0.,], amps=[1.,], consts=[0.,]):
    funcs = []
    ampfacts = []
    for scale, loc, amp, const in zip(scales,locs,amps,consts):
        func = sps.norm(loc, scale)
        funcs.append(func)
        peakval = func.pdf(loc)
        ampfacts.append(amp / peakval)

    def out(xs):
        ys = []
        for ampfact, func, x, const in zip(ampfacts, funcs, xs, consts):
            ys.append(ampfact * func.pdf(x) + const)
        return ys
    return out


def make_sine(periods, phases=[0.,], amps=[1.,], consts=[0.,]):
    # returns a sinusoid function that is always non-negative

    def out(xs):
        ys = []
        for period, phase, amp, const, x in zip(periods, phases, amps, consts, xs):
            ys.append(amp * (np.sin(period * x + phase)) + (const + amp))
        return ys
    return out


def make_cadence(xs, xerrs):
    new_xs = []
    for x, xerr in zip(xs, xerrs):
        assert (np.all((x[1:] - x[:-1]) > xerr))
        jitter = (np.random.uniform(np.shape(x)) - 0.5) * xerr
        new_xs.append(x + jitter)
    return new_xs


def apply_err(ys, yerrfracs):
    # only uniform errors, can't differ at each point
    new_ys = []
    new_yerrs = []
    for y, yerrfrac in zip(ys, yerrfracs):
        yerr = yerrfrac * y
        new_ys.append(y + sps.norm(0., yerr).rvs(np.shape(y)))
        new_yerrs.append(yerr)
    return (new_ys, new_yerrs)


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
