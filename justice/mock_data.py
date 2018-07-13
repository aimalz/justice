"""Simulates mock data"""

import numpy as np
import scipy.stats as sps

from util import LC, Aff, transform

def make_gauss(scale, loc=0., amp=1., const=0.):
    func = sps.norm(loc, scale)
    out = lambda x: amp * func.pdf(x) + const
    return out


def make_sine(period, phase=0., amp=1., const=0.):
    func = lambda x: amp * (np.sin(period * x + phase)) + const
    return func


def sample_obs(x, err):
    assert (np.all((x[1:] - x[:-1]) > xerr))
    jitter = (np.random.uniform(np.shape(x)) - 0.5) * xerr * 2.
    new_x = x + jitter
    return new_x


def noisify_obs(y, yerr):
    # only uniform errors, can't differ at each point
    new_yerr = yerr * np.ones_like(y)
    new_y = y + sps.norm(0., yerr).rvs(np.shape(y))
    return (new_y, new_yerr)

def make_dataset(num_obj, def_cadence, cls_models, cls_params, cls_wts=None):
    num_cls = len(cls_models)
    lcs = []
    true_cls = np.random.choice(range(num_cls), num_obj, p=cls_wts)
    for cls_id in true_cls:
        x = sample_obs(def_cadence, 0.5)
        model = cls_models[cls_id](**cls_params[cls_id])
        y, yerr = noisify_obs(model(x), 0.1)
        lcs.append(LC(x, y, yerr))
    return lcs
