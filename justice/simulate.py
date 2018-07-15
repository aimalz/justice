"""Simulates mock data"""

from collections import namedtuple
import numpy as np
import scipy.stats as sps

LC = namedtuple('LC', ('x', 'y', 'yerr'))# time, flux/mag, flux/mag err

Aff = namedtuple('Aff', ('tx', 'ty', 'dx', 'dy'))# translation, dilation in x, y

def transform(lc, aff):
    # check that error really does behave this way
    new_x = (aff.dx * lc.x) + aff.tx
    new_y = (aff.dy * lc.y) + aff.ty
    new_yerr = np.sqrt(aff.dy) * lc.yerr
    return LC(new_x, new_y, new_yerr)

def make_gauss(scale, loc=0., amp=1., const=0.):
    func = sps.norm(loc, scale)
    peakval = func.pdf(loc)
    ampfact = amp / peakval
    out = lambda x: ampfact * func.pdf(x) + const
    return out

def make_sine(period, phase=0., amp=1., const=0.):
    const += amp
    func = lambda x: amp * (np.sin(period * x + phase)) + const
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
        x = make_cadences(def_cadence, 0.5)
        model = cls_models[cls_id](**cls_params[cls_id])
        y, yerr = make_err(model(x), 0.1)
        lcs.append(LC(x, y, yerr))
    return lcs
