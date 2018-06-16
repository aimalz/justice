from collections import namedtuple
import itertools
import random
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import matplotlib.pyplot as plt
import corner

def make_gauss(scale, loc=0., amp=1., const=0.):
    func = sps.norm(loc, scale)
    out = lambda x: amp * func.pdf(x) + const
    return out

def make_sine(period, phase=0., amp=1., const=0.):
    func = lambda x: amp * (np.sin(period * x + phase)) + const
    return func

def make_cadence(x, scatter):
    assert(np.all((x[1:]-x[:-1]) > scatter))
    jitter = (np.random.uniform(np.shape(x)) - 0.5) * scatter * 2.
    perturbed = x + jitter
    return perturbed

def noisify_obs(y, scatter):
    errs = scatter * np.ones_like(y)
    new_y = y + sps.norm(0., scatter).rvs(np.shape(y))
    return(new_y, errs)

LC = namedtuple('LC', ('x', 'y'))#, 'yerr'))

def transform(lc, deltax, deltay, stretchx, stretchy):
    new_x = (stretchx * lc.x) + deltax
    new_y = (stretchy * lc.y) + deltay
    # new_yerr = np.sqrt(np.abs(new_y / lc.y) * lc.yerr**2)
    return LC(new_x, new_y)#, lc.yerr)

def make_dataset(num_obj, def_cadence, cls_models, cls_params, cls_wts=None):
    num_cls = len(cls_models)
    lcs = []
    truth = np.random.choice(range(num_cls), num_obj, p=cls_wts)
    ids, inds, cts = np.unique(truth, return_counts=True, return_inverse=True)
    for i in range(num_obj):
        times = make_cadence(def_cadence, 0.5)
        model = cls_models[ids[inds[i]]](**cls_params[ids[inds[i]]])
        phot, err = noisify_obs(model(times), 0.1)
        lcs.append(LC(times, phot))
    return(lcs)
