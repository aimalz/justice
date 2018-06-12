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
