import numpy as np
import scipy.stats as sps
from justice.lightcurve import BandData, _LC


def make_cadence(xs, xerrs):
    # A cadence of observation times
    xs = np.array(xs)
    xerrs = np.array(xerrs)
    assert np.all((xs[1:] - xs[:-1] > xerrs[1:] + xerrs[:-1]))
    jitters = (np.random.uniform(np.shape(xs)) - 0.5) * xerrs
    return xs + jitters

# A shape function takes a cadence and output fluxes


def make_gauss_shape_fn(scale, center, amplitude, const):
    dist = sps.norm(center, scale)
    peakval = dist.pdf(center)
    ampfact = amplitude / peakval

    def gauss(cadence):
        return dist.pdf(cadence) * ampfact + const
    return gauss


def make_sine_shape_fn(period, phase, amplitude, const):

    def sine(cadence):
        return amplitude * np.sin(period * cadence + phase) + const

    return sine


def make_dataset(num_obj, xs, shape_fn, cls_wts=None):
    bands = []
    true_cls = np.random.choice(shape_fn, num_obj, p=cls_wts)
    for cls in true_cls:
        cadence = make_cadence(xs, np.array([0.5, ] * len(xs)))
        bands.append(BandData.from_cadence_shape_and_errfracs(cadence, cls, np.array([0.1] * len(xs))))

    return bands


class TestLC(_LC):
    @property
    def _expected_bands(self):
        return 'b'

    @classmethod
    def make_super_easy(cls):
        band = BandData(np.array([2, 3]), np.array([5, 6]), np.array([1, 1]))
        return TestLC(b=band)

    @classmethod
    def make_easy_gauss(cls):
        gauss_fcn = make_gauss_shape_fn(1.0, 0, 1, 0)
        xs = make_cadence(np.arange(0, 22.6, .1), [0.] * 226)

        band = BandData.from_cadence_shape_and_errfracs(xs, gauss_fcn, [0] * 226)
        return TestLC(b=band)

    @classmethod
    def make_hard_gauss(cls):
        gauss_fcn = make_gauss_shape_fn(1.5, 1, 1.2, .3)
        xs = make_cadence(np.arange(0, 11.5, .1), [0.05] * 115)

        band = BandData.from_cadence_shape_and_errfracs(xs, gauss_fcn, [0.2] * 115)
        return TestLC(b=band)
