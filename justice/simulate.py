import numpy as np
import scipy.stats as sps

from justice import lightcurve


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
        cadence = make_cadence(xs, np.array([
            0.5,
        ] * len(xs)))
        bands.append(
            lightcurve.BandData.from_cadence_shape_and_errfracs(
                cadence, cls, np.array([0.1] * len(xs))
            )
        )

    return bands


class TestLC(lightcurve._LC):
    @property
    def expected_bands(self):
        return ['b']

    @classmethod
    def make_super_easy(cls, time=None):
        time = time if time is not None else np.array([2, 3])
        band = lightcurve.BandData(
            time=time, flux=np.array([5, 6]), flux_err=np.array([1, 1])
        )
        return TestLC(b=band)

    @classmethod
    def make_easy_gauss(cls):
        gauss_fcn = make_gauss_shape_fn(1.0, 0, 1, 0)
        xs = make_cadence(np.arange(0, 22.6, .1), [0.] * 226)

        band = lightcurve.BandData.from_cadence_shape_and_errfracs(
            xs, gauss_fcn, [0] * 226
        )
        return TestLC(b=band)

    @classmethod
    def make_hard_gauss(cls, scale=1.5) -> 'TestLC':
        """Makes a difficult Gaussian value.

        :param scale: Width of gaussian.
        :return: Light curve.
        """
        gauss_fcn = make_gauss_shape_fn(scale, 1, 1.2, .3)
        xs = make_cadence(np.arange(0, 11.5, .1), [0.025] * 115)

        band = lightcurve.BandData.from_cadence_shape_and_errfracs(
            xs, gauss_fcn, [0.2] * 115
        )
        return TestLC(b=band)

    @classmethod
    def make_realistic_gauss(cls, scale=30.0, xerr=0.25, noise=0.1) -> 'TestLC':
        gauss_fcn = make_gauss_shape_fn(scale=scale, center=20, amplitude=1.2, const=.3)
        xs = make_cadence(np.arange(0, 230, 2), [xerr] * 115)

        band = lightcurve.BandData.from_cadence_shape_and_errfracs(
            xs, gauss_fcn, [noise] * 115
        )
        return TestLC(b=band)
