import numpy as np
from math import pi
import justice.simulate as sim


def test_make_gauss():
    gauss_fcn = sim.make_gauss_shape_fn(1.0, 0, 1, 0)
    xs = sim.make_cadence(np.arange(0.0, 1.0, 0.1), [0.] * 10)
    ys = gauss_fcn(xs)
    expected = np.array([
        1., 0.99501248, 0.98019867, 0.95599748, 0.92311635, 0.8824969, 0.83527021,
        0.78270454, 0.72614904, 0.66697681
    ])
    assert np.sum(np.abs(expected - ys)) < 1e-6


def test_make_sine():

    sine_fcn = sim.make_sine_shape_fn(1.0, 0, 1, 0)
    xs = sim.make_cadence(np.arange(0.0, pi * 9 / 8, pi / 8), [0.] * 9)
    ys = sine_fcn(xs)
    assert np.abs(ys[0]) < 1e-6
    assert np.abs(ys[4] - 1) < 1e-6
    assert np.abs(ys[8]) < 1e-6


def test_make_dataset():
    num_obj = 10
    gauss = sim.make_gauss_shape_fn(10, 100, 50, 1)
    sine = sim.make_sine_shape_fn(20, 0, 5, 5)
    cls_wts = None  # even split for now
    xs = np.arange(0., 200., 5.)
    sim.make_dataset(num_obj, xs, [gauss, sine], cls_wts=cls_wts)
    return
