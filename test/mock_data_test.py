import numpy as np

import justice.simulate as sim


def test_make_gauss():
    gauss_fcn = sim.make_gauss([1.0,])
    xs = [np.arange(0.0, 1.0, 0.1),]
    ys = gauss_fcn(xs)
    expected = [1.,
                0.99501248,
                0.98019867,
                0.95599748,
                0.92311635,
                0.8824969,
                0.83527021,
                0.78270454,
                0.72614904,
                0.66697681]
    assert np.sum(np.abs(expected - ys[0])) < 1e-6


def test_make_dataset():
    num_obj = 10
    cls_models = [sim.make_gauss, sim.make_sine]
    cls_params = [{'scales': [10.], 'locs': [100.], 'amps': [50.], 'consts': [1.]},
                  {'periods': [20.], 'phases': [0.], 'amps':[5.], 'consts': [5.]}]
    cls_wts = None  # even split for now
    def_cadence = [np.arange(0., 200., 5.),]
    sim.make_dataset(num_obj, def_cadence, cls_models, cls_params, cls_wts=None)
    return
