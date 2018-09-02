import numpy as np

import justice.simulate as sim


def test_make_gauss():
    gauss_fcn = sim.make_gauss([1.0, ])
    xs = sim.make_cadence([np.arange(0.0, 1.0, 0.1), ], [0.])
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
    assert np.sum(np.abs(expected - ys[:, 0])) < 1e-6


def test_make_gauss_multiband():
    gauss_fcn = sim.make_gauss([1.0, 2.0], locs=[0, 1], amps=[1, 2], consts=[2, 3])
    xs = sim.make_cadence([np.arange(0.0, 1.0, 0.1), np.arange(.5, 2.5, 0.3)], [0., 0.])
    ys = gauss_fcn(xs)
    expected = np.array([np.array([3., 2.99501248, 2.98019867, 2.95599748, 2.92311635,
                                   2.8824969, 2.83527021, 2.78270454, 2.72614904, 2.66697681]),
                         np.array([4.93846647, 4.99002496, 4.99750156, 4.96039735, 4.88117613,
                                   4.76499381, 4.6191433])], dtype=object)
    for color_idx in range(len(expected)):
        np.testing.assert_allclose(ys[color_idx], expected[color_idx])


def test_make_dataset():
    num_obj = 10
    cls_models = [sim.make_gauss, sim.make_sine]
    cls_params = [{'scales': [10.], 'locs': [100.], 'amps': [50.], 'consts': [1.]},
                  {'periods': [20.], 'phases': [0.], 'amps':[5.], 'consts': [5.]}]
    cls_wts = None  # even split for now
    def_cadence = [np.arange(0., 200., 5.), ]
    sim.make_dataset(num_obj, def_cadence, cls_models, cls_params, cls_wts=None)
    return
