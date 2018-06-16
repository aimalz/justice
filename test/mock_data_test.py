import numpy as np

from justice import mock_data

def test_make_gauss():
    gauss_fcn = mock_data.make_gauss(1.0)
    xs = np.arange(0.0, 1.0, 0.1)
    ys = gauss_fcn(xs)
    expected = [ 0.39894228, 0.39695255, 0.39104269, 0.38138782, 0.36827014,
    0.35206533, 0.3332246, 0.31225393, 0.28969155, 0.26608525]
    assert np.sum(np.abs(expected - ys)) < 1e-6

def test_make_dataset():
    num_obj = 10
    cls_models = [mock_data.make_gauss, mock_data.make_sine]
    cls_params = [{'scale': 10., 'loc': 100., 'amp': 50., 'const': 1.},
              {'period': 20., 'phase': 0., 'amp': 5., 'const': 5.}]
    cls_wts = None # even split for now
    def_cadence = np.arange(0., 200., 5.)
    mock_data.make_dataset(num_obj, def_cadence, cls_models, cls_params, cls_wts=None)
    return
