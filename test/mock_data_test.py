import numpy as np

from justice import mock_data

def test_make_gauss():
    gauss_fcn = mock_data.make_gauss(1.0)
    xs = np.arange(0.0, 1.0, 0.1)
    ys = gauss_fcn(xs)
    expected = [ 0.39894228, 0.39695255, 0.39104269, 0.38138782, 0.36827014,
    0.35206533, 0.3332246, 0.31225393, 0.28969155, 0.26608525]
    assert np.sum(np.abs(expected - ys)) < 1e-6
