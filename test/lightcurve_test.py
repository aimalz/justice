# -*- coding: utf-8 -*-
"""Tests for the light curve class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from justice import lightcurve


def _assert_near(a, b, eps=1e-5):
    a = np.array(a, copy=False)
    b = np.array(b, copy=False)
    assert np.mean(np.abs(a - b)) < eps, "{} not close to {}".format(a, b)


def test_per_band_centering():
    first = lightcurve.BandData.from_dense_array(
        np.array([
            [1, 10, 0.1],
            [2, 11, 0.11],
            [3, 12, 0.1],
            [4, 12, 0.1],
            [5, 12, 0.12],
        ])
    )
    second = lightcurve.BandData.from_dense_array(
        np.array([
            [4.7, 1, 0.1],
            [5, 2, 0.14],
        ])
    )
    lc = lightcurve.OGLEDatasetLC(I=first, V=second)
    xf = lc.per_band_normalization(100.0)
    lct = xf.transform(lc)
    _assert_near(lct['I'].time, [-2.15, -1.15, -0.15, 0.85, 1.85])
    _assert_near(lct['V'].time, [1.55, 1.85])
    _assert_near(np.max(lct['I'].flux), 100)
    _assert_near(np.max(lct['V'].flux), 100)
