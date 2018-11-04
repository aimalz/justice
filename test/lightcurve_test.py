# -*- coding: utf-8 -*-
"""Tests for the light curve class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from justice import simulate, lightcurve


def _assert_near(a, b, eps=1e-5):
    a = np.array(a, copy=False)
    b = np.array(b, copy=False)
    assert np.mean(np.abs(a - b)) < eps, "{} not close to {}".format(a, b)


def test_before_mask():
    lc = simulate.TestLC.make_super_easy()
    band = lc.bands['b']
    assert band.before_time(1.5).flux.tolist() == []
    assert band.before_time(2.5).flux.tolist() == [5]
    assert band.before_time(3.5).flux.tolist() == [5, 6]

    assert band.after_time(1.5).flux.tolist() == [5, 6]
    assert band.after_time(2.5).flux.tolist() == [6]
    assert band.after_time(3.5).flux.tolist() == []

    assert band.closest_point(2.1) == lightcurve.BandPoint(2, 5, 1, 1)
    assert band.closest_point(2.7) == lightcurve.BandPoint(3, 6, 1, 1)
