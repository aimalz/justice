# -*- coding: utf-8 -*-
"""Tests for the transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from justice import xform, simulate
import lightcurve_test


@pytest.mark.skip()
def test_defaulting():
    tf = xform.Xform(tx=3.0)
    assert tf.ty == 0.0
    assert tf.dx == 1.0
    assert tf.dy == 1.0
    xform.Xform()  # Make sure defaults-only xform is ok.


def test_linear_band_data_xform():
    xf = xform.LinearBandDataXform(1, 2, 3, 4)
    lc = simulate.TestLC.make_super_easy()
    after = xf.apply(lc.bands['b'])
    lightcurve_test._assert_near(after.time, [9, 12])
    lightcurve_test._assert_near(after.flux, [28, 32])
    lightcurve_test._assert_near(after.flux_err, [2, 2])


def test_independent_lc_xforms():
    xf = xform.LinearBandDataXform(1, 2, 3, 4)
    lcxf = xform.IndependentLCXform(b=xf)
    lc = simulate.TestLC.make_super_easy()
    after = lcxf.apply(lc)
