# -*- coding: utf-8 -*-
"""Tests for the transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from justice import xform, simulate
import lightcurve_test


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

    lightcurve_test._assert_near(after.bands['b'].time, [9, 12])
    lightcurve_test._assert_near(after.bands['b'].flux, [28, 32])
    lightcurve_test._assert_near(after.bands['b'].flux_err, [2, 2])


def test_band_name_mapper():
    bnm = xform.BandNameMapper(b=400)
    lc = simulate.TestLC.make_hard_gauss()
    lc2d = bnm.make_lc2d(lc)
    assert lc2d.invars.shape[0] == 2
    assert lc2d.outvars.shape[0] == 2
