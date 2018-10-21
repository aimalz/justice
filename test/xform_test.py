# -*- coding: utf-8 -*-
"""Tests for the transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from justice import xform, simulate, lightcurve
from test import lightcurve_test
from tensorflow.contrib.framework import nest


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
'''
def test_transform():
    tf = xform.Xform(tx=-3215, ty={'a':0.2, 'b':0.2}, dx=10.0, dy={'a':2.0, 'b':2.0}, rs=.1)
    lca = simulate.TestLC.make_super_easy(time=np.array([3215, 3217]))
    lc = tf.transform(lca)
    assert lc.bands['b'].time[0] == 0.0
    assert lc.bands['b'].time[1] == 20
'''
