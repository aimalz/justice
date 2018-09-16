# -*- coding: utf-8 -*-
"""Tests for the transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from justice import xform, lightcurve


def test_defaulting():
    tf = xform.Xform(tx=3.0)
    assert tf.ty == 0.0
    assert tf.dx == 1.0
    assert tf.dy == 1.0
    xform.Xform()  # Make sure defaults-only xform is ok.


def test_to_and_from_array():
    tf = xform.Xform(tx=0.1, ty=0.2, dx=1.0, dy=2.0, bc=[.1,])
    params = tf.as_array()
    tf2 = xform.make_xform(params)
    assert tf == tf2


def test_transform():
    tf = xform.Xform(tx=-3215, ty=0.2, dx=10.0, dy=2.0, bc=[1.,])
    lca = lightcurve.LC(
        x=np.array([[3215], [3217]], dtype=np.float64),
        y=np.array([[12], [17]], dtype=np.float64),
        yerr=np.array([[1.0], [0.1]], dtype=np.float64),
    )
    lc = tf.transform(lca)
    assert lc.x[0, 0] == 0.0
    assert lc.x[1, 0] == 20
