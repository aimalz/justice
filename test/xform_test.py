# -*- coding: utf-8 -*-
"""Tests for the transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from justice import xform, simulate

@pytest.mark.skip()
def test_defaulting():
    tf = xform.Xform(tx=3.0)
    assert tf.ty == 0.0
    assert tf.dx == 1.0
    assert tf.dy == 1.0
    xform.Xform()  # Make sure defaults-only xform is ok.


def test_to_and_from_array():
    tf = xform.Xform(
        tx=0.1, ty={'a':0.2, 'b':0.2}, dx=1.0, dy={'a':2.0, 'b':2.0}, rs=.1)
    params = np.array(nest.flatten(tf))
    tf2 = xform.make_xform(params)
    assert tf == tf2


def test_transform():
    tf = xform.Xform(tx=-3215, ty={'a':0.2, 'b':0.2}, dx=10.0, dy={'a':2.0, 'b':2.0}, rs=.1)
    lca = simulate.TestLC.make_super_easy(time=np.array([3215, 3217]))
    lc = tf.transform(lca)
    assert lc.bands['b'].time[0] == 0.0
    assert lc.bands['b'].time[1] == 20
