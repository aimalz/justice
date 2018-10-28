# -*- coding: utf-8 -*-
"""Tests for the light curve class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from justice import lightcurve
from justice.datasets.ogle_data import OGLEDatasetLC


def _assert_near(a, b, eps=1e-5):
    a = np.array(a, copy=False)
    b = np.array(b, copy=False)
    assert np.mean(np.abs(a - b)) < eps, "{} not close to {}".format(a, b)
