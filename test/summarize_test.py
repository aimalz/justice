# -*- coding: utf-8 -*-
"""Tests for overlap light curve merging logic."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from justice import lightcurve, summarize


def test_overlap_cost_decreases_as_overlap_increases():
    cost_fcn = summarize.OverlapCostComponent([1.0, 0.1, 0.0])
    lca = lightcurve.LC(x=np.array([0.0, 10.0]), y=None, yerr=None)
    lcb = lightcurve.LC(x=np.array([0.0, 10.0]), y=None, yerr=None)
    assert cost_fcn.cost(lca, lcb) == 0

    lcb = lightcurve.LC(x=np.array([100.0, 200.0]), y=None, yerr=None)
    assert cost_fcn.cost(lca, lcb) == 1

    lcb = lightcurve.LC(x=np.array([5.0, 15.0]), y=None, yerr=None)
    assert cost_fcn.cost(lca, lcb) == 0.1
