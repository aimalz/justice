# -*- coding: utf-8 -*-
"""Tests period distribution code."""
from justice import simulate
from justice.features import period_distribution


def test_different_period_gauss_data():
    faster_lc = simulate.TestLC.make_realistic_gauss(15.0)
    slower_lc = simulate.TestLC.make_realistic_gauss(30.0)

    period_distribution.IndependentLs().apply(faster_lc)
    period_distribution.IndependentLs().apply(slower_lc)

    # TODO(gatoatigrado): Compare max point of two distributions.
    # However, I'm not getting sensible results from LS yet (probably bad frequency
    # scale initialization).
