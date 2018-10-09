# -*- coding: utf-8 -*-
"""Tests frequency distribution code."""
from justice import simulate
from justice.features import frequency_distribution


def test_different_period_gauss_data():
    faster_lc = simulate.TestLC.make_realistic_gauss(15.0)
    slower_lc = simulate.TestLC.make_realistic_gauss(30.0)

    frequency_distribution.IndependentLs().transform(faster_lc)
    frequency_distribution.IndependentLs().transform(slower_lc)

    # TODO(gatoatigrado): Compare max point of two distributions.
    # However, I'm not getting sensible results from LS yet (probably bad frequency
    # scale initialization).
