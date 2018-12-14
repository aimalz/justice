# -*- coding: utf-8 -*-
"""Tests for synthetic positives generation."""
import random

import numpy as np
import tensorflow as tf
from scipy.stats import norm

from justice import simulate
from justice.align_model import synthetic_positives


def test_xform_params_distribution():
    pg = synthetic_positives.BasicPositivesGenerator(
        dilate_time_stdev_factor=2.0,
        dilate_flux_stdev_factor=3.0,
        rng=random.Random(1234),
        max_time_dilation = 6,
        max_flux_dilation = 16
    )
    xforms = [pg.make_xform() for _ in range(1000)]
    time_dilations = [xf._dilate_time for xf in xforms]
    flux_dilations = [xf._dilate_flux for xf in xforms]

    # Hard rules - should never be violated
    assert 1/6. < min(time_dilations)
    assert max(time_dilations) < 6
    assert 1/16. < min(flux_dilations)
    assert max(flux_dilations) < 16

    # These *could* be violated by chance. If one fails, graph some distributions,
    # manually check sanity, and adjust bounds if it looks good.
    assert abs(np.median(time_dilations) - 1.0) < 0.06
    assert abs(np.median(flux_dilations) - 1.0) < 0.06

    def adjusted_percentile(max_dilation):
        # need to rebalance the CDF since we are clipping the tails of the distribution
        n_stds_clipped = -np.log(max_dilation)
        return 100 * (norm.cdf(1) - norm.cdf(n_stds_clipped)) / (1 - 2 * norm.cdf(n_stds_clipped))

    first_percentile_time_dilations = np.percentile(time_dilations, adjusted_percentile(6/2.0))
    first_percentile_flux_dilations = np.percentile(flux_dilations, adjusted_percentile(16/3.0))
    assert abs(first_percentile_time_dilations - 2) < 0.1
    assert abs(first_percentile_flux_dilations - 3) < 0.14


def _total_points(fpp: synthetic_positives.PositivePairSubsampler):
    return fpp.lca.total_points_all_bands(), fpp.lcb.total_points_all_bands()


def test_smoke_subsampler():
    pg = synthetic_positives.BasicPositivesGenerator(
        dilate_time_stdev_factor=2.0,
        dilate_flux_stdev_factor=3.0,
        rng=random.Random(1234)
    )
    pair_subsampler = synthetic_positives.PositivePairSubsampler(
        rng=np.random.RandomState(123)
    )
    fpp = pg.make_positive_pair(simulate.TestLC.make_realistic_gauss())
    fpp2 = pair_subsampler.apply(fpp)
    assert _total_points(fpp) == (115, 115)
    assert _total_points(fpp2) == (104, 78)


def test_feature_extraction(tf_sess):
    pg = synthetic_positives.BasicPositivesGenerator(
        dilate_time_stdev_factor=2.0,
        dilate_flux_stdev_factor=3.0,
        rng=random.Random(1234)
    )
    params = {"batch_size": 20, "window_size": 10, "lc_bands": ["b"]}
    rv_fex = synthetic_positives.RawValuesFullPositives.from_params(params)
    lcs = [simulate.TestLC.make_realistic_gauss(scale) for scale in [10.0, 15.0, 30.0]]
    fpp_gen = (pg.make_positive_pair(lc) for lc in lcs)
    dataset = rv_fex.make_dataset(fpp_gen)
    assert isinstance(dataset, tf.data.Dataset)
    get_next_tensor = dataset.make_one_shot_iterator().get_next()
    values = []
    while True:
        try:
            values.append(tf_sess.run(get_next_tensor))
        except tf.errors.OutOfRangeError:
            break
    assert len(values) == 3
