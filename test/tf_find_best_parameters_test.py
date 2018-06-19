# -*- coding: utf-8 -*-
"""Tests for TensorFlow parameter stuff."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from justice import mock_data, tf_find_best_parameters


@pytest.yield_fixture(autouse=True, name="graph")
def graph():
    """Try to make sure data doesn't leak between tests."""
    with tf.Graph().as_default() as g:
        yield g


def make_sine_model(lc_with_err):
    flt_param = lambda x, y: tf.random_uniform(
        (), x, y, tf.float64)
    return tf_find_best_parameters.SineModel(
        lc_with_err=lc_with_err,
        period_init=flt_param(0.1, 4),
        phase_init=flt_param(0, 2 * np.pi),
        amp_init=flt_param(0, 10)
    )


def test_sine_model(graph):
    actual_period = 3.0
    actual_phase = 0.0
    actual_amplitude = 1.0
    actual_const = 1.0

    # This means that "nailing" the point results in a PDF value of about 1.
    err_standard_deviation = 0.147

    def_cadence = np.arange(0., 200., 5.)
    times = mock_data.make_cadence(def_cadence, 0.5)
    pure = mock_data.make_sine(
        period=actual_period,
        phase=actual_phase,
        amp=actual_amplitude,
        const=actual_const)(times)
    actual, error = mock_data.noisify_obs(pure, err_standard_deviation)
    lc = tf_find_best_parameters.LCWithErr(times, actual, error)

    # Make sure the model can be constructed and such (in `graph`).
    model = make_sine_model(lc)
    probs = model.log_probs(lc)
    assert times.shape == probs.shape

    # Actually run the solver.
    solver = tf_find_best_parameters.Solver(
        lambda: make_sine_model(lc))
    logprobs, params = solver.get_params(lc, learning_rate=0.1, steps=1000)
    params_str = (
        "{amp} * sin({period} * x + {phase}) + {const}".format(**params))
    print("Final logprob", sum(logprobs))
    print("Final parameters", params_str)
