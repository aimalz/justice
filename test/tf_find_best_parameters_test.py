# -*- coding: utf-8 -*-
"""Tests for TensorFlow parameter stuff."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf

from justice import mock_data, tf_find_best_parameters


@pytest.yield_fixture(autouse=True, name="graph")
def graph():
    with tf.Graph().as_default() as g:
        yield g


def make_sine_model():
    flt_param = lambda x, y: tf.random_uniform(
        (), x, y, tf.float64)
    return tf_find_best_parameters.SineModel(
        period_init=flt_param(-10, 10),
        phase_init=flt_param(0, 2 * np.pi),
        amp_init=flt_param(0, 10),
        const_init=flt_param(0, 10)
    )


def test_sine_model(graph):
    actual_period = 3.0
    actual_phase = 0.0
    actual_amplitude = 1.0
    actual_const = 0.0

    def_cadence = np.arange(0., 200., 5.)
    times = mock_data.make_cadence(def_cadence, 0.5)
    pure = mock_data.make_sine(
        period=actual_period,
        phase=actual_phase,
        amp=actual_amplitude,
        const=actual_const)(times)
    actual, error = mock_data.noisify_obs(pure, 0.1)
    lc = tf_find_best_parameters.LCWithErr(times, actual, error)

    # Make sure the model can be constructed and such (in `graph`).
    model = make_sine_model()
    probs = model.log_probs(lc)
    assert times.shape == probs.shape

    # Actually run the solver.
    solver = tf_find_best_parameters.Solver(make_sine_model)
    logprobs, params = solver.get_params(lc, learning_rate=0.1, steps=1000)
    print("Final logprob", sum(logprobs))
    print("Final parameters", params)
