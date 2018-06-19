# -*- coding: utf-8 -*-
"""Uses tensorflow/sgd to find the best parameters, given a parameterizable model and data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple

import numpy as np
import tensorflow as tf

LCWithErr = namedtuple("LCWithErr", ["times", "values", "error"])


class Model(object):
    """Functions related to generating a model.

    Generally any parameters should be declared (with tf.get_variable) in the init phase.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def parameter_values(self, session):
        """Gets solved parameter values.

        :param session: tf.Session
        :return: Dict of parameter values.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log_probs(self, lc_with_err):
        """Log loss for the current model.

        :param lc_with_err: Light curve with error.
        :return: 1d logprob tensor of predicted points given data, assuming any variance
            (error) in data is independent.
        """
        raise NotImplementedError()

    def regularizer_cost(self, step_fraction):
        return tf.constant(0.0, dtype=tf.float64)


class SineModel(Model):
    """Sine wave model."""

    def __init__(self, lc_with_err, period_init, phase_init, amp_init):
        self.period_init = period_init
        self.phase_init = phase_init
        self.amp_init = amp_init
        self.const_init = tf.constant(np.mean(lc_with_err.values))
        self.period = tf.get_variable("period", dtype=tf.float64,
                                      initializer=period_init)
        self.phase = tf.get_variable("phase", dtype=tf.float64,
                                     initializer=phase_init)
        self.amp = tf.get_variable("amplitude", dtype=tf.float64,
                                   initializer=amp_init)
        self.const_factor = tf.get_variable("const", dtype=tf.float64,
                                            initializer=self.const_init)

    def parameter_values(self, session, normalize=True):
        amp, period, phase = session.run([self.amp, self.period, self.phase])
        if normalize:
            if amp < 0:
                amp = -amp
                period = -period
                phase = -phase
            if period < 0:
                period = -period
                phase = np.pi - phase
            while phase < 0:
                phase += 2 * np.pi
            while phase > 2 * np.pi:
                phase -= 2 * np.pi
        return {
            'period': period,
            'phase': phase,
            'amp': amp,
            'const': session.run(self.const_factor)
        }

    def format_params(self, params):
        return (
            "{amp} * sin({period} * x + {phase}) + {const}".format(**params))

    def log_probs(self, lc_with_err):
        predicted = (
                self.amp * tf.sin(self.period * lc_with_err.times + self.phase) +
                self.const_factor)
        error_dist = tf.distributions.Normal(
            loc=lc_with_err.values, scale=lc_with_err.error, validate_args=True)
        return error_dist.log_prob(predicted)


class Solver(object):

    def __init__(self, model_fn):
        """Initializes a solver.

        :param model_fn: Function () -> Model.
        :return:
        """
        self.model_fn = model_fn

    def get_params(self, lc_with_err, learning_rate=1e-2, steps=1000,
                   random_restarts=1):
        """Gets matching model parameters.

        :param lc_with_err: Light curve.
        :type lc_with_err: LCWithErr
        :param steps: Number of steps to run solver for.
        :return: Optimized parameters.
        """
        with tf.Graph().as_default() as g:
            model = self.model_fn()
            with tf.Session() as sess:
                global_step = tf.train.create_global_step()
                step_fraction = tf.cast(global_step, tf.float64) / float(steps)

                # Set up loss.
                log_probs = model.log_probs(lc_with_err)
                mean_log_prob = tf.reduce_mean(log_probs)
                loss = (-mean_log_prob) + model.regularizer_cost(step_fraction)

                # Set up solver.
                learning_rate_tensor = tf.train.exponential_decay(
                    learning_rate, global_step, steps, 0.1, staircase=False)
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate_tensor)
                train_op = opt.minimize(-mean_log_prob, global_step=global_step)

                global_var_init = tf.global_variables_initializer()

                def run_random_restart():
                    sess.run(global_var_init)

                    # print("After init")
                    # print(model.parameter_values(sess))
                    # print("logprob", sess.run(mean_log_prob))
                    # print("logprob sum", sum(sess.run(log_probs)))
                    # print("Step", sess.run(global_step))

                    for _ in xrange(steps):
                        sess.run(train_op)

                    # print("After training")
                    # print(model.parameter_values(sess))
                    # print("logprob", sess.run(mean_log_prob))
                    # print("logprob sum", sum(sess.run(log_probs)))
                    final_logprobs = sess.run(log_probs)
                    params = model.parameter_values(sess)
                    return sess.run(loss), final_logprobs, params

                best_score, best_probs, best_params = min(
                    run_random_restart() for _ in xrange(random_restarts))
                return best_probs, best_params
