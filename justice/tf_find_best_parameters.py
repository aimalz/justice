# -*- coding: utf-8 -*-
"""Uses tensorflow/sgd to find the best parameters, given a parameterizable model and data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple

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

    def __init__(self, period_init, phase_init, amp_init, const_init):
        self.period_init = period_init
        self.phase_init = phase_init
        self.amp_init = amp_init
        self.const_init = const_init
        self.period = tf.get_variable("period", dtype=tf.float64,
                                      initializer=period_init)
        self.phase = tf.get_variable("phase", dtype=tf.float64,
                                     initializer=phase_init)
        self.amp = tf.get_variable("amplitude", dtype=tf.float64,
                                   initializer=amp_init)
        self.const_factor = tf.get_variable("const", dtype=tf.float64,
                                            initializer=const_init)

    def parameter_values(self, session):
        return {
            'period': session.run(self.period),
            'phase': session.run(self.phase),
            'amp': session.run(self.amp),
            'const': session.run(self.const_factor)
        }

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

    def get_params(self, lc_with_err, learning_rate=1e-2, steps=1000):
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

                # Initialize and solve.
                sess.run(tf.global_variables_initializer())

                # print("After init")
                # print(model.parameter_values(sess))
                # print("logprob", sess.run(mean_log_prob))
                # print("logprob vector", sess.run(log_probs))

                for _ in xrange(steps):
                    sess.run(train_op)

                # print("After training")
                # print(model.parameter_values(sess))
                # print("logprob", sess.run(mean_log_prob))
                # print("logprob vector", sess.run(log_probs))

                final_logprobs = sess.run(log_probs)
                return final_logprobs, model.parameter_values(sess)
