# -*- coding: utf-8 -*-
"""tf.estimator.Estimator model function for the similarity model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers


def extract_vector(window_data, params):
    batch_size = params["batch_size"]
    window_size = params["window_size"]
    dropout_rate = params["dropout_keep_prob"]
    layer1_dim, layer2_dim = params["layer_sizes"]

    assert window_data.shape == (batch_size, window_size, 3)

    # <float64>: [batch_size, window_size]
    flux_values = window_data[:, :, 1]

    # <float64>: [batch_size, window_size - 1]
    diffs = flux_values[:, 1:] - flux_values[:, :-1]

    layer1 = layers.fully_connected(diffs, layer1_dim, activation_fn=tf.nn.relu)
    layer1 = tf.nn.dropout(layer1, keep_prob=dropout_rate)
    layer2 = layers.fully_connected(layer1, layer2_dim, activation_fn=tf.nn.relu)
    layer2_norm = tf.expand_dims(tf.norm(layer2, axis=1), dim=1)
    assert layer2_norm.shape == (batch_size, 1)
    return layer2 / layer2_norm


def model_fn(features, labels, mode, params):
    left, right = features['left'], features['right']

    left_vec, right_vec = extract_vector(left, params), extract_vector(right, params)
    predictions = tf.reduce_sum(left_vec * right_vec, axis=1)

    loss = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.losses.mean_squared_error(
            labels=features['goal'], predictions=predictions)

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=1e-3,
        )
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
    )
