# -*- coding: utf-8 -*-
"""tf.estimator.Estimator model function for the similarity model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers


def extract_vector(window_data, params, name):
    batch_size = params["batch_size"]
    window_size = params["window_size"]
    dropout_rate = params["dropout_keep_prob"]
    symmetric = params.get("symmetric", True)
    layer1_dim, layer2_dim = params["layer_sizes"]

    assert window_data.shape == (batch_size, window_size, 3)

    # <float64>: [batch_size, window_size]
    flux_values = window_data[:, :, 1]

    # <float64>: [batch_size, window_size - 1]
    diffs = flux_values[:, 1:] - flux_values[:, :-1]

    reuse = tf.AUTO_REUSE if symmetric else False
    scope_name = "extract_vector" if symmetric else "extract_vector_{}".format(name)
    with tf.variable_scope(scope_name, reuse=reuse):
        layer1 = layers.fully_connected(diffs, layer1_dim, activation_fn=tf.nn.relu)
        layer1 = tf.nn.dropout(layer1, keep_prob=dropout_rate)
        layer2 = layers.fully_connected(layer1, layer2_dim, activation_fn=tf.nn.relu)

    # Normalize the vectors. If vector magnitude is causing precision issues, we could
    # add a regularization loss.
    layer2_norm = tf.expand_dims(tf.norm(layer2, axis=1), axis=1)
    assert layer2_norm.shape == (batch_size, 1)

    # Display mean norm (across batch) in TensorBoard.
    tf.summary.scalar("{}_norm".format(name), tf.reduce_mean(layer2_norm))

    tf.summary.scalar("{}_layer2_min".format(name), tf.reduce_min(layer2))

    return layer2 / layer2_norm


def model_fn(features, labels, mode, params):
    left, right = features['left'], features['right']

    left_vec = extract_vector(left, params, "left")
    right_vec = extract_vector(right, params, "right")
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
