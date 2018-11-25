# -*- coding: utf-8 -*-
r"""Basic vector similarity model.

Models alignment between to light curves lc_a and lc_b at points x_a \in lc_a and
x_b \in lc_b. A non-linear transformation is applied to features for both light
curves in those desired regions, and then the score of the merge will be the dot
product between these outputs.

This is sometimes called a "dual encoder" or "two tower" model.
"""
import tensorflow as tf

from justice.align_model import vector_similarity
from justice.align_model.lr_prefixing import lr_per_side_sub_model_fn
from justice.features import dense_extracted_features


def model_fn(features, labels, mode, params):
    del labels  # unused
    batch_size = params["batch_size"]
    dropout_rate = params.get("dropout_rate", 0.1)
    output_size = params.get("output_size", 64)

    def fc_layer(hidden_size, index):
        return [
            tf.keras.layers.Dropout(dropout_rate, name=f"layer_{index}_dropout"),
            tf.keras.layers.Dense(hidden_size, name=f"layer_{index}_dense"),
            tf.keras.layers.Activation("relu", name=f"layer_{index}_relu"),
        ]

    fc_layers = [
        layer for i, hidden_size in enumerate(params.get("hidden_sizes", [128]))
        for layer in fc_layer(hidden_size, i)
    ]
    per_side_model_keras = tf.keras.models.Sequential(
        fc_layers + [
            tf.keras.layers.Dense(output_size, name="output_dense"),
        ],
        name="per_side_model"
    )

    def per_side_model(side_features, params):
        inputs = dense_extracted_features.feature_model_fn(side_features, params=params)
        batch_size, twice_window_size, channels, embedding_size, nbands = map(
            int, inputs.shape
        )
        curr_layer = tf.reshape(inputs, [batch_size, -1])
        return per_side_model_keras(curr_layer)

    left, right = lr_per_side_sub_model_fn(per_side_model, features, params=params)
    sim = vector_similarity.Similarity(
        left, right, batch_size=batch_size, hidden_size=output_size
    )

    loss = None
    aux_loss_factor = params.get("aux_loss_factor", 0.1)
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = features["labels"]
        loss = sim.loss(labels) + aux_loss_factor * sim.aux_loss()

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.contrib.opt.NadamOptimizer()
        train_op = optimizer.minimize(loss)

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions={'score': sim.score()}, loss=loss, train_op=train_op
    )
