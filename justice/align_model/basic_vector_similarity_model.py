# -*- coding: utf-8 -*-
r"""Basic vector similarity model.

Models alignment between to light curves lc_a and lc_b at points x_a \in lc_a and
x_b \in lc_b. A non-linear transformation is applied to features for both light
curves in those desired regions, and then the score of the merge will be the dot
product between these outputs.

This is sometimes called a "dual encoder" or "two tower" model.
"""
import tensorflow as tf

from justice.align_model import vector_similarity, graph_typecheck
from justice.align_model.lr_prefixing import lr_per_side_sub_model_fn
from justice.features import dense_extracted_features


def model_fn(features, labels, mode, params):
    del labels  # unused
    batch_size = params["batch_size"]
    dropout_rate = params.get("dropout_rate", 0.1)
    output_size = params.get("output_size", 64)
    twice_window_size = 2 * params["window_size"]
    embedding_size = 32
    channels = 3
    nbands = len(params["lc_bands"])

    def fc_layer(hidden_size, index):
        return [
            tf.keras.layers.Dropout(dropout_rate, name=f"layer_{index}_dropout"),
            tf.keras.layers.Dense(hidden_size, name=f"layer_{index}_dense"),
            tf.keras.layers.Activation("elu", name=f"layer_{index}_elu"),
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
    input_conv = tf.keras.layers.Convolution2D(
        filters=128,
        kernel_size=(3, 5),
        input_shape=(twice_window_size, nbands, channels * embedding_size)
    )

    def per_side_model(side_features, params):
        inputs = dense_extracted_features.feature_model_fn(side_features, params=params)
        # Channels is dflux/dt, etc.
        graph_typecheck.assert_shape(
            inputs, [batch_size, twice_window_size, channels, embedding_size, nbands]
        )
        if params["first_layer"] == "conv":
            inputs_shuffled = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])
            graph_typecheck.assert_shape(
                inputs_shuffled,
                [batch_size, twice_window_size, nbands, channels, embedding_size]
            )
            conv_result = input_conv(
                tf.reshape(inputs_shuffled, [batch_size, twice_window_size, nbands, -1])
            )
            curr_layer = tf.reshape(conv_result, [batch_size, -1])
        elif params["first_layer"] == "none":
            curr_layer = tf.reshape(inputs, [batch_size, -1])
        else:
            raise ValueError(f"No known first_layer type {params['first_layer']}")

        return per_side_model_keras(curr_layer)

    left, right = lr_per_side_sub_model_fn(per_side_model, features, params=params)
    sim = vector_similarity.Similarity(
        left, right, batch_size=batch_size, hidden_size=output_size
    )

    loss = None
    aux_loss_factor = params.get("aux_loss_factor", 0.1)
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = features["labels"]
        sim_loss = sim.loss(labels)
        loss = sim_loss + aux_loss_factor * sim.aux_loss()

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = params.get("learning_rate", 1e-3)
        clip_norm = params.get("clip_norm", 1.0)

        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate)
        grads_and_vars = [
            gv for gv in optimizer.compute_gradients(loss) if gv[0] is not None
        ]
        grads, variables = zip(*grads_and_vars)
        grads = [tf.clip_by_norm(t, clip_norm=clip_norm) for t in grads]
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.apply_gradients(
            zip(grads, variables), global_step=global_step
        )

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions={'score': sim.score()}, loss=loss, train_op=train_op
    )
