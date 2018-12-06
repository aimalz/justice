# -*- coding: utf-8 -*-
"""Tests for vector similarity helpers."""
import tensorflow as tf

from justice.align_model import vector_similarity


def test_similarity(tf_sess):
    a = tf.constant([[1, 2, 0], [-0.1, -0.2, -0.3]], tf.float32)
    b = tf.constant([[1, 2, 0], [-0.1, -0.2, -0.3]], tf.float32)
    sim = vector_similarity.Similarity(a, b, batch_size=2, hidden_size=3)
    loss_correct = tf_sess.run(sim.loss(tf.constant([True, True])))
    loss_false_positive = tf_sess.run(sim.loss(tf.constant([False, True])))
    aux_loss = tf_sess.run(sim.aux_loss())
    assert abs(loss_correct) < 1e-7, "Should be near-correct."
    assert abs(loss_false_positive - 7.82) < 0.1, "Should have high loss."
    assert abs(aux_loss - 0.96) < 0.1, "Should have some nonzero aux loss."

    a = tf.constant([[0.4472136, 0.8944272, 0.], [-0.1, -0.2, -0.3]], tf.float32)
    b = tf.constant([[-0.5345224, 0.2672612, -0.8017837],
                     [-0.1, -0.2, -0.3]], tf.float32)
    sim = vector_similarity.Similarity(a, b, batch_size=2, hidden_size=3)
    loss_correct = tf_sess.run(sim.loss(tf.constant([False, True])))
    loss_false_negative = tf_sess.run(sim.loss(tf.constant([True, True])))
    assert abs(loss_correct) < 1e-7
    assert abs(loss_false_negative - 8.06) < 0.1
