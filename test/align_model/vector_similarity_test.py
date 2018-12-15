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
    assert abs(loss_correct) < 1e-4, "Should be near-correct."
    assert abs(loss_false_positive - 5.75) < 0.1, "Should have high loss."
    assert abs(aux_loss - 0.96) < 0.1, "Should have some nonzero aux loss."

    a = tf.constant([[0.4472136, 0.8944272, 0.], [-0.1, -0.2, -0.3]], tf.float32)
    b = tf.constant([[-0.4472136, -0.8944272, 0.], [-0.1, -0.2, -0.3]], tf.float32)
    sim = vector_similarity.Similarity(a, b, batch_size=2, hidden_size=3)
    loss_correct = tf_sess.run(sim.loss(tf.constant([False, True])))
    loss_false_negative = tf_sess.run(sim.loss(tf.constant([True, True])))
    assert abs(loss_correct) < 1e-4
    assert abs(loss_false_negative - 5.76) < 0.1


def test_similarity_fuzz_test(tf_sess):
    batch_size = 32
    dim = 8
    a = tf.random.uniform([batch_size, dim], minval=-100, maxval=100, name="a")
    b = tf.random.uniform([batch_size, dim], minval=-100, maxval=100, name="b")
    labels = tf.random.uniform([batch_size],
                               minval=0,
                               maxval=1,
                               dtype=tf.int32,
                               name="labels")
    sim = vector_similarity.Similarity(
        a, b, batch_size=batch_size, hidden_size=dim, norm=True
    )
    loss = sim.loss(labels)
    for _ in range(10):
        assert 0 < tf_sess.run(loss) < 10.0
