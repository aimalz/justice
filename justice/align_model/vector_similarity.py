# -*- coding: utf-8 -*-
"""Helpers for computing cost-weighted vector similarity."""
import tensorflow as tf

from justice.align_model import graph_typecheck


class Similarity:
    def __init__(
        self,
        left: tf.Tensor,
        right: tf.Tensor,
        *,
        batch_size: int,
        hidden_size: int,
        norm: bool = True
    ):
        graph_typecheck.assert_shape(left, [batch_size, hidden_size])
        graph_typecheck.assert_shape(right, [batch_size, hidden_size])
        self.left = left
        self.right = right
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.norm = norm

    def _normalize(self, tensor):
        if self.norm:
            return tensor / tf.maximum(
                1e-6, tf.norm(tensor, ord=2, axis=1, keepdims=True)
            )
        else:
            return tensor

    def loss(self, labels: tf.Tensor, score: tf.Tensor = None):
        """Computes the log loss of the scores.

        :param labels: Labels tensor
        :param score: Result from `self.score()`, if reusing such computation is desired.
        :return: Tensor with the loss.
        """
        graph_typecheck.assert_shape(labels, [self.batch_size])
        score = self.score() if score is None else score
        return tf.losses.log_loss(labels=labels, predictions=score, epsilon=1e-5)

    def score(self):
        left = self._normalize(self.left)
        right = self._normalize(self.right)
        dotprod = tf.reduce_sum(left * right, axis=1)
        graph_typecheck.assert_shape(dotprod, [self.batch_size])
        # Normalize output to between 0 and 1.
        result = 0.5 * (dotprod + 1.0)
        graph_typecheck.assert_shape(result, [self.batch_size])
        return result

    def predictions(self, score: tf.Tensor = None):
        score = self.score() if score is None else score
        return score > 0.5

    def per_batch_accuracy(self, labels: tf.Tensor, predictions: tf.Tensor):
        """Returns immediate accuracy, not averaging over batches."""
        if labels.dtype != predictions.dtype:
            raise TypeError(
                "Labels has dtype {} but predictions has dtype {}"
                .format(labels.dtype, predictions.dtype)
            )
        is_correct = tf.cast(tf.equal(predictions, labels), tf.float32)
        return tf.reduce_mean(is_correct)

    def aux_loss(self):
        if self.norm:
            return tf.losses.mean_squared_error(
                labels=tf.ones([2 * self.batch_size]),
                predictions=tf.concat([
                    tf.norm(self.left, ord=2, axis=1),
                    tf.norm(self.right, ord=2, axis=1),
                ],
                    axis=0),
            )
        else:
            return tf.zeros([])
