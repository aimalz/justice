# -*- coding: utf-8 -*-
"""Tests helper functions for TF dataset builder."""
import tensorflow as tf

from justice.features import tf_dataset_builder


def test_auto_dtype():
    assert tf_dataset_builder.auto_dtype(NotImplemented, True) is tf.bool
    assert tf_dataset_builder.auto_dtype(NotImplemented, 1234) is tf.int64
