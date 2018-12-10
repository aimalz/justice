# -*- coding: utf-8 -*-
"""Helpers for building TF datasets."""
import numpy as np
import tensorflow as tf


def auto_dtype(key, value):
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        return tf.float32
    elif isinstance(value, (float, np.floating)):
        return tf.float32
    elif isinstance(value, (int, np.integer)):
        return tf.int64
    elif isinstance(value, (bool, np.bool_)):
        return tf.bool
    elif '_padding' in key:
        assert isinstance(
            value, int
        ) or (isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer))
        return tf.int32
    else:
        typ = type(value.dtype)
        raise ValueError(
            f"Unrecognized data type for key {key!r}, value {value.dtype!r} ({typ})"
        )


def auto_shape(value):
    if isinstance(value, np.ndarray):
        return value.shape
    else:
        return ()


def dataset_from_generator_auto_dtypes(generator):
    print("dataset_from_generator_auto_dtypes")
    first_features = next(generator)
    dtypes = {key: auto_dtype(key, value) for key, value in first_features.items()}
    shapes = {key: auto_shape(value) for key, value in first_features.items()}

    def gen():
        yield first_features
        yield from generator

    return tf.data.Dataset.from_generator(gen, output_types=dtypes, output_shapes=shapes)
