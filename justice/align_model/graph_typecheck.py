# -*- coding: utf-8 -*-
"""Helper function for constructing TensorFlow graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import typing

import tensorflow as tf


def assert_shape(tensor, shape):
    """Checks that a tensor shape is as expected.

    :param tensor: Tensor.
    :param shape: List of expected dimensions.
    :return: `tensor`, for convenience.
    """
    assert isinstance(shape, list) and all(isinstance(x, int) for x in shape)
    tensor_shape = list(map(int, tensor.shape))
    if tensor_shape != shape:
        raise ValueError(f"Expected tensor shape {shape} but got {tensor_shape}.")
    return tensor


def print_single(tensor, message):
    return tf.Print(tensor, [tensor], message=message, summarize=int(1e5))


def assert_tensor_dict(tensor_dict: typing.Dict[str, tf.Tensor]
                       ) -> typing.Dict[str, tf.Tensor]:
    for k, v in tensor_dict.items():
        if not isinstance(k, str):
            raise ValueError(f"Expected dict key to be string, got {k!r}")
        if not isinstance(v, tf.Tensor):
            raise ValueError(f"Expected dict value to be a tensor, got {v!r}")
    return tensor_dict
