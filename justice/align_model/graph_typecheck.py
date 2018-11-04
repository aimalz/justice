# -*- coding: utf-8 -*-
"""Helper function for constructing TensorFlow graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    # else:
    #     print("Shape checks out!", shape)
    return tensor


def print_single(tensor, message):
    return tf.Print(tensor, [tensor], message=message, summarize=int(1e5))
