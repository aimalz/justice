# -*- coding: utf-8 -*-
"""Gets training data from the sample data file."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

from justice.datasets import sample_data

if tuple(map(int, tf.__version__.split('.')))[0:2] < (1, 9):
    raise ImportError(
        "TensorFlow 1.9 or above is required, found {}".format(tf.__version__)
    )


def _prefix_fcn(diffs_within_median):
    num_elts, = diffs_within_median.shape
    result = np.zeros((num_elts + 1, ), dtype=np.int32)
    accumulator = 0
    for i, value in enumerate(diffs_within_median):
        if value:
            accumulator += 1
        else:
            accumulator = 0
        result[i + 1] = accumulator
    return result


def get_window_valid_indices(diffs, lower, upper, window_size):
    prefix_within_median = _prefix_fcn((lower < diffs) & (diffs < upper))
    assert prefix_within_median.shape[0] == diffs.shape[0] + 1

    # This is a bit tricky. For window size 1, the index range starts at 0 and goes to |prefix_within_median|.
    # All points are valid since they are all ≥ 0. For window size 2, the range starts at -1 and goes to
    # |prefix_within_median| - 1. The first index is never chosen, so we won't end up with negative indices.
    #
    # Basically elements of |prefix_within_median| that are greater than the threshold are at the right end
    # of the window, so we have to shift them over.
    return np.arange(1 - window_size, prefix_within_median.shape[0] + 1 -
                     window_size)[prefix_within_median >= window_size - 1]


def sample_data_input_fn(params):
    """Extracts windowed samples from sample data.

    Since our sample data set has semi-frequent samples, we just pretend that
    the time scale is fixed for those that are "close enough", and create
    windowed sets of data. For example, if our time points are,

    [0, 1, 2, 3, 4, 10, 11, 12, 12, 13, 14]

    then the `time_diffs` for this sample would be

    [   1, 1, 1, 1,  6,  1,  1,  0,  1,  1]

    using a cutoff of 0.8 and 1.2 for whether time diffs represent a contiguous
    segment, we'll now arrive at partial sums of contiguous segments
    (`prefix_within_median`) as,

    [   1, 2, 3, 4,  0,  1,  2,  0,  1,  2]

    If we wanted windows of size 3, we could extract 3 windows from the first
    segment, where "+"s represent items in the window,

    [0, 1, 2, 3, 4, 10, 11, 12, 12, 13, 14]    # Original array

    [+, +, +,                             ]
    [   +, +, +,                          ]
    [      +, +, +                        ]

    and 1 window from both the second and third segment.
    """
    window_size = params['window_size']
    batch_size = params['batch_size']

    dataset_names = sample_data.get_data_names()
    all_downsampled = [sample_data.get_downsampled_data(name) for name in dataset_names]
    np_dtype = all_downsampled[0].dtype
    _, num_columns = all_downsampled[0].shape
    assert num_columns == 3

    # For each data item, this computes
    time_diffs = [(x[1:, 0] - x[:-1, 0]) for x in all_downsampled]
    median_time_diff = np.median(np.concatenate(time_diffs, axis=0))
    lower, upper = median_time_diff * 0.8, median_time_diff * 1.2
    valid_start_window_indices = [
        get_window_valid_indices(d, lower, upper, window_size) for d in time_diffs
    ]
    for name, valid_indices in zip(dataset_names, valid_start_window_indices):
        if np.size(valid_indices) == 0:
            raise ValueError("{} has no valid window ranges".format(name))

    def get_samples_py_op(idx_array):
        assert isinstance(idx_array, np.ndarray)
        assert idx_array.shape == (batch_size, )
        samp_results = np.zeros((batch_size, window_size, num_columns), dtype=np_dtype)
        for i, sample_idx in enumerate(idx_array):
            start_idx = random.choice(valid_start_window_indices[sample_idx])
            samp_results[i, :, :] = all_downsampled[sample_idx][start_idx: (
                start_idx + window_size)]
        assert samp_results.shape == (batch_size, window_size, num_columns)
        return samp_results

    def get_window_sample(idx_tensor):
        samples = tf.py_func(get_samples_py_op, [idx_tensor], np_dtype)
        samples.set_shape((batch_size, window_size, num_columns))
        return samples

    def random_negative_py_op(idx_array):
        assert isinstance(idx_array, np.ndarray)
        neg_idx_array = np.copy(idx_array)
        for i, idx in enumerate(idx_array):
            while neg_idx_array[i] == idx_array[i]:
                neg_idx_array[i] = random.randint(0, len(all_downsampled) - 1)
        return neg_idx_array

    def get_negative_window_sample(idx_tensor):
        neg_idx_tensor = tf.py_func(
            random_negative_py_op,
            [idx_tensor],
            idx_tensor.dtype)
        return get_window_sample(neg_idx_tensor)

    # Current sample method: First select sample index, then select window.
    num_samples = len(all_downsampled)
    if num_samples < 2:
        raise ValueError("Need at least 2 light curves for negative samples!")
    dataset = tf.data.Dataset.range(num_samples)
    dataset = dataset.repeat().shuffle(num_samples * 2).batch(batch_size)

    positive = dataset.map(lambda idx_tensor: {
        'left': get_window_sample(idx_tensor),
        'right': get_window_sample(idx_tensor),
        'goal': tf.constant([1.0] * batch_size, dtype=tf.float64)
    })
    negative = dataset.map(lambda idx_tensor: {
        'left': get_window_sample(idx_tensor),
        'right': get_negative_window_sample(idx_tensor),
        'goal': tf.constant([0.0] * batch_size, dtype=tf.float64)
    })

    # TODO(gatoatigrado): Experiment with shuffling positive & negative within a batch.
    # Currently each batch is just positive or negative.
    assert positive.output_shapes == negative.output_shapes
    assert negative.output_types == positive.output_types
    dataset = tf.contrib.data.sample_from_datasets((positive, negative))
    assert dataset.output_shapes == negative.output_shapes
    return dataset


def main():
    dataset = sample_data_input_fn({
        'window_size': 5,
        'batch_size': 8,
    })
    with tf.Session() as sess:
        tensors = dataset.make_one_shot_iterator().get_next()
        for _ in range(10):
            results = sess.run(tensors)
            print(results['goal'])


if __name__ == '__main__':
    main()
