# -*- coding: utf-8 -*-
"""TF dataset for evaluating each point."""
import collections
import typing

import numpy as np
import tensorflow as tf
from justice import lightcurve


def auto_dtype(key, value):
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        return tf.float32
    elif isinstance(value, (float, np.floating)):
        return tf.float32
    elif isinstance(value, (int, np.integer)):
        return tf.int64
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


DatasetWithLength = collections.namedtuple(
    "DatasetWithLength", ["dataset", "num_batches", "num_non_padding"]
)


def unbatch(predictions, num_batches, overflow):
    for i, batch in enumerate(predictions):
        if i == num_batches - 1 and overflow != 0:
            yield from batch[:-overflow]
        else:
            yield from batch


class PerPointDatasetGenerator(object):
    def __init__(self, extract_fcn, batch_size):
        self.extract_fcn = extract_fcn
        self.batch_size: int = batch_size
        assert isinstance(batch_size, int)

    def make_dataset(self, lc: lightcurve._LC):
        """Inefficient function to make a dataset."""
        times = np.unique(np.concatenate(lc.all_times(), axis=0))
        first_features = dict(self.extract_fcn(lc, times[0]), time=times[0])
        dtypes = {key: auto_dtype(key, value) for key, value in first_features.items()}
        shapes = {key: auto_shape(value) for key, value in first_features.items()}
        overflow = (-len(times)) % self.batch_size

        def gen():
            for time in times:
                extracted_dict = self.extract_fcn(lc, time)
                extracted_dict['time'] = time
                yield extracted_dict
            for _ in range(overflow):
                yield first_features

        # Sanity checks, delete in a bit for cleanliness.
        assert 0 <= overflow < self.batch_size
        assert (len(times) + overflow) % self.batch_size == 0

        return DatasetWithLength(
            dataset=tf.data.Dataset.from_generator(
                gen, output_types=dtypes, output_shapes=shapes
            ).batch(self.batch_size, drop_remainder=True),
            num_batches=(len(times) + overflow) // self.batch_size,
            num_non_padding=len(times)
        )

    def make_dataset_lcs(self, lcs: typing.List[lightcurve._LC]) -> tf.data.Dataset:
        """Inefficient function to make a dataset."""
        first_lc_times = np.unique(np.concatenate(lcs[0].all_times(), axis=0))
        first_features = dict(
            self.extract_fcn(lcs[0], first_lc_times[0]), time=first_lc_times[0]
        )
        dtypes = {key: auto_dtype(key, value) for key, value in first_features.items()}
        shapes = {key: auto_shape(value) for key, value in first_features.items()}

        def gen(lc):
            times = np.unique(np.concatenate(lc.all_times(), axis=0))
            for time in times:
                extracted_dict = self.extract_fcn(lc, time)
                extracted_dict['time'] = time
                yield extracted_dict

        def gen_all():
            for lc in lcs:
                yield from gen(lc)

        return tf.data.Dataset.from_generator(
            gen_all, output_types=dtypes, output_shapes=shapes
        ).batch(
            self.batch_size, drop_remainder=True
        )

    def predict_single_lc(
        self, estimator: tf.estimator.Estimator, lc: lightcurve._LC, arrays_to_list=True
    ):
        ds_info = None

        def input_fn():
            nonlocal ds_info
            ds_info = self.make_dataset(lc)
            return ds_info.dataset

        predictions = estimator.predict(input_fn=input_fn, yield_single_examples=True)
        for i, example in enumerate(predictions):
            assert ds_info is not None, "input_fn() should be called by now!"
            if i < ds_info.num_non_padding:
                if arrays_to_list:
                    yield {
                        key: (value.tolist() if isinstance(value, np.ndarray) else value)
                        for key, value in example.items()
                    }
                else:
                    yield example
