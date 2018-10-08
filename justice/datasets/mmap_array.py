# -*- coding: utf-8 -*-
"""Helper class for mmap'd arrays. These are typically fast to load into memory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import os.path
import pathlib
import pickle

import numpy as np
import pandas as pd

from justice import path_util

default_array_dir: pathlib.Path = path_util.data_dir / "mmap_arrays"


class MmapArrayFile(object):
    def __init__(
        self, name: str, array_dir=default_array_dir, create_parent_dir=True, order='C'
    ):
        if order not in {'C', 'F'}:
            raise ValueError("order parameter must be 'C' or 'F'. See numpy docs.")
        self.array_dir = pathlib.Path(array_dir)
        self.name = name
        self.order = order

        # NOTE: name could include a '/' so don't just use array_dir.
        path = (self.array_dir / self.name).resolve()
        if not path.parent.is_dir():
            os.makedirs(str(path.parent))

    @property
    def mmap_filename(self):
        ext = ".column-major-numpy-mmap" if self.order == "F" else ".numpy-mmap"
        return self.array_dir / (self.name + ext)

    @property
    def info_file(self):
        return self.array_dir / (self.name + "-info.json")

    def exists(self):
        return self.mmap_filename.is_file() and self.info_file.is_file()

    def write(self, array):
        if not isinstance(array, np.ndarray):
            raise TypeError("Expected an ndarray")

        np.memmap(
            str(self.mmap_filename),
            dtype=array.dtype,
            mode='w+',
            shape=array.shape,
            order=self.order
        )[:] = array
        try:
            with open(str(self.info_file), 'w') as f:
                json.dump({'shape': array.shape, 'dtype': array.dtype.name}, f)
        except Exception:
            os.unlink(str(self.info_file))
            raise

    def read(self):
        with open(str(self.info_file), 'r') as f:
            shape_dtype = json.load(f)
            shape_dtype['shape'] = tuple(shape_dtype['shape'])
            shape_dtype['dtype'] = getattr(np, shape_dtype['dtype'])
        return np.memmap(
            str(self.mmap_filename), mode='r', order=self.order, **shape_dtype
        )


class IndexedArray(object):
    def __init__(self, index_df, lc_data):
        self.index_df = index_df
        self.lc_data = lc_data

    def get_data(self, row):
        return self.lc_data[row.start:row.end, :]


class IndexedArrayDescriptor(object):
    def __init__(self, base_dir, index_name="index_df.pickle", array_name="all"):
        self.index_filename = os.path.join(base_dir, index_name)
        self.all_lc_data = MmapArrayFile(array_name, array_dir=base_dir, order='C')

    def read(self):
        with open(self.index_filename, 'rb') as f:
            index_df = pickle.load(f)
        return IndexedArray(index_df=index_df, lc_data=self.all_lc_data.read())

    def write(self, index_rows, data, set_index=None):
        """Writes parsed data.

        :param index_rows: List of dicts, each with information about each sub-array.
        :param data: List of sub-arrays; each sub-array is a list of
            (time, flux, flux_err) values.
        :param set_index: Name of column to use as data frame index.
        """
        if len(index_rows) != len(data):
            raise ValueError("Number of index rows must match number of data elements")

        # Accumulate list of starting indices. This will yield (n + 1) values.
        start_indices = [0]
        for sub_array in data:
            start_indices.append(start_indices[-1] + len(sub_array))
            if sub_array:
                assert len(sub_array[0]) == 3, "Expected tuples of (time, flux, err)"

        # Write index data frame with start and end indices.
        index_df = pd.DataFrame.from_records(index_rows)
        index_df['start'] = np.array(start_indices[:-1], dtype=np.int64)
        index_df['end'] = np.array(start_indices[1:], dtype=np.int64)
        if set_index is not None:
            index_df = index_df.set_index(set_index)
        with open(self.index_filename, 'wb') as f:
            pickle.dump(index_df, f, protocol=2)

        # Write all of the data.
        all_values = np.array([
            time_flux_err_tuple for sub_array in data for time_flux_err_tuple in sub_array
        ],
            dtype=np.float64)
        self.all_lc_data.write(all_values)
