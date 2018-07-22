# -*- coding: utf-8 -*-
"""Helper class for mmap'd arrays. These are typically fast to load into memory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import os.path

import numpy as np

default_array_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "../../data/mmap_arrays"))


class MmapArrayFile(object):
    def __init__(self, name, array_dir=default_array_dir, create_parent_dir=True,
                 order='C'):
        if order not in {'C', 'F'}:
            raise ValueError("order parameter must be 'C' or 'F'. See numpy docs.")
        self.array_dir = array_dir
        self.name = name
        self.order = order

        # NOTE: name could include a '/' so don't just use array_dir.
        parent_dir = os.path.dirname(os.path.join(array_dir, self.name))
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)

    @property
    def mmap_filename(self):
        ext = ".column-major-numpy-mmap" if self.order == "F" else ".numpy-mmap"
        return os.path.join(self.array_dir, self.name + ext)

    @property
    def info_file(self):
        return os.path.join(self.array_dir, self.name + "-info.json")

    def exists(self):
        return os.path.isfile(self.mmap_filename) and os.path.isfile(self.info_file)

    def write(self, array):
        if not isinstance(array, np.ndarray):
            raise TypeError("Expected an ndarray")

        np.memmap(self.mmap_filename, dtype=array.dtype, mode='w+', shape=array.shape,
                  order=self.order)[:] = array
        try:
            with open(self.info_file, 'w') as f:
                json.dump({'shape': array.shape, 'dtype': array.dtype.name}, f)
        except Exception:
            os.unlink(self.info_file)
            raise

    def read(self):
        with open(self.info_file, 'r') as f:
            shape_dtype = json.load(f)
            shape_dtype['shape'] = tuple(shape_dtype['shape'])
            shape_dtype['dtype'] = getattr(np, shape_dtype['dtype'])
        return np.memmap(self.mmap_filename, mode='r', order=self.order, **shape_dtype)