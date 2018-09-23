# -*- coding: utf-8 -*-
"""Manages NOAO dataset files.

A sample of GAIA data was downloaded via the NOAO DataLab tool; please email the
authors if you want instructions for how to request instructions for downloading this
yourself.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import pickle
import random
import time

import numpy as np
import pandas as pd

from justice import mmap_array

pickle_file = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "../../gaia-selection.pickle")
)
gaia_dir = os.path.join(mmap_array.default_array_dir, 'gaia')
source_id_to_ranges_index = mmap_array.MmapArrayFile(
    'source_id_to_ranges_index', array_dir=gaia_dir, order='C'
)
all_lc_data = mmap_array.MmapArrayFile('all', array_dir=gaia_dir, order='C')


def write_mmap_file():
    if not os.path.isfile(pickle_file):
        raise ValueError("Please download the GAIA pickle file (see module docstring).")

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        assert isinstance(data, pd.DataFrame)

    data = data.sort_values('source_id').reset_index()
    index_data = [(source_id, min(group.index), max(group.index))
                  for source_id, group in data.groupby("source_id")]
    source_id_to_ranges_index.write(np.array(index_data, dtype=np.int64))

    if not all_lc_data.exists():
        array = data[['time', 'flux', 'flux_error']].values
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64, "Please re-download data in float64 format"
        all_lc_data.write(array)


def benchmark(num_selects=100):
    start_time = time.time()
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        assert isinstance(data, pd.DataFrame)
        source_ids = sorted(frozenset(data['source_id'].values))
    print("Time for reading pickle file: {:.2} s".format(time.time() - start_time))

    start_time = time.time()
    source_ids = sorted(frozenset(data['source_id']))
    for _ in range(num_selects):
        random_id = random.choice(source_ids)
        subsel = data[data['source_id'] == random_id].values
        assert isinstance(subsel, np.ndarray)
    print(
        "Time per dataframe random id selection: {:.2} ms".format(
            1000 * (time.time() - start_time)
        )
    )

    start_time = time.time()
    index_array = source_id_to_ranges_index.read()
    array = all_lc_data.read()
    print(
        "Time for reading mmap data and index: {:.2} s".format(time.time() - start_time)
    )

    start_time = time.time()
    for _ in range(num_selects):
        random_id, min_idx, max_idx = random.choice(index_array)
        subsel = array[min_idx:(max_idx + 1)]
        assert isinstance(subsel, np.ndarray)
    print(
        "Time per random id selection (mmap arrays): {} ms".format(
            1000 * (time.time() - start_time)
        )
    )


if __name__ == '__main__':
    if source_id_to_ranges_index.exists() and all_lc_data.exists():
        print("mmap arrays already generated, benchmarking ...")
        benchmark()
    else:
        print("Generating mmap version ... re-run this script to benchmark.")
        write_mmap_file()
