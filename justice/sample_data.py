import glob
import json
import os
import os.path

import numpy as np
import pandas as pd

import download_data

array_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/mmap_arrays"))

lc_data_dir = os.path.join(download_data.time_series_dir, 'lc_data')


def get_data_names():
    if not os.path.isdir(download_data.time_series_dir):
        raise EnvironmentError("Please run the download_data script first.")
    csv_files = glob.glob(os.path.join(lc_data_dir, '*.csv'))
    if not csv_files:
        raise ValueError("Didn't find any CSV files in time_series_demo/lc_data. Maybe that repo changed?")
    return [os.path.basename(f).replace(".csv", "") for f in csv_files]


def get_sample_data(name):
    source_filename = os.path.join(lc_data_dir, name + '.csv')
    if not os.path.isfile(source_filename):
        raise EnvironmentError("Expected to find a file {!r}".format(source_filename))

    if not os.path.isdir(array_dir):
        os.makedirs(array_dir)

    mmap_filename = os.path.join(array_dir, name + ".numpy-mmap")
    info_file = os.path.join(array_dir, name + "-info.json")
    if not (os.path.isfile(mmap_filename) and os.path.isfile(info_file)):
        data = pd.read_csv(source_filename, names=['time', 'value', 'error']).as_matrix()
        entries, num_cols = data.shape
        assert num_cols == 3, "sanity check"
        assert 100 < entries < int(1e9), "sanity check"
        np.memmap(mmap_filename, dtype=data.dtype, mode='w+', shape=data.shape)[:] = data
        try:
            with open(info_file, 'w') as f:
                json.dump({'shape': data.shape, 'dtype': data.dtype.name}, f)
        except Exception:
            os.unlink(info_file)
            raise

    with open(info_file, 'r') as f:
        shape_dtype = json.load(f)
        shape_dtype['shape'] = tuple(shape_dtype['shape'])
        shape_dtype['dtype'] = getattr(np, shape_dtype['dtype'])
    return np.memmap(mmap_filename, mode='r', **shape_dtype)


if __name__ == '__main__':
    for data_name in get_data_names():
        get_sample_data(data_name)
