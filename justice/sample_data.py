import glob
import os
import os.path
import time

import numpy as np
import pandas as pd

import download_data
from justice import mmap_array

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

    f = mmap_array.MmapArrayFile(name)
    if not f.exists():
        data = pd.read_csv(source_filename, names=['time', 'value', 'error']).as_matrix()
        entries, num_cols = data.shape
        assert num_cols == 3, "sanity check"
        assert 100 < entries < int(1e9), "sanity check"
        f.write(data)
    return f.read()


def get_downsampled_data(name):
    f = mmap_array.MmapArrayFile(name + "-downsampled")
    if not f.exists():
        original = get_sample_data(name)
        npt_lsst = np.ceil(np.ptp(original[:, 0]) / 1.6).astype('int')
        f.write(original[::npt_lsst, :])
    return f.read()


if __name__ == '__main__':
    start_time = time.time()
    for data_name in get_data_names():
        get_downsampled_data(data_name)
    print("Elapsed: {:.3f} s".format(time.time() - start_time))
