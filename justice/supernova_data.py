# -*- coding: utf-8 -*-
"""Parser for plain text supernova data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle
import glob
import os.path
import sys

import numpy as np
import pandas as pd

from justice import mmap_array

sn_dir = os.path.join(mmap_array.default_array_dir, 'sn_phot_cc')
index_filename = os.path.join(sn_dir, 'index_df.pickle')
all_lc_data = mmap_array.MmapArrayFile('all', array_dir=sn_dir, order='C')


def parse_file(filename):
    # Everything seems to be of the form "A: rest of line", so parse this first.
    lines_with_type_tag = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            typ, value = line.split(":", 1)
            lines_with_type_tag.append((typ, value.strip()))

    # Get the type tag, there should be only one.
    sn_type, = (int(l) for typ, l in lines_with_type_tag if typ == "SNTYPE")
    sn_id, = (int(l) for typ, l in lines_with_type_tag if typ == "SNID")
    var_list, = (l.split() for typ, l in lines_with_type_tag if typ == "VARLIST")
    observations = [
        l.split() for typ, l in lines_with_type_tag if typ == "OBS"
    ]
    df = pd.DataFrame(observations, columns=var_list).astype({
        'MJD': np.float64,
        'FLUXCAL': np.float64,
        'FLUXCALERR': np.float64,
    })

    # by band filter
    by_flt = {
        k: subframe[['MJD', 'FLUXCAL', "FLUXCALERR"]].values.astype(np.float64)
        for k, subframe in df.groupby("FLT")
    }
    return sn_id, sn_type, by_flt


def generate_data_main():
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument("input_dir")
    args = cmd_args.parse_args()
    dat_files = glob.glob(os.path.join(args.input_dir, "*.DAT"))

    index_rows = []
    all_values = []  # will be concatenated.
    next_start = 0
    for dat_file in dat_files:
        sn_id, sn_type, by_flt = parse_file(dat_file)
        print(".", end="", file=sys.stderr)
        sys.stdout.flush()
        for flt, values in by_flt.items():
            start = next_start
            next_start += values.shape[0]
            index_rows.append({
                'id': sn_id,
                'type': sn_type,
                'flt': flt,
                'start': start,
                'end': next_start,
            })
            all_values.append(values)
    print(file=sys.stderr)
    data = np.concatenate(all_values, axis=0)
    assert data.shape[0] == next_start
    assert data.shape[0] == index_rows[-1]['end']

    # Write index and data.
    index_df = pd.DataFrame.from_records(index_rows)
    with open(index_filename, 'wb') as f:
        cPickle.dump(index_df, f, protocol=2)
    all_lc_data.write(data)


class SNDataset(object):
    def __init__(self):
        with open(index_filename, 'rb') as f:
            self.index_df = cPickle.load(f)
        self.lc_data = all_lc_data.read()
        self.rng = np.random.RandomState()  # Pre-init for faster sampling.

    def random_lc(self):
        row = self.index_df.sample(1, random_state=self.rng).iloc[0]
        data = self.lc_data[row.start:row.end, :]
        return row, data


if __name__ == '__main__':
    generate_data_main()
    SNDataset().random_lc()  # Check that it works.
