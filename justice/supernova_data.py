# -*- coding: utf-8 -*-
"""Parser for plain text supernova data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os.path
import sys
import pickle

import numpy as np
import pandas as pd

from justice import mmap_array

sn_dir = os.path.join(mmap_array.default_array_dir, 'sn_phot_cc')
index_filename = os.path.join(sn_dir, 'index_df.pickle')
all_lc_data = mmap_array.MmapArrayFile('all', array_dir=sn_dir, order='C')


def parse_truth(filename):
    with open(filename, 'r') as truthfile:
        return dict(line.strip().split() for line in truthfile if ".DAT" in line)


def make_parse_fcn(truth_filename):
    truth_dict = parse_truth(truth_filename) if truth_filename else {}

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
        orig_sn_type, = (l for typ, l in lines_with_type_tag if typ == "SNTYPE")
        sn_type = int(truth_dict.get(os.path.basename(filename), orig_sn_type))
        # if int(orig_sn_type) == -9 and sn_type != -9:
        # print("Found answer in key file.")
        sn_id, = (int(l) for typ, l in lines_with_type_tag if typ == "SNID")
        var_list, = (l.split() for typ, l in lines_with_type_tag if typ == "VARLIST")
        observations = [l.split() for typ, l in lines_with_type_tag if typ == "OBS"]
        df = pd.DataFrame(
            observations, columns=var_list
        ).astype({
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

    return parse_file


def generate_data_main():
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument(
        "--truth-file",
        default=None,
        help="File for an answer key, with lines like 'DES_SN____.DAT <id>''"
    )
    cmd_args.add_argument("input_dirs", nargs="+")
    args = cmd_args.parse_args()
    dat_files = [
        filename  # don't auto-format
        for input_dir in args.input_dirs
        for filename in glob.glob(os.path.join(input_dir, "*.DAT"))
    ]

    index_rows = []
    all_values = []  # will be concatenated.
    next_start = 0
    parse_file = make_parse_fcn(args.truth_file)
    for i, dat_file in enumerate(dat_files):
        sn_id, sn_type, by_flt = parse_file(dat_file)
        if i % 10 == 0:
            print(".", end="", file=sys.stderr)
            sys.stderr.flush()
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
        pickle.dump(index_df, f, protocol=2)
    all_lc_data.write(data)


class SNDataset(object):
    def __init__(self):
        with open(index_filename, 'rb') as f:
            self.index_df = pickle.load(f)
        self.lc_data = all_lc_data.read()
        self.rng = np.random.RandomState()  # Pre-init for faster sampling.
        self.all_ids = self.index_df['id'].unique()
        self.ids_with_answers = (
            self.index_df[self.index_df['type'] != -9]['id'].unique()
        )

    def random_lc(self):
        row = self.index_df.sample(1, random_state=self.rng).iloc[0]
        data = self.lc_data[row.start:row.end, :]
        return row, data

    def random_lc_all_flux(self, with_answers=True):
        if with_answers:
            random_id = self.rng.choice(self.ids_with_answers)
        else:
            random_id = self.rng.choice(self.all_ids)
        row_and_data = []
        for row in self.index_df[self.index_df['id'] == random_id].itertuples():
            data = self.lc_data[row.start:row.end, :]
            row_and_data.append((row, data))
        return random_id, row_and_data


if __name__ == '__main__':
    generate_data_main()
    SNDataset().random_lc()  # Check that it works.
