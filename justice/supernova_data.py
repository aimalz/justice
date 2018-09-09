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


def generate_binary_data(
    dat_files,
    truth_file,
    override_index_filename,
    override_all_lc_data,
    print_status=True
):
    index_rows = []
    all_values = []  # will be concatenated.
    next_start = 0
    parse_file = make_parse_fcn(truth_file)
    for i, dat_file in enumerate(dat_files):
        sn_id, sn_type, by_flt = parse_file(dat_file)
        if print_status and i % 10 == 0:
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
    if print_status:
        print(file=sys.stderr)
    data = np.concatenate(all_values, axis=0)
    assert data.shape[0] == next_start
    assert data.shape[0] == index_rows[-1]['end']
    # Write index and data.
    index_df = pd.DataFrame.from_records(index_rows)
    with open(override_index_filename, 'wb') as f:
        pickle.dump(index_df, f, protocol=2)
    override_all_lc_data.write(data)


def generate_data_main():
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument(
        "--truth-file",
        default=None,
        help="File for an answer key, with lines like 'DES_SN____.DAT <id>''"
    )
    cmd_args.add_argument("input_dirs", nargs="+")
    args = cmd_args.parse_args()
    truth_file = args.truth_file
    dat_files = [
        filename  # don't auto-format
        for input_dir in args.input_dirs
        for filename in glob.glob(os.path.join(input_dir, "*.DAT"))
    ]
    override_index_filename = index_filename
    override_all_lc_data = all_lc_data

    generate_binary_data(
        dat_files, truth_file, override_index_filename, override_all_lc_data
    )


def format_dense_multi_band_from_lc_dict(lc_dict, band_order=('g', 'r', 'i', 'z')):
    """Formats a multi-band LC dictionary to a dense dataset.

    Currently reformats a time series to dense data, as if every curve had sampled at the same time.
    This method is currently slow.

    :param lc_dict: Dictionary from lc_dict_for_id.
    :param band_order: Order of expected bands.
    :return: lightcurve.LC object.
    """
    from justice import lightcurve
    if frozenset(lc_dict.keys()) != frozenset(band_order):
        raise ValueError("Unexpected keys {}".format(lc_dict.keys()))

    def closest_in_time(band, time):
        index = np.argmin(np.abs(band[:, 0] - time))
        # [3]-shaped array
        return band[index]

    bands = [lc_dict[band] for band in band_order]

    # [num_points, num_bands, 3]-shaped array
    dense_data = np.array([[closest_in_time(band, time)
                            for band in bands]
                           for time in bands[0][:, 0]],
                          dtype=np.float64)
    return lightcurve.LC(
        x=dense_data[:, :, 0], y=dense_data[:, :, 1], yerr=dense_data[:, :, 2]
    )


class SNDataset(object):
    """API over dataset for supernovae.


    Attributes:
      ids_with_answers: IDs with any answers.
    """

    def __init__(
        self, override_index_filename=index_filename, override_all_lc_data=all_lc_data
    ):
        with open(override_index_filename, 'rb') as f:
            self.index_df = pickle.load(f)
        self.lc_data = override_all_lc_data.read()
        self.rng = np.random.RandomState()  # Pre-init for faster sampling.
        self.all_ids = self.index_df['id'].unique()
        self.ids_with_answers = (
            self.index_df[self.index_df['type'] != -9]['id'].unique()
        )

    def random_lc(self):
        row = self.index_df.sample(1, random_state=self.rng).iloc[0]
        data = self.lc_data[row.start:row.end, :]
        return row, data

    def lcs_for_id(self, id_):
        """Get light curves for an ID.

        :param id_: ID to get data for.
        :return: (index_row_namedtuple, subarray) pairs.
            subarray is a [num_points, 3]-shaped array.
        """
        row_and_data = []
        for row in self.index_df[self.index_df['id'] == id_].itertuples():
            data = self.lc_data[row.start:row.end, :]
            row_and_data.append((row, data))
        return row_and_data

    def lc_dict_for_id(self, id_):
        """Returns a dictionary mapping band to light curve array.

        :param id_: ID to get data for.
        :return: Dict from band ('i', 'r', etc.) to [num_points, 3]-shaped array.
            The first column is time, second is flux, third is flux error.
        """
        row_and_data = self.lcs_for_id(id_)
        result = {row.flt: array for row, array in row_and_data}
        if len(row_and_data) != len(result):
            raise ValueError("band does not uniquely identify sub-ranges")
        return result

    def random_answer_id(self):
        return self.rng.choice(self.ids_with_answers)

    def random_id(self):
        return self.rng.choice(self.all_ids)


if __name__ == '__main__':
    generate_data_main()
    SNDataset().random_lc()  # Check that it works.
