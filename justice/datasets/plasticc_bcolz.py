# -*- coding: utf-8 -*-
"""Imports PLasTiCC datasets to bcolz."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import os.path
import shutil
import subprocess

import bcolz

from justice import path_util

_root_dir = path_util.data_dir / 'plasticc_bcolz'

_col_to_data_type = {
    'object_id': 'i4',
    'mjd': 'f4',
    'passband': 'i4',
    'flux': 'f4',
    'flux_err': 'f4',
    'detected': 'i1',
    'ra': 'f4',
    'decl': 'f4',
    'gal_l': 'f4',
    'gal_b': 'f4',
    'ddf': 'i4',
    'hostgal_specz': 'f4',
    'hostgal_photoz': 'f4',
    'hostgal_photoz_err': 'f4',
    'distmod': 'f4',
    'mwebv': 'f4',
    'target': 'i4',
}
_col_to_converter = {
    key: (
        int if value.startswith("i") else
        (float if value.startswith("f") else NotImplemented)
    )
    for key, value in _col_to_data_type.items()
}


class BcolzDataset(object):
    def __init__(self, bcolz_dir):
        self.bcolz_dir = bcolz_dir

        if not bcolz_dir.parent.is_dir():
            os.makedirs(str(bcolz_dir.parent))

    def read_table(self):
        return bcolz.open(self.bcolz_dir, 'r')

    def clear_files(self):
        if self.bcolz_dir.exists():
            if input(f"Type 'y' to delete existing {self.bcolz_dir}: ").strip() != "y":
                raise EnvironmentError("Output directory already exists!")
            assert "plasticc" in str(self.bcolz_dir)  # rmtree safety
            shutil.rmtree(str(self.bcolz_dir))

    @property
    def column_names_json(self):
        return self.bcolz_dir / "column_names.json"

    def write_column_names(self, column_names):
        assert isinstance(column_names, list)
        with open(self.column_names_json, 'w') as f:
            json.dump(column_names, f)

    def read_column_names(self):
        with open(self.column_names_json, 'r') as f:
            return json.load(f)


def create_dataset(*, csv_reader, out_dir, num_rows):
    first_row = next(csv_reader)
    types = ",".join(_col_to_data_type[col] for col in first_row.keys())
    converters = [_col_to_converter[col] for col in first_row.keys()]

    def _data_gen():
        num_read = 0
        first_row_values = tuple(c(v) for c, v in zip(converters, first_row.values()))
        print(f"First row: {first_row_values}")
        yield first_row_values
        num_read += 1
        for row in csv_reader:
            yield tuple(c(v) for c, v in zip(converters, row.values()))
            num_read += 1
            if num_read % 1_000_000 == 0:
                print(f"    Read {num_read} rows ...")
        assert num_read == num_rows, f"Estimate of number of rows {num_rows} != actual {num_read}!"

    print(f"Generating bcolz table with {num_rows} rows, data types {types}")
    gen = _data_gen()
    table = bcolz.fromiter(gen, dtype=types, count=num_rows, rootdir=str(out_dir))
    try:
        next(gen)
        raise ValueError("Not all rows consumed!")
    except StopIteration:
        return first_row, table


def main():
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument("--name")
    cmd_args.add_argument("--source-file", required=True)
    args = cmd_args.parse_args()

    if args.name is None:
        args.name = os.path.basename(args.source_file).replace(".csv", "")
    out_dir = _root_dir / args.name
    dataset = BcolzDataset(out_dir)

    dataset.clear_files()  # prompt and clear files if they exist.

    print("Scanning number of rows")
    with open(args.source_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header_row = next(reader)
        for name in header_row:
            assert name in _col_to_data_type, f"Unknown column {name}"
        wc_output = subprocess.check_output(["wc", "-l", args.source_file])
        num_rows = int(wc_output.strip().split()[0]) - 1  # has a header

    with open(args.source_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        first_row, table = create_dataset(
            csv_reader=reader, num_rows=num_rows, out_dir=out_dir
        )
        del table  # unused

    dataset.write_column_names(list(first_row.keys()))


if __name__ == '__main__':
    main()
