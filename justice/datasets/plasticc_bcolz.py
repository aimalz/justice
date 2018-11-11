# -*- coding: utf-8 -*-
"""Imports PLasTiCC datasets to bcolz."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import shutil

import bcolz
import pandas as pd
import numpy as np
import pathlib

from justice import path_util

_root_dir = path_util.data_dir / 'plasticc_bcolz'


class BcolzDataset(object):
    def __init__(self, bcolz_dir):
        if isinstance(bcolz_dir, (str, bytes)):
            bcolz_dir = pathlib.Path(bcolz_dir)
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


def _safe_cast(name, series: pd.Series):
    if series.dtype == np.float64:
        assert series.abs().max() < 1e37, "Max too close to float32 max."
        return series.astype(np.float32)
    elif series.dtype == np.int64:
        if name == "detected":
            assert series.abs().max() < 128, "Max too close to int8 max."
            return series.astype(np.int8)
        else:
            assert series.abs().max() < 2e9, "Max too close to int32 max."
            return series.astype(np.int32)
    else:
        raise TypeError(f"Unexpected non-int/float column type {series.dtype}")


def _convert_df_to_32_bit(df):
    for col in df.columns.tolist():
        df[col] = _safe_cast(col, df[col])


def create_dataset(*, source_file, out_dir):
    data_frame_chunks = pd.read_csv(source_file, chunksize=1_000_000)
    first_chunk: pd.DataFrame = next(data_frame_chunks)
    _convert_df_to_32_bit(first_chunk)
    column_names = first_chunk.columns.tolist()

    # Note: To work around a bug when `names` is present but `columns` is empty,
    # construct this manually.
    table = bcolz.ctable.fromdataframe(
        first_chunk,
        # For some reason, higher compression levels are actually performing worse.
        cparams=bcolz.cparams(clevel=3, cname="lz4hc", shuffle=1),
        rootdir=str(out_dir),
    )

    for next_chunk in data_frame_chunks:
        _convert_df_to_32_bit(next_chunk)
        table.append(cols=[next_chunk[col] for col in column_names])
    table.flush()
    num_rows = table.shape[0]
    size_mb = table.cbytes / (1024.0**2)
    print(
        f"Created bcolz table with {num_rows} rows, compression settings "
        f"{table.cparams}, final size {size_mb:.1f} MiB"
    )


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
    create_dataset(source_file=args.source_file, out_dir=out_dir)


if __name__ == '__main__':
    main()
