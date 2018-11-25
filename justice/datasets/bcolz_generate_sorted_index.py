# -*- coding: utf-8 -*-
"""Generates an index table for a bcolz table."""
import argparse
import pathlib
import itertools
import sys

import bcolz
import pandas as pd

from justice.datasets import plasticc_bcolz


def index_name(orig_table_name, column_name="object_id"):
    return f"{orig_table_name}__{column_name}_index"


def main():
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument("--dataset-dir", default=str(plasticc_bcolz._root_dir))
    cmd_args.add_argument("--table-name", required=True)
    cmd_args.add_argument("--column-name", required=True)
    args = cmd_args.parse_args()

    table_dir = pathlib.Path(args.dataset_dir) / args.table_name
    index_bcolz_dir = pathlib.Path(args.dataset_dir
                                   ) / index_name(args.table_name, args.column_name)
    assert table_dir.is_dir(), f"{table_dir} doesn't exist!"
    assert not index_bcolz_dir.exists(), f"{index_bcolz_dir} already exists!"
    table = bcolz.open(table_dir)
    col_values = table[args.column_name]

    seen = set()
    unique_obj_ids = []
    boundary_indices = [0]
    for group, values in itertools.groupby(col_values):
        if len(unique_obj_ids) % 50_000 == 0:
            print(".", end="")
            sys.stdout.flush()
        if group in seen:
            raise ValueError(
                f"Column {args.column_name} isn't contiguous; saw "
                f"{group} twice."
            )
        seen.add(group)
        count = sum(1 for _ in values)
        unique_obj_ids.append(group)
        boundary_indices.append(boundary_indices[-1] + count)
    print()

    assert len(boundary_indices) == len(unique_obj_ids) + 1
    df = pd.DataFrame.from_dict({
        'object_id': unique_obj_ids,
        'start_idx': boundary_indices[:-1],
        'end_idx': boundary_indices[1:]
    })
    table = bcolz.ctable.fromdataframe(
        df,
        cparams=bcolz.cparams(clevel=3, cname='lz4hc', shuffle=1),
        rootdir=index_bcolz_dir,
    )
    table.flush()


if __name__ == '__main__':
    main()
