# -*- coding: utf-8 -*-
"""Functions related to manipulating the PLAsTiCC dataset.

Note some functions are already in lightcurve.py's PlasticcDatasetLC class.
"""
import os.path
import random

import sqlite3

import pandas as pd

_data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))


def _filtered_fetch(cursor, *, col_idx, bad_col_value="target"):
    """Filters out erroneous row filled with column names.

    TODO(gatoatigrado): Fix SQL ingestion or use of fetchall in the future.
    """
    return [row for row in cursor if row[col_idx] != bad_col_value]


class PlasticcDataset(object):
    def __init__(self, *, filename, base_table_name, seed=None):
        self.base_table_name = base_table_name
        self.rng = random.Random(seed)
        self.filename = filename
        self.conn = sqlite3.connect(filename)
        self.index_df = pd.DataFrame(
            _filtered_fetch(
                self.conn.execute(
                    "select object_id, target from "
                    "{}_meta".format(base_table_name)
                ).fetchall(),
                col_idx=-1
            ),
            columns=["source_id", "target"]
        )

    def __getitem__(self, key: str) -> pd.Series:
        return self.index_df.iloc[key]

    def get_lc(self, obj_id):
        from justice import lightcurve
        return lightcurve.PlasticcDatasetLC.get_lc(
            self.filename, self.base_table_name, obj_id=obj_id
        )

    def target_breakdown(self) -> pd.Series:
        """Returns the number of light curves for each classified type."""
        return self.index_df.groupby("target").size()

    def random_idx(self):
        return self.rng.randint(0, len(self.index_df) - 1)

    def random_lc(self):
        obj_id = int(self[self.random_idx()].source_id)
        return obj_id, self.get_lc(obj_id)

    @classmethod
    def training_data(cls):
        return cls(
            filename=os.path.join(_data_dir, "plasticc_training_data.db"),
            base_table_name="training_set"
        )

    @classmethod
    def test_data(cls):
        return cls(
            filename=os.path.join(_data_dir, "plasticc_test_data.db"),
            base_table_name="test_set"
        )
