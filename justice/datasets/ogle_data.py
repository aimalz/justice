# -*- coding: utf-8 -*-
"""Defines OGLE datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random

from justice import lightcurve
from justice.datasets import mmap_array

ogle_dir = mmap_array.default_array_dir / 'ogle_iii'


class OGLEDatasetLC(lightcurve._LC):
    """OGLE dataset light curve."""
    expected_bands = ['I', 'V']


def for_subset(name):
    return mmap_array.IndexedArrayDescriptor(base_dir=(ogle_dir / name))


class OgleDataset(mmap_array.IndexedArray):
    def __init__(self, index_df, lc_data):
        super(OgleDataset, self).__init__(index_df, lc_data)
        self.all_ids = sorted(frozenset(index_df.index.tolist()))

    def __getitem__(self, key):
        assert key.startswith("OGLE")
        return self.index_df.ix[key]

    def random_id(self):
        return random.choice(self.all_ids)

    def lcs_for_id(self, id_):
        """Get light curves for an ID.

        :param id_: ID to get data for.
        :return: (index_row_namedtuple, sub_array) pairs.
            sub_array is a [num_points, 3]-shaped array.
        """
        row_and_data = []
        for row in self.index_df.ix[id_].itertuples():
            data = self.lc_data[row.start:row.end, :]
            row_and_data.append((row, data))
        return row_and_data

    def lc_for_id(self, id_):
        """Returns a light curve for an ID.

        :param id_: ID to get data for.
        :return: Dict from band ('i', 'r', etc.) to [num_points, 3]-shaped array.
            The first column is time, second is flux, third is flux error.
        """
        row_and_data = self.lcs_for_id(id_)
        result = {
            row.band: lightcurve.BandData.from_dense_array(array)
            for row, array in row_and_data
        }
        if len(row_and_data) != len(result):
            raise ValueError("band does not uniquely identify sub-ranges")
        return OGLEDatasetLC(**result)

    @classmethod
    def read_for_name(cls, name):
        ia = for_subset(name).read()
        return cls(ia.index_df, ia.lc_data)
