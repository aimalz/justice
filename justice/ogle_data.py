# -*- coding: utf-8 -*-
"""Defines OGLE datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os.path

from justice import mmap_array

ogle_dir = os.path.join(mmap_array.default_array_dir, 'ogle_iii')


def for_subset(name):
    return mmap_array.IndexedArrayDescriptor(base_dir=os.path.join(ogle_dir, name))


class OgleDataset(mmap_array.IndexedArray):
    def __init__(self, index_df, lc_data):
        super(OgleDataset, self).__init__(index_df, lc_data)
        self.all_ids = sorted(frozenset(index_df['id'].tolist()))

    def random_id(self):
        return random.choice(self.all_ids)

    def lcs_for_id(self, id_):
        """Get light curves for an ID.

        :param id_: ID to get data for.
        :return: (index_row_namedtuple, sub_array) pairs.
            sub_array is a [num_points, 3]-shaped array.
        """
        row_and_data = []
        for row in self.index_df[self.index_df['id'] == id_].itertuples():
            data = self.lc_data[row.start:row.end, :]
            row_and_data.append((row, data))
        return row_and_data

    @classmethod
    def read_for_name(cls, name):
        ia = for_subset(name).read()
        return cls(ia.index_df, ia.lc_data)
