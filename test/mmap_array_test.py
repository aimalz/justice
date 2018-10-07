# -*- coding: utf-8 -*-
"""Basic tests for writing mmap arrays."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from justice.datasets import mmap_array


def test_write_indexed(tmpdir):
    index_rows = [
        {
            'id': 1234,
            'band': 'i'
        },
        {
            'id': 342,
            'band': 'z'
        },
        {
            'id': 613,
            'band': 'i'
        },
        {
            'id': 613,
            'band': 'z'
        },
    ]
    data = [
        [(123, 4, 1), (123, 5, 1)],
        [(34, 4, 1)],
        [(61.1, 4, 10), (61.2, 4, 10)],
        [(61, 4, 1), (61.3, 4, 1)],
    ]
    descriptor = mmap_array.IndexedArrayDescriptor(
        base_dir=tmpdir.mkdir("test_write_indexed")
    )
    assert descriptor.index_filename.endswith("index_df.pickle")

    descriptor.write(index_rows, data)
    ia = descriptor.read()
    assert ia.index_df.iloc[0].start == 0
    assert ia.index_df.iloc[0].end == 2
    assert ia.index_df.iloc[1].start == 2
    assert ia.index_df.iloc[1].end == 3

    assert ia.get_data(ia.index_df.iloc[0]).tolist() == [[123.0, 4.0, 1.0],
                                                         [123.0, 5.0, 1.0]]
    assert ia.get_data(ia.index_df.iloc[3]).tolist() == [[61.0, 4.0, 1.0],
                                                         [61.3, 4.0, 1.0]]
