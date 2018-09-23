# -*- coding: utf-8 -*-
"""Test that SNDataset API functions well."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from justice import supernova_data


def test_random_id_functions(sn_dataset):
    """
    :type sn_dataset: justice.supernova_data.SNDataset
    :param sn_dataset: Supernova dataset fixture.
    """
    random_ids = frozenset(sn_dataset.random_answer_id() for _ in range(10))
    assert random_ids == frozenset([92234, 197655])


def test_lc_dict_for_id(sn_dataset):
    """
    :type sn_dataset: justice.supernova_data.SNDataset
    :param sn_dataset: Supernova dataset fixture.
    """
    dct = sn_dataset.lc_dict_for_id(92234)
    assert set(dct.keys()) == {'g', 'r', 'i', 'z'}
    for val in dct.values():
        assert isinstance(val, np.ndarray)
        entries, num_columns = val.shape
        assert 0 < entries < 100
        assert num_columns == 3


def test_format_dense_multi_band_lc(sn_dataset):
    dct = sn_dataset.lc_dict_for_id(92234)
    lc = supernova_data.format_dense_multi_band_from_lc_dict(dct)
    first_times_all_channels = [band.time[0] for band in lc.bands.values()]
    all_shapes = frozenset(array.shape for array in first_times_all_channels)
    assert len(all_shapes) == 1
    first_times_all_channels = np.stack(first_times_all_channels, axis=0)
    assert (np.max(first_times_all_channels) - np.min(first_times_all_channels)) < 1.0
    assert 56177.0 < first_times_all_channels[0] < 56179.0
