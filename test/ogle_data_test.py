# -*- coding: utf-8 -*-
"""Tests parsing of OGLE data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import pathlib
import pytest

from justice.datasets import ogle_data_builder, ogle_data


@pytest.mark.skip()
def test_build_and_get_random(testdata_dir, tmpdir):
    ogle_dir = str(tmpdir.mkdir("test_ogle"))
    ogle_data.ogle_dir = pathlib.Path(ogle_dir)
    ia_descriptor = ogle_data.for_subset("test_name")
    assert ia_descriptor.index_filename.startswith(ogle_dir)

    testdata_dir = os.path.join(testdata_dir, "ogle_lmc_cep_like")
    ogle_data_builder.build(
        testdata_dir,
        ia_descriptor,
        expected_bands=('I', 'V'),
        star_table=os.path.join(testdata_dir, 'info.txt')
    )

    test_dataset = ogle_data.OgleDataset.read_for_name("test_name")
    assert test_dataset.all_ids == ['OGLE-LMC-CEP-0001', 'OGLE-LMC-CEP-0002']
    lc = test_dataset.lc_for_id('OGLE-LMC-CEP-0001')
    assert test_dataset['OGLE-LMC-CEP-0002'].field.tolist() == ['LMC157.6', 'LMC157.6']
    assert lc.bands.keys() == {"I", "V"}
    assert lc["I"].time[0:2].tolist() == [479.65878256202586, 1273.9239423496151]
    assert lc["I"].flux[0:2].tolist() == [0.00029608150135886817, 0.00032968374903654405]
    assert lc["I"].flux_err[0:2].tolist() == [
        2.405219134871237e-07, 3.246220341918471e-07
    ]
