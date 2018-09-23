# -*- coding: utf-8 -*-
"""Tests parsing of OGLE data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from justice import ogle_data, ogle_data_builder


def test_build_and_get_random(testdata_dir, tmpdir):
    ogle_dir = str(tmpdir.mkdir("test_ogle"))
    ogle_data.ogle_dir = ogle_dir
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
    lc_dict = test_dataset.lc_dict_for_id('OGLE-LMC-CEP-0001')
    assert test_dataset['OGLE-LMC-CEP-0002'].field.tolist() == ['LMC157.6', 'LMC157.6']
    assert lc_dict.keys() == {"I", "V"}
    sample_data = lc_dict["I"][0:2, :].tolist()
    expected = [[479.65878256202586, 8.154761695700554e-08, 6.624523579080573e-11],
                [1273.9239423496151, 9.080244446205889e-08, 8.940833243072116e-11]]
    assert sample_data == expected
