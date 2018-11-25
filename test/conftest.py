# -*- coding: utf-8 -*-
"""Pytest fixtures shared between other modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import shutil
import tempfile

import pytest
import tensorflow as tf

# noinspection PyTrailingSemicolon
import matplotlib; matplotlib.use("Agg")  # noqa, yapf: disable
from justice.datasets import supernova_data, mmap_array

_testdata_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../testdata"))


def _get_dat_files():
    files = glob.glob(os.path.join(_testdata_dir, "*.DAT"))
    assert len(files) == 2  # adjust as needed
    return files


@pytest.fixture()
def testdata_dir():
    return _testdata_dir


@pytest.fixture(scope="session")
def sn_index_and_mmap_file():
    temp_dir = tempfile.mkdtemp()
    try:
        index_filename = os.path.join(temp_dir, "index_df.pickle")
        all_lc_data = mmap_array.MmapArrayFile('all', array_dir=temp_dir, order='C')
        supernova_data.generate_binary_data(
            dat_files=_get_dat_files(),
            truth_file=None,
            override_index_filename=index_filename,
            override_all_lc_data=all_lc_data,
            print_status=False,
        )
        yield index_filename, all_lc_data
    finally:
        assert ("temp" in temp_dir) or ("tmp" in temp_dir)  # shutil safety
        shutil.rmtree(temp_dir)


@pytest.fixture
def sn_dataset(sn_index_and_mmap_file):
    index_filename, lc_data = sn_index_and_mmap_file
    return supernova_data.SNDataset(
        override_index_filename=index_filename, override_all_lc_data=lc_data
    )


@pytest.fixture(scope="function")
def tf_sess():
    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:
            yield sess
