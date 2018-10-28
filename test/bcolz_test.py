import pytest

from justice.datasets import plasticc_data


@pytest.mark.requires_real_data
def test_can_load_bcolz_from_obj_id():
    plasticc_data.PlasticcDatasetLC.get_lc('data/plasticc_bcolz/', 'test_set', 13)


@pytest.mark.requires_real_data
def test_can_load_bcolz_from_target():
    lcs = plasticc_data.PlasticcDatasetLC.get_lc_by_target('data/plasticc_bcolz/', 67)
    assert len(lcs) == 208
