import itertools

import pytest

from justice.datasets import plasticc_data


@pytest.mark.requires_real_data
def test_can_load_bcolz_from_obj_id():
    source = plasticc_data.PlasticcBcolzSource.get_default()
    lc = plasticc_data.PlasticcDatasetLC.get_lc(source, 'test_set', 13)
    assert isinstance(lc, plasticc_data.PlasticcDatasetLC)
    assert lc.is_sane()


@pytest.mark.requires_real_data
def test_can_load_bcolz_from_target():
    source = plasticc_data.PlasticcBcolzSource.get_default()
    lcs = plasticc_data.PlasticcDatasetLC.get_lcs_by_target(source, 67)
    assert len(lcs) == 208


@pytest.mark.requires_real_data
def test_bcolz_nonsense_1():
    source = plasticc_data.PlasticcBcolzSource.get_default()
    lcs = plasticc_data.PlasticcDatasetLC.get_lcs_by_target(source, -1)
    assert len(lcs) == 0


@pytest.mark.requires_real_data
def test_bcolz_nonsense_2():
    source = plasticc_data.PlasticcBcolzSource.get_default()
    try:
        plasticc_data.PlasticcDatasetLC.get_lc(source, 'training_set', -8675309)
    except KeyError:
        pass


@pytest.mark.requires_real_data
def test_can_pass_plasticc_bcolz_source_to_get_lc():
    source = plasticc_data.PlasticcBcolzSource.get_default()
    lc = plasticc_data.PlasticcDatasetLC.get_lc(source, 'test_set', 13)
    assert isinstance(lc, plasticc_data.PlasticcDatasetLC)
    assert lc.is_sane()


@pytest.mark.requires_real_data
def test_can_pass_plasticc_bcolz_source_to_get_lcs_by_target():
    source = plasticc_data.PlasticcBcolzSource.get_default()
    lcs = plasticc_data.PlasticcDatasetLC.get_lcs_by_target(source, 67)
    assert len(lcs) == 208


@pytest.mark.requires_real_data
def test_iterate_through_light_curves():
    source = plasticc_data.PlasticcBcolzSource.get_default()
    lcs_generator = plasticc_data.PlasticcDatasetLC.bcolz_get_all_lcs(
        source, 'test_set', chunk_size=2
    )
    lcs = list(itertools.islice(lcs_generator, 5))
    assert len(lcs) == 5
    for lc in lcs:
        assert isinstance(lc, plasticc_data.PlasticcDatasetLC)
        assert len(lc.all_times_unique()) > 0
