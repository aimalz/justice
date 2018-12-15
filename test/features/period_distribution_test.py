import pytest
from justice.datasets import plasticc_data
from justice.features import period_distribution
import numpy as np


@pytest.mark.slow
@pytest.mark.requires_real_data
def test_lomb_scargle():
    source = plasticc_data.PlasticcBcolzSource.get_default()
    lc, = plasticc_data.PlasticcDatasetLC.bcolz_get_lcs_by_obj_ids(
        bcolz_source=source, dataset="training_set", obj_ids=[615]
    )
    period_transform = period_distribution.MultiBandLs()
    mbp = period_transform.apply(lc)
    period_best = mbp.best_periods[0]
    assert np.abs(period_best - .324499300710728) < 1e-5
