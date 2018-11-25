import pytest
from justice.datasets import plasticc_data
from justice.features import period_distribution
import numpy as np

@pytest.mark.requires_real_data
def test_lomb_scargle():
    training = plasticc_data.PlasticcDataset.training_data()
    lcs = plasticc_data.PlasticcDatasetLC.get_lcs_by_target("data/plasticc_training_data.db",92)
    period_transform = period_distribution.MultiBandLs()
    lc = lcs[0]
    mbp = period_transform.apply(lc)
    period_best = mbp.best_period
    assert np.abs(period_best-.324499300710728) < 1e-5
