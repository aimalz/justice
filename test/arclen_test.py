import pytest

import numpy as np
import justice.lightcurve
import justice.simulate as sim
import justice.summarize as summ
from justice import xform


def test_connect_the_dots():
    glc1 = sim.TestLC.make_easy_gauss()
    assert (np.abs(glc1.connect_the_dots() - 225.14157021861462) < 1e-4)


def test_arclen():
    glc = sim.TestLC.make_hard_gauss()

    aff = glc.get_xform([50., {'a': 1., 'b': 1.}, 1.5, {'a': 1., 'b': 1.}, 0.])
    glc2 = xform.transform(glc, aff)

    aff2 = summ.opt_arclen(glc, glc2, vb=False, options={'maxiter': 10})
    del aff2  # unused
