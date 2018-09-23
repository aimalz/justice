import pytest

import numpy as np
import justice.lightcurve
import justice.simulate as sim
import justice.summarize as summ
from justice import xform

def test_connect_the_dots():
    glc1 = sim.TestLC.make_easy_gauss()
    print(glc1.connect_the_dots())

def test_arclen():
    glc = sim.TestLC.make_hard_gauss()

    aff = glc.get_xform([50., 1., 1., 1.5, 1., 1.])
    glc2 = xform.transform(glc, aff)

    aff2 = summ.opt_arclen(glc, glc2, vb=False, options={'maxiter': 10})
    del aff2  # unused

