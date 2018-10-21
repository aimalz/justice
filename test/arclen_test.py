import numpy as np
import justice.simulate as sim
import justice.summarize as summ
from justice import xform


def test_connect_the_dots():
    glc1 = sim.TestLC.make_easy_gauss()
    assert (np.abs(glc1.connect_the_dots() - 225.14157021861462) < 1e-4)


def test_arclen():
    glc = sim.TestLC.make_hard_gauss()

    aff = xform.SimultaneousLCXform(xform.LinearBandDataXform(50, 1, 1.5, 1))

    glc2 = aff.apply(glc)

    aff2 = summ.opt_alignment(glc, glc2, vb=False, options={'maxiter': 10}, scoretype='arclen')
    del aff2  # unused
