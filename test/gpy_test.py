import pytest

import justice.simulate as sim
import justice.summarize as summ
from justice import xform


@pytest.mark.no_precommit
def test_gpy():
    glc = sim.TestLC.make_hard_gauss()

    aff = xform.LinearBandDataXform([50., 1., 1.5, 1.])
    lcxf = xform.SimultaneousLCXform(aff)
    glc2 = lcxf.apply(glc)

    aff2 = summ.opt_alignment(glc, glc2, vb=False, options={'maxiter': 10}, scoretype='gp')
    del aff2  # unused


@pytest.mark.no_precommit
def test_gpy_with_overlap_cost():
    glc = sim.TestLC.make_hard_gauss()

    aff = xform.LinearBandDataXform([50., 1., 1.5, 1.])
    lcxf = xform.SimultaneousLCXform(aff)
    glc2 = lcxf.apply(glc)

    overlap_cost_fcn = summ.OverlapCostComponent([1.0, 0.1, 0.0], 1.0)
    aff2 = summ.opt_alignment(
        glc, glc2, vb=False, options={'maxiter': 10},
        overlap_cost_fcn=overlap_cost_fcn, scoretype='gp'
    )
    del aff2  # unused
