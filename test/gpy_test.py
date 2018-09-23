import justice.simulate as sim
import justice.summarize as summ
from justice import xform


def test_gpy():
    glc = sim.TestLC.make_hard_gauss()

    aff = glc.get_xform([50., 1., 1., 1.5, 1., 1.])
    glc2 = xform.transform(glc, aff)

    aff2 = summ.opt_gp(glc, glc2, vb=False, options={'maxiter': 10})


def test_gpy_with_overlap_cost():
    glc = sim.TestLC.make_hard_gauss()

    aff = glc.get_xform([50., 1., 1., 1.5, 1., 1.])
    glc2 = xform.transform(glc, aff)

    overlap_cost_fcn = summ.OverlapCostComponent([1.0, 0.1, 0.0], 1.0)
    aff2 = summ.opt_gp(
        glc, glc2, vb=False, options={'maxiter': 10}, overlap_cost_fcn=overlap_cost_fcn
    )
