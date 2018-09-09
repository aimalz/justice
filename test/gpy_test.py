import numpy as np
import justice.simulate as sim
import justice.summarize as summ

import justice.affine_xform as affine

def test_gpy():
    def_cadence = [np.arange(0., 500., 25.),]*2
    gmodel = sim.make_gauss([100., 100.], [300., 600.], [5., 5.], [1., 1.])
    gtimes = sim.make_cadence(def_cadence, [0.5, 0.5])
    gtrue = gmodel(gtimes)
    gphot, gerr = sim.apply_err(gtrue, [0.1, 0.1])

    glc = sim.LC(gtimes, gphot, gerr)

    aff = affine.Aff(50., 1., 1., 1.5)
    glc2 = affine.transform(glc, aff)

    aff2 = summ.opt_gp(glc, glc2, vb=False, options={'maxiter':100})
