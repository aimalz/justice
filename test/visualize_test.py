import justice.simulate as sim

import justice.xform as xform
import justice.visualize as vis


def test_plot_single_lc_color_bands():
    glc = sim.TestLC.make_easy_gauss()
    vis.plot_single_lc_color_bands(glc, "Test Figure")


def test_plot_lcs():
    glc1 = sim.TestLC.make_easy_gauss()
    glc2 = sim.TestLC.make_hard_gauss()

    vis.plot_lcs([glc1, glc2])


def test_plot_arclen_res():
    glc1 = sim.TestLC.make_easy_gauss()
    glc2 = sim.TestLC.make_hard_gauss()

    xform1 = xform.LinearBandDataXform(200, 0, 1, 1)
    lcxf = xform.SameLCXform(xform1)

    vis.plot_arclen_res(glc1, glc2, lcxf)


def test_plot_gp_res():
    glc1 = sim.TestLC.make_easy_gauss()
    glc2 = sim.TestLC.make_hard_gauss()

    vis.plot_gp_res(glc1, glc2)
