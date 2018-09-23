import numpy as np
import justice.simulate as sim
import justice.summarize as summ

import justice.xform as xform
import justice.visualize as vis

def test_plot_single_lc_color_bands():
    glc = sim.TestLC.make_easy_gauss()
    vis.plot_single_lc_color_bands(glc, "Test Figure")
