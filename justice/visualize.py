"""Some handy diagnostic plots"""

import matplotlib.pyplot as plt

from util import transform, merge

# would like to have these pass axes between each other to combine what's being plotted
# also want to accommodate multiple filters/bands of y

def setup_plot():
    pass

def wrapup_plot():
    passs

def plot_lcs(lcs, save=None):
    if type(lcs) != list:
        lcs = [lcs]
    fig = plt.figure()
    for lci in lcs:
        plt.errorbar(lci.x, lci.y, yerr=lci.yerr, linestyle='None', marker='.')
    plt.xlabel('time')
    plt.ylabel('brightness')
    if type(save) == str:
        plt.savefig(save, dpi=250)
    return(fig)

def plot_arclen_res(lca, lcb, aff, save=None):
    fig = plt.figure()
    plt.errorbar(lca.x, lca.y, yerr=lca.yerr, label='reference')
    plt.errorbar(lcb.x, lcb.y, yerr=lcb.yerr, label='proposal')
    lcc = transform(lcb, aff)
    plt.errorbar(lcc.x, lcc.y, yerr=lcc.yerr, label='transformed')
    lcd = merge(lca, lcc)
    plt.errorbar(lcd.x, lcd.y, yerr=lcd.yerr, label='merged')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('brightness')
    if type(save) == str:
        plt.savefig(save, dpi=250)
    return(fig)

def plot_gp_res(lctrain, lcpred, save=None):
    fig = plt.figure()
    plt.fill_between(lcpred.x, lcpred.y-lcpred.yerr, lcpred.y+lcpred.yerr, alpha=0.1)
    plt.plot(lcpred.x, lcpred.y)
    plt.errorbar(lctrain.x, lctrain.y, yerr=lctrain.yerr, linestyle='None', marker='.')
    plt.xlabel('time')
    plt.ylabel('brightness')
    if type(save) == str:
        plt.savefig(save, dpi=250)
    return(fig)
