"""Some handy diagnostic plots"""

import matplotlib.pyplot as plt

from justice.affine_xform import transform
from justice.lightcurve import merge

# would like to have these pass axes between each other to combine what's being plotted
# also want to accommodate multiple filters/bands of y


def setup_plot():
    pass


def wrapup_plot():
    pass


def plot_lcs(lcs, save=None):
    #This needs a way to have names of the bands, but it works for now.
    if type(lcs) != list:
        lcs = [lcs]
    fig = plt.figure()
    numbands = len(lcs[0].x)
    for i in range(numbands):
        plt.subplot(numbands, 1, i+1)
        for lci in lcs:
            plt.errorbar(lci.x[i], lci.y[i], yerr=lci.yerr[i], linestyle='None', marker='.')
    plt.xlabel('time')
    plt.ylabel('brightness')
    if type(save) == str:
        plt.savefig(save, dpi=250)
    return(fig)


def plot_arclen_res(lca, lcb, aff, save=None):
    fig = plt.figure()
    lcc = transform(lcb, aff)
    lcd = merge(lca, lcc)
    numbands = len(lca.x)
    for i in range(numbands):
        plt.subplot(numbands, 1, i+1)
        plt.errorbar(lca.x[i], lca.y[i], yerr=lca.yerr[i], label='reference')
        plt.errorbar(lcb.x[i], lcb.y[i], yerr=lcb.yerr[i], label='proposal')
        plt.errorbar(lcc.x[i], lcc.y[i], yerr=lcc.yerr[i], label='transformed')
        plt.errorbar(lcd.x[i], lcd.y[i], yerr=lcd.yerr[i], label='merged')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('brightness')
    if type(save) == str:
        plt.savefig(save, dpi=250)
    return(fig)


def plot_gp_res(lctrain, lcpred, save=None):
    fig = plt.figure()
    numbands = len(lctrain.x)
    for i in range(numbands):
        plt.fill_between(lcpred.x[i], lcpred.y[i] - lcpred.yerr[i], lcpred.y[i] + lcpred.yerr[i], alpha=0.1)
        plt.plot(lcpred.x[i], lcpred.y[i])
        plt.errorbar(lctrain.x[i], lctrain.y[i], yerr=lctrain.yerr[i], linestyle='None', marker='.')
    plt.xlabel('time')
    plt.ylabel('brightness')
    if type(save) == str:
        plt.savefig(save, dpi=250)
    return(fig)
