"""Some handy diagnostic plots"""

import matplotlib.pyplot as plt

from util import transform, merge

# would like to have these pass axes between each other to combine what's being plotted
# all untested for now!

def plot_lcs(lcs):
    for lci in lcs:
        plt.errorbar(lci.x, lci.y, yerr=lci.yerr, linestyle='None', capsize=0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return

def plot_arclen(lca, lcb, res):
    aff = res.x
    plt.errorbar(lca.x, lca.y, yerr=lca.yerr, label='original', capsize=0)
    plt.errorbar(lcb.x, lcb.y, yerr=lcb.yerr, label='proposal', capsize=0)
    lcc = transform(lcb, aff)
    plt.errorbar(lcc.x, lcc.y, yerr=lcc.yerr, label='transformed', capsize=0)
    lcd = merge(lca, lcc)
    plt.errorbar(lcd.x, lcd.y, yerr=lcd.yerr, label='merged', capsize=0)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return

def plot_gp(lctrain, lctest):
    plt.fill_between(lctest.x, lctest.y-lctest.yerr, lctest.y+lctest.yerr, alpha=0.1)
    plt.errorbar(lctest.x, lctest.y, yerr=lctest.yerr)
    plt.errorbar(lctrain.x, lctrain.y, yerr=lctrain.yerr, linestyle='None', capsize=0)
    plt.show()
    plt.xlabel('x')
    plt.ylabel('y')
    return
