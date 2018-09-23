"""Some handy diagnostic plots"""

import matplotlib.pyplot as plt

from justice.xform import transform
#from justice.lightcurve import merge

# would like to have these pass axes between each other to combine what's being plotted
# also want to accommodate multiple filters/bands of y


def setup_plot():
    pass


def wrapup_plot():
    pass


_default_colors = (
    '#800000',
    '#ff6600',
    '#330000',
    '#00ff00',
)


def plot_single_lc_color_bands(lc, title, figsize=(10, 5), colors=_default_colors):
    """Plot helper, usually call this from iPython.

    :param lc: LC object.
    :param title: string title
    :param figsize: Figure size tuple.
    :param colors: Tuple of colors for each band.
    """
    fig = plt.figure(figsize=figsize)
    plt.title(title)

    if len(colors) != len(lc.bands)
        raise ValueError(
            "Have {} colors but {} bands".format(
                len(colors), len(lc.bands))

    for band, color in zip(lc.bands, colors):
        plt.errorbar(
            lc.x[:, band_idx],
            lc.y[:, band_idx],
            yerr=lc.yerr[:, band_idx],
            fmt='o',
            color=color
        )
    return fig


def plot_lcs(lcs, save=None):
    # This needs a way to have names of the bands, but it works for now.
    if not isinstance(lcs, list):
        lcs = [lcs]
    fig = plt.figure()
    numbands = lcs[0].x.shape[1]
    for i in range(numbands):
        plt.subplot(numbands, 1, i + 1)
        for lci in lcs:
            plt.errorbar(
                lci.x[:, i],
                lci.y[:, i],
                yerr=lci.yerr[:, i],
                linestyle='None',
                marker='.'
            )
    plt.xlabel('time')
    plt.ylabel('brightness')
    if isinstance(save, str):
        plt.savefig(save, dpi=250)
    return (fig)


def plot_arclen_res(lca, lcb, xform, save=None):
    fig = plt.figure()
    lcc = transform(lcb, xform)
    lcd = merge(lca, lcc)
    numbands = lca.x.shape[1]
    for i in range(numbands):
        plt.subplot(numbands, 1, i + 1)
        plt.errorbar(lca.x[:, i], lca.y[:, i], yerr=lca.yerr[:, i], label='reference')
        plt.errorbar(lcb.x[:, i], lcb.y[:, i], yerr=lcb.yerr[:, i], label='proposal')
        plt.errorbar(lcc.x[:, i], lcc.y[:, i], yerr=lcc.yerr[:, i], label='transformed')
        plt.errorbar(lcd.x[:, i], lcd.y[:, i], yerr=lcd.yerr[:, i], label='merged')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('brightness')
    if isinstance(save, str):
        plt.savefig(save, dpi=250)
    return (fig)


def plot_gp_res(lctrain, lcpred, save=None):
    fig = plt.figure()
    numbands = lctrain.x.shape[1]
    for i in range(numbands):
        plt.subplot(numbands, 1, i + 1)
        plt.fill_between(
            lcpred.x[:, i],
            lcpred.y[:, i] - lcpred.yerr[:, i],
            lcpred.y[:, i] + lcpred.yerr[:, i],
            alpha=0.1
        )
        plt.plot(lcpred.x[:, i], lcpred.y[:, i])
        plt.errorbar(
            lctrain.x[:, i],
            lctrain.y[:, i],
            yerr=lctrain.yerr[:, i],
            linestyle='None',
            marker='.'
        )
    plt.xlabel('time')
    plt.ylabel('brightness')
    if isinstance(save, str):
        plt.savefig(save, dpi=250)
    return (fig)
