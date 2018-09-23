"""Some handy diagnostic plots"""

import matplotlib.pyplot as plt

from justice import xform

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


def plot_single_lc_color_bands(lc, title, figsize=(10, 5), colors=None):
    """Plot helper, usually call this from iPython.

    :param lc: LC object.
    :param title: string title
    :param figsize: Figure size tuple.
    :param colors: Tuple of colors for each band.
    """

    if colors is None:
        colors = _default_colors[0:len(lc.bands)]

    fig = plt.figure(figsize=figsize)
    plt.title(title)

    if len(colors) != len(lc.bands):
        raise ValueError(
            "Have {} colors but {} bands".format(
                len(colors), len(
                    lc.bands)))

    for band, color in zip(lc.bands, colors):
        plt.errorbar(
            lc.bands[band].time,
            lc.bands[band].flux,
            yerr=lc.bands[band].flux_err,
            fmt='o',
            color=color
        )
    return fig


def plot_lcs(lcs, save=None):
    # This needs a way to have names of the bands, but it works for now.
    if not isinstance(lcs, list):
        lcs = [lcs]
    fig = plt.figure()
    numbands = lcs[0].nbands
    bands = lcs[0].bands
    for i, b in enumerate(bands):
        plt.subplot(numbands, 1, i + 1)
        for lci in lcs:
            plt.errorbar(
                lci.bands[b].time,
                lci.bands[b].flux,
                yerr=lci.bands[b].flux_err,
                linestyle='None',
                marker='.'
            )
    plt.xlabel('time')
    plt.ylabel('flux')
    if isinstance(save, str):
        plt.savefig(save, dpi=250)
    return (fig)


def plot_arclen_res(lca, lcb, xforma, save=None):
    fig = plt.figure()
    lcc = xforma.transform(lcb)
    lcd = lca + lcc
    numbands = lca.nbands
    for i, b in enumerate(lca.bands):
        lcab = lca.bands[b]
        lcbb = lcb.bands[b]
        lccb = lcc.bands[b]
        lcdb = lcd.bands[b]
        plt.subplot(numbands, 1, i + 1)
        plt.errorbar(lcab.time, lcab.flux, yerr=lcab.flux_err, label='reference')
        plt.errorbar(lcbb.time, lcbb.flux, yerr=lcbb.flux_err, label='proposal')
        plt.errorbar(lccb.time, lccb.flux, yerr=lccb.flux_err, label='transformed')
        plt.errorbar(lcdb.time, lcdb.flux, yerr=lcdb.flux_err, label='merged')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('brightness')
    if isinstance(save, str):
        plt.savefig(save, dpi=250)
    return fig


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
    return fig
