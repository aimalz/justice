"""Some handy diagnostic plots"""
from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt

import astropy.stats

# would like to have these pass axes between each other to combine what's being plotted
# also want to accommodate multiple filters/bands of y
from justice.features import period_distribution


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


def plot_lcs(
    lcs,
    *,
    save=None,
    plot_period=False,
    title=None,
    period_transform: period_distribution.LsTransformBase = None
):
    """Plot multiple (or single) lightcurves at once

    :param lcs: list of lightcurves
    :param save: boolean to save or not
    :param plot_period: Whether to plot periods alongside light curves.
    :param period: Period scale. Defaults to a np.linspace.
    :param Period_transform: Transformation class.
    :return: figure object
    """
    # This needs a way to have names of the bands, but it works for now.
    if not isinstance(lcs, list):
        lcs = [lcs]
    fig = plt.figure(figsize=(15, 18))

    numbands = lcs[0].nbands
    bands = lcs[0].bands

    period_per_lc: List[period_distribution.MultiBandPeriod] = []
    if plot_period:
        period_per_lc = list(map(period_transform.apply, lcs))

    fig, ax = plt.subplots(
        nrows=numbands,
        ncols=(2 if plot_period else 1),
        sharex='col',
        # sharey='row',
        figsize=((12 if plot_period else 8), 6),
        squeeze=False
    )

    if title:
        fig.suptitle(title)

    for i, b in enumerate(bands):
        for lci in lcs:
            detec = lci.bands[b].detected
            detected_plot = ax[i, 0].errorbar(
                lci.bands[b].time[detec == 1],
                lci.bands[b].flux[detec == 1],
                yerr=lci.bands[b].flux_err[detec == 1],
                linestyle='None',
                marker='.'
            )
            not_detected_plot = ax[i, 0].errorbar(
                lci.bands[b].time[detec == 0],
                lci.bands[b].flux[detec == 0],
                yerr=lci.bands[b].flux_err[detec == 0],
                linestyle='None',
                alpha=.2,
                marker='.'
            )
            plt.legend([detected_plot, not_detected_plot], ["detected", "not detected"],
                       loc="upper right")

        if i == numbands - 1:
            ax[i, 0].set_xlabel('time')  # Only set on bottom plot.
            if plot_period:
                ax[i, 1].set_xlabel('period')

        if plot_period:
            for multi_band_period in period_per_lc:
                ax[i, 1].plot(multi_band_period.period, multi_band_period[b])

    plt.ylabel(b).set_rotation(0)
    plt.xlabel('time')
    plt.subplots_adjust(hspace=0.4)
    if title:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        plt.tight_layout()
    if isinstance(save, str):
        plt.savefig(save, dpi=250)
    return fig


def plot_arclen_res(lca, lcb, xforma, save=None):
    """
    Plot the result of a trial merger for arclen

    :param lca: Original lightcurve
    :param lcb: Lightcurve to merge
    :param xforma: Transform to show
    :param save: save fig or not
    :return: figure
    """
    fig = plt.figure(figsize=(10, 10))
    lcc = xforma.apply(lcb)
    fig = plt.figure()
    lcc = xforma.apply(lcb)
    lcd = lca + lcc
    numbands = lca.nbands
    for i, b in enumerate(lca.bands):
        lcab = lca.bands[b]
        lcbb = lcb.bands[b]
        lccb = lcc.bands[b]
        lcdb = lcd.bands[b]
        plt.subplot(numbands, 1, i + 1)
        plt.errorbar(
            lcab.time,
            lcab.flux,
            yerr=lcab.flux_err,
            fmt='+',
            label='reference')
        plt.errorbar(lcbb.time, lcbb.flux, yerr=lcbb.flux_err, fmt='+', label='proposal')
        plt.errorbar(
            lccb.time, lccb.flux, yerr=lccb.flux_err, fmt='+', label='transformed'
        )
        plt.errorbar(
            lcdb.time, lcdb.flux, yerr=lcdb.flux_err, fmt='-', label='merged', alpha=.3
        )
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('brightness')
    if isinstance(save, str):
        plt.savefig(save, dpi=250)
    return fig


def plot_gp_res(lctrain, lcpred, save=None):
    """
    Plots the results of a Gaussian Process fit

    :param lctrain: The light curve trained on
    :param lcpred: Predicted light curve
    :param save: save figure or not
    :return: figure
    """
    fig = plt.figure()
    numbands = lctrain.nbands
    for i, b in enumerate(lctrain.bands):
        plt.subplot(numbands, 1, i + 1)
        plt.fill_between(
            lcpred.bands[b].time,
            lcpred.bands[b].flux - lcpred.bands[b].flux_err,
            lcpred.bands[b].flux + lcpred.bands[b].flux_err,
            alpha=0.1
        )
        plt.plot(lcpred.bands[b].time, lcpred.bands[b].flux)
        plt.errorbar(
            lctrain.bands[b].time,
            lctrain.bands[b].flux,
            yerr=lctrain.bands[b].flux_err,
            linestyle='None',
            marker='.'
        )
    plt.xlabel('time')
    plt.ylabel('flux')
    if isinstance(save, str):
        plt.savefig(save, dpi=250)
    return fig
