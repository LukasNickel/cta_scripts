import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.table import Table
from cta_tools.utils import get_value


def compare_rates(counts1, counts2, bins, l1=None, l2=None):
    """
    should be able to use it to plot rates against time as well with manual bins
    """
    fig = plt.figure()
    gs = fig.add_gridspec(3,1)
    ax_hist = fig.add_subplot(gs[:-1, :])
    ax_hist.set_xscale('log')
    ax_hist.set_yscale('log')
    ax_ratio = fig.add_subplot(gs[-1, :], sharex=ax_hist)
    n1, bins1, patches1 = ax_hist.hist(bins[:-1], weights=counts1, bins=bins, histtype='step', label=l1)
    n2, bins2, patches2 = ax_hist.hist(bins[:-1], weights=counts2, bins=bins, histtype='step', label=l2)
    c = 0.5 * (bins[:-1] + bins[1:])
    r = n1 / n2
    xerr = np.diff(bins)
    if isinstance(c, u.Quantity):
        c = c.to_value()
        xerr = xerr.to_value()
    ax_ratio.errorbar(c, r, xerr=xerr, linestyle="", color="black")
    ax_ratio.axhline(y=1, color="black", linestyle="--", alpha=.3, linewidth=1)
    ax_hist.legend()
    return fig


def hist_feature(data, column, ax=None, bins=None, logx=True, logy=True, label=None):
    if not ax:
        fig, ax = plt.subplots()
    values = data[column]
    if isinstance(data, Table):
        if values.unit:
            values = values.value
        else:
            values = values.data
    values = get_value(data, column)
    if logx:
        if bins is None:
            bins = np.logspace(np.log10(np.nanmin(values)), np.log10(np.nanmax(values)), 30)
        ax.set_xscale("log")
    else:
        if bins is None:
            bins = np.logspace(np.nanmin(values), np.nanmax(values), 30)
    ax.hist(
        values,
        bins,
        histtype="step",
        weights=data["weights"] if "weights" in data.keys() else None,
        label=label
    )
    ax.set_title(column)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    return ax


def plot_binned_time_evolution(binned,  ax=None):
    """
    delta t is wrt the first event of the df.
    This can be tricky when comparing different runs! find a better solution!
    index obs id maybe
    also seperate tels?
    """
    ax.errorbar(
        binned["center"],
        binned["mean"],
        xerr=0.5 * binned["width"],
        yerr=binned["std"],
        # label=label,
        linestyle="",
    )
    ax.set_xlabel("time w.r.t. the first event [s]")
    return ax
