import matplotlib.pyplot as plt
import numpy as np
from pyirf.binning import create_bins_per_decade
import astropy.units as u

def plot_dl1(data):
    figures = []

    logx = [
        "hillas_intensity",
    ]
    linx = [
        "hillas_length",
        "hillas_width",
        "hillas_skewness",
        "hillas_kurtosis",
        "timing_slope",
        "timing_intercept",
        "leakage_pixels_width_1",
        "leakage_intensity_width_2",
        "leakage_pixels_width_2",
        "morphology_num_islands",
        "morphology_num_pixels",
        "concentration_cog",
        "concentration_core",
        "concentration_pixel",
        "intensity_max",
        "intensity_min",
        "intensity_mean",
        "intensity_std",
        "intensity_skewness",
        "intensity_kurtosis",
        "peak_time_max",
        "peak_time_min",
        "peak_time_mean",
        "peak_time_std",
        "peak_time_kurtosis",
        "peak_time_skewness",
        # delta_t
    ]

    bins=None
    for feature in logx:
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        for j, (key, dataset) in enumerate(data.items()):
            if feature not in dataset.keys():
                continue
            v = dataset[feature]
            if v.unit:
                v = v.value
            else:
                v = v.data
            if j == 0:
                bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 30)
            ax.hist(
                v,
                bins,
                histtype="step",
                label=key,
                weights=dataset["weights"] if "weights" in dataset.keys() else None,
            )
        ax.set_title(feature)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend()
    for feature in linx:
        print(feature)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        for j, (key, dataset) in enumerate(data.items()):
            print(j, dataset.keys())
            if feature not in dataset.keys():
                print('skip this')
                continue
            v = dataset[feature]
            if v.unit:
                v = v.value
            else:
                v = v.data
            if j == 0:
                bins = np.linspace(v.min(), v.max(), 30)
            ax.hist(
                v,
                bins,
                histtype="step",
                label=key,
                weights=dataset["weights"] if "weights" in dataset.keys() else None,
            )
        ax.set_title(feature)
        ax.set_yscale("log")
        ax.legend()
    return figures
    
    
    
def compare_rates(e1, e2, bins=None, l1=None, l2=None, w1=None, w2=None):
    fig = plt.figure()
    gs = fig.add_gridspec(3,1)
    ax_hist = fig.add_subplot(gs[:-1, :])
    ax_hist.set_xscale('log')
    ax_hist.set_yscale('log')
    ax_ratio = fig.add_subplot(gs[-1, :], sharex=ax_hist)
    if bins is None:
        bins = create_bins_per_decade(50*u.GeV, 10*u.TeV, 5)
    n1, bins1, patches1 = ax_hist.hist(e1, bins=bins, histtype='step', label=l1, weights=w1)
    n2, bins2, patches2 = ax_hist.hist(e2, bins=bins, histtype='step', label=l2, weights=w2)
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
    
    
    
    
    