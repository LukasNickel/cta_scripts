import matplotlib.pyplot as plt
import numpy as np


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

    for feature in logx:
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        for j, (key, dataset) in enumerate(data.items()):
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
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        for j, (key, dataset) in enumerate(data.items()):
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
