from cta_tools.plotting import preliminary
import pandas as pd
#import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
from astropy.table import vstack
from cta_tools.io import read_lst_dl1, read_mc_dl1, read_sim_info, save_plot_data, read_plot_data
from cta_tools.utils import get_value
import matplotlib
from pyirf.spectral import (
    calculate_event_weights,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
    PowerLaw,
)
import logging
if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages
import click


log = logging.getLogger(__name__)
log.setLevel("INFO")

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
    "gammaness",
    # delta_t
]


data_structure = {k:{"bins":None, "values":None} for k in linx+logx}

def plot(data, feature, keys, logx=True, ax=None, name="", scale=1):
    """
    scale to norm it 
    """
    if not ax:
        fig, ax = plt.subplots()
    for key in keys:
        ax.hist(
            data[feature]["bins"][:-1],
            bins=data[feature]["bins"],
            weights=data[feature]["values"][key] / scale,
            histtype="step",
            label=name
        )
    ax.set_title(feature)
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.legend()
    return ax


# use after caches have been constructed with plot features
# make this a yaml file or smth for the file lists and spectras to weight to
@click.option("--name", "-n", multiple=True)
@click.option("--input_file", "-f", type=click.Path(exists=True), multiple=True)
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
@click.command()
def main(
name,
    input_file,
    output,
):
    observations_plot_data = {}
    assert len(input_file) >= 1
    for f, n in zip(input_file, name):
        #from IPython import embed; embed()
        observations_plot_data[n] = read_plot_data(f, data_structure)

    figs = []

    # 1 plot absolute, one relative pls
    for feature in logx:
        log.info(feature)
        if not observations_plot_data[name[0]][feature]["values"].empty:
            fig, ax = plt.subplots()
            for n, plot_data in observations_plot_data.items():
                plot(plot_data, feature, ["observations"], logx=True, ax=ax, name=n)
            figs.append(fig)
    for feature in linx:
        log.info(feature)
        if not observations_plot_data[name[0]][feature]["values"].empty:
            fig, ax = plt.subplots()
            for n, plot_data in observations_plot_data.items():
                plot(plot_data, feature, ["observations"], logx=False, ax=ax, name=n)
            figs.append(fig)
 
    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figs:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)


if __name__ == "__main__":
    main()
