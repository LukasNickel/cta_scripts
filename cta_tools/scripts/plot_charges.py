from cta_tools.plotting import preliminary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
from astropy.table import vstack
from cta_tools.io import read_lst_dl1, read_mc_dl1, read_sim_info, save_plot_data, read_plot_data
from cta_tools.utils import get_value
from cta_tools.plotting.features import plot_binned_time_evolution, compare_rates
import matplotlib
import click
from tqdm import tqdm
import logging
from pyirf.spectral import (
    calculate_event_weights,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
    PowerLaw,
)
if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


data_structure = {k:{"bins":None, "values":None} for k in ["pixels", "datamc", "time"]}



def load_data(files, cache):
    if cache.exists():
        plot_values = read_plot_data(cache, data_structure)
    else:
        proton_file = files["protons"]
        electron_file = files["electrons"]
        observation_files = files["observations"]
        plot_values = {}
        obstime = 0 * u.s
        observations = {}
        for f in tqdm(observation_files):
            data = read_lst_dl1(f, images=True, drop_nans=False)
            observations[data[0]["obs_id"]] = data
            run_time = (data["time"][-1] - data["time"][0]).to(u.s)
            obstime += run_time
        combined = vstack(list(observations.values()))
        combined["weights"] = 1 / obstime.to_value(u.s)

        proton_sim_info = read_sim_info(proton_file)
        protons = read_mc_dl1(proton_file, drop_nans=False, images=True)

        protons["weights"] = calculate_event_weights(
            protons["true_energy"],
            IRFDOC_PROTON_SPECTRUM,
            PowerLaw.from_simulation(proton_sim_info, 1*u.s),
        )
        electron_sim_info = read_sim_info(electron_file)
        electrons = read_mc_dl1(electron_file, drop_nans=False, images=True)
        electrons["weights"] = calculate_event_weights(
            electrons["true_energy"],
            IRFDOC_ELECTRON_SPECTRUM,
            PowerLaw.from_simulation(electron_sim_info, 1*u.s),
        )
        background = vstack([protons, electrons])

        ## pixels
        # This plot is not super helpful, can we do better?
        max_ = np.percentile(combined["image"], 99)
        min_ = np.percentile(combined["image"], 1)
        bins = np.linspace(min_, max_, 30)
        count_df = pd.DataFrame()
        for pixel, values in enumerate(combined["image"].T) :
            count_df[pixel], _ = np.histogram(values, bins=bins)
        plot_values["pixels"] = {
            "bins": pd.Series(bins),
            "values": count_df,
        }

        ## datamc
        datamc_bins = np.linspace(
            min(min_, np.percentile(background["image"], 1)),
            max(max_, np.percentile(background["image"], 99)),
            30,
        )
        datamc_df = pd.DataFrame()
        datamc_df["data"], _ = np.histogram(combined["image"], bins=datamc_bins)
        datamc_df["mc"], _ = np.histogram(background["image"], bins=datamc_bins)
        plot_values["datamc"] = {
            "bins": pd.Series(datamc_bins),
            "values": datamc_df,
        }

        ## time
        combined["delta_t_sec"] = (combined["time"] - combined["time"][0]).sec
        last = combined["delta_t_sec"].max()
        time_bins = np.linspace(0, last, 20)
        # pandas cant work with the image columns, so we build the binned df manually
        indices = np.digitize(combined["delta_t_sec"], time_bins)
        mean = []
        std = []
        # This loses only the very last event and should be the way to go hopefully
        # basically since the bins are bin edges, np.digitize will always have just one event in the last bin
        # (with the default right=False, otherwise the same applies for the first bin)
        for bin_index in np.unique(indices)[:-1]:
            group = combined[indices == bin_index]
            mean.append(np.mean(group["image"]))
            std.append(np.std(group["image"]))
        time_df = pd.DataFrame({
            "center": 0.5 * (time_bins[:-1] + time_bins[1:]),
            "width": np.diff(time_bins),
            "mean": mean,
            "std": std,
        })
        
        plot_values["time"] = {
            "bins": pd.Series(time_bins),
            "values": time_df,
        }
        save_plot_data(cache, plot_values)
    return plot_values





@click.argument(
    "input_files",
    nargs=-1,
)
@click.option("--protons", "-p", type=click.Path(exists=True))
@click.option("--electrons", "-e", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
@click.command()
def main(
    input_files,
    protons,
    electrons,
    output,
):

    cache = Path(output).with_suffix(".h5")
    plot_data = load_data({"observations": input_files, "protons": protons, "electrons": electrons}, cache)
    
    figs = []
    # pixels
    fig, ax = plt.subplots()
    for p in plot_data["pixels"]["values"].keys():
        ax.hist(
            plot_data["pixels"]["bins"][:-1],
            bins=plot_data["pixels"]["bins"],
            weights=plot_data["pixels"]["values"][p],
        )
    figs.append(fig)
    
    # datamc
    fig = compare_rates(
        plot_data["datamc"]["values"]["data"].values,
        plot_data["datamc"]["values"]["mc"].values,
        plot_data["datamc"]["bins"].values,
    )
    figs.append(fig)

    # time
    fig, ax = plt.subplots()
    plot_binned_time_evolution(plot_data["time"]["values"], ax=ax)
    figs.append(fig)
    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figs:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)
