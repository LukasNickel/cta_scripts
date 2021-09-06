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
from collections import defaultdict
if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


data_structure = {k:{"bins":None, "values":None} for k in ["pixels", "datamc", "time", "surviving"]}



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

        if proton_file:
            proton_sim_info = read_sim_info(proton_file)
            protons = read_mc_dl1(proton_file, drop_nans=False, images=True)

            protons["weights"] = calculate_event_weights(
                protons["true_energy"],
                IRFDOC_PROTON_SPECTRUM,
                PowerLaw.from_simulation(proton_sim_info, 1*u.s),
            )
        if electron_file:
            electron_sim_info = read_sim_info(electron_file)
            electrons = read_mc_dl1(electron_file, drop_nans=False, images=True)
            electrons["weights"] = calculate_event_weights(
                electrons["true_energy"],
                IRFDOC_ELECTRON_SPECTRUM,
                PowerLaw.from_simulation(electron_sim_info, 1*u.s),
            )
        if protons and electrons:
            background = vstack([protons, electrons])
        elif protons:
            background = protons
        elif electrons:
            background = electrons
        else:
            background = None

        max_ = np.percentile(combined["image"], 99)
        min_ = np.percentile(combined["image"], 1)
        
        pixel_values = defaultdict(list)
        for pixel, values in enumerate(combined["image"].T) :
            pixel_values["std"].append(np.std(values))
            pixel_values["mean"].append(np.mean(values))
            pixel_values["median"].append(np.median(values))
            per_25 = np.percentile(values, 25)
            per_75 = np.percentile(values, 75)
            iqr = per_75 - per_25
            pixel_values["25"].append(per_25)
            pixel_values["75"].append(per_75)
            pixel_values["iqr"].append(iqr)
            pixel_values["min"].append(np.min(values))
            pixel_values["max"].append(np.max(values))
        plot_values["pixels"] = {
            "bins": pd.Series(),
            "values": pd.DataFrame(pixel_values),
        }
        if background:
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
        
        # fraction surviving events
        combined["size"] = combined["image"].sum(axis=1)
        combined["survived"] = combined["image_mask"].sum(axis=1) >= 0
        size_bins = np.logspace(
            np.log10(1),
            np.log10(combined["size"].max()),
            100
        )
        fraction_df = pd.DataFrame()
        fraction = []
        indices = np.digitize(combined["size"], size_bins)
        for b in range(100):
            bin_index=b+1
            group = combined[indices == bin_index]
            fraction.append(np.mean(group["survived"]))
        fraction_df["data"] = fraction #np.histogram(combined["survived"], bins=size_bins)
        if background:
            background["size"] = background["image"].sum(axis=1)
            background["survived"] = background["image_mask"].sum(axis=1) >= 0
            fraction_mc = []
            indices = np.digitize(background["size"], size_bins)
            for b in range(100):
                bin_index=b+1
                group = background[indices == bin_index]
                fraction_mc.append(np.mean(group["survived"]))
            fraction_df["mc"] = fraction_mc #np.histogram(combined["survived"], bins=size_bins)

        plot_values["surviving"] = {
            "bins": pd.Series(size_bins),
            "values": fraction_df,
        }

        save_plot_data(cache, plot_values)

    return plot_values


@click.argument(
    "input_files",
    nargs=-1,
)
@click.option("--protons", "-p", type=click.Path(exists=True), default=False)
@click.option("--electrons", "-e", type=click.Path(exists=True), default=False)
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
    ax.plot(
        plot_data["pixels"]["values"].index, 
        plot_data["pixels"]["values"]["median"],
        "k.",
    )
    ax.plot(
        plot_data["pixels"]["values"].index, 
        plot_data["pixels"]["values"]["25"],
        "r--",
    )
    ax.plot(
        plot_data["pixels"]["values"].index, 
        plot_data["pixels"]["values"]["75"],
        "r--",
    )
    ax.plot(
        plot_data["pixels"]["values"].index, 
        plot_data["pixels"]["values"]["max"],
        "b.",
    )
    ax.plot(
        plot_data["pixels"]["values"].index, 
        plot_data["pixels"]["values"]["min"],
        "g.",
    )
    ax.set_ylim(
        min(plot_data["pixels"]["values"]["25"] - 3*plot_data["pixels"]["values"]["iqr"]),
        max(plot_data["pixels"]["values"]["75"] + 3*plot_data["pixels"]["values"]["iqr"]),
    )
    figs.append(fig)
    
    # if mc is there!
    # datamc
    fig = compare_rates(
        plot_data["datamc"]["values"]["data"].values,
        plot_data["datamc"]["values"]["mc"].values,
        plot_data["datamc"]["bins"].values,
        l1="data",
        l2="mc",
    )
    # title and axes labels
    figs.append(fig)

    # time
    fig, ax = plt.subplots()
    plot_binned_time_evolution(plot_data["time"]["values"], ax=ax)
    ax.set_title("Mean charge over time")
    ax.set_ylabel("charge [pe]")
    figs.append(fig)


    # fraction
    fig, ax = plt.subplots()
    ax.plot(
        plot_data["surviving"]["bins"],#["center"], 
        plot_data["surviving"]["values"]["data"],
        label="data",
    )
    if "mc" in plot_data["surviving"]["values"].keys():
        ax.plot(
            plot_data["surviving"]["bins"],#["center"], 
            plot_data["surviving"]["values"]["data"],
            label="mc",
        )
    ax.legend()
    ax.set_xscale("log")
    ax.set_title("Fraction surviving events")
    ax.set_xlabel("Size")
    figs.append(fig)



    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figs:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)
