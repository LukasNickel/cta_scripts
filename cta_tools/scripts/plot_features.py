from cta_tools.plotting import preliminary
import re
import pandas as pd
#import click
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
from astropy.table import vstack, join
from cta_tools.io import (
    read_lst_dl1,
    read_lst_dl2,
    read_mc_dl1,
    read_mc_dl2,
    read_sim_info,
    save_plot_data,
    read_plot_data,
)
from cta_tools.utils import get_value
import matplotlib
from pyirf.spectral import (
    calculate_event_weights,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
    CRAB_MAGIC_JHEAP2015,
    PowerLaw,
)
import logging
if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages
import click
from cta_tools.logging import setup_logging


log = setup_logging()
plt.rcParams.update({'figure.max_open_warning': 0})


logx = [
    "hillas_intensity",
    "intensity",
    "reco_energy",
]
linx = [
    "hillas_length",
    "hillas_width",
    "hillas_skewness",
    "hillas_kurtosis",
    "length",
    "width",
    "skewness",
    "kurtosis",
    "timing_slope",
    "timing_intercept",
    "time_gradient",
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
    "gh_score",
    # delta_t
]



data_structure = {k:{"bins":None, "values":None} for k in linx+logx}


def load_run(path):
    data = read_lst_dl1(path)
    try:
        dl2 = read_lst_dl2(path)
        data = join(data, dl2, keys=["obs_id", "event_id"], table_names=["dl1", "dl2"])
        for c in list(data.columns):
            if c.endswith("dl1"):
                data.rename_column(c, re.sub("_dl1", "", c))
            elif c.endswith("dl2"):
                del data[c]
            continue
    except:
        log.info("Only using dl1 data")
    log.info(f"Loading of {path} finished")
    log.info(f"{len(data)} events")
    return data


def load_mc(path, obstime, spectrum):
    sim_info = read_sim_info(path)
    data = read_mc_dl1(path)
    try:
        dl2 = read_mc_dl2(path)
        data = join(data, dl2, keys=["obs_id", "event_id"], table_names=["dl1", "dl2"])
        for c in list(data.columns):
            if c.endswith("dl1"):
                data.rename_column(c, re.sub("_dl1", "", c))
            elif c.endswith("dl2"):
                del data[c]
            continue
    except:
        log.info("Only using dl1 data")
    data["weights"] = calculate_event_weights(
        data["true_energy"],
        spectrum,
        PowerLaw.from_simulation(sim_info, obstime),
    )
    log.info(f"Loading of {path} finished")
    log.info(f"{len(data)} events")
    return data


def plot(data, feature, keys, logx=True):
    fig, ax = plt.subplots()
    log.info(f"Creating plot with keys: {keys} for feature {feature}")
    for key in keys:
        ax.hist(
            data[feature]["bins"][:-1],
            bins=data[feature]["bins"],
            weights=data[feature]["values"][key],
            histtype="step",
            label=key
        )
    ax.set_title(feature)
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.legend()
    return fig


# make this a yaml file or smth for the file lists and spectras to weight to
@click.argument(
    "input_files",
    nargs=-1,
)
@click.option("--protons", "-p")
@click.option("--electrons", "-e")
@click.option("--source_gammas", "-g")
@click.option("--binsizes_config", "-b")
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
@click.command()
def main(
    input_files,
    protons,
    electrons,
    source_gammas,
    binsizes_config,
    output,
):
    observations = {}
    if binsizes_config:
        with open(binsizes_config) as f:
            binsizes = yaml.safe_load(f)
    else:
        binsizes = None

    cache = Path(output).with_suffix(".h5")
    if cache.exists():
        plot_data = read_plot_data(cache, data_structure)
    else:
        plot_data = data_structure.copy()
        obstime = 0 * u.s
        runs = [load_run(f) for f in input_files]
        for run in runs:
            observations[run[0]["obs_id"]] = run
            run_time = (run["time"][-1] - run["time"][0]).to(u.s)
            obstime += run_time

        combined = vstack(list(observations.values()))

        if protons:
            protons = load_mc(protons, obstime, IRFDOC_PROTON_SPECTRUM)
        if electrons:
            electrons = load_mc(electrons, obstime, IRFDOC_ELECTRON_SPECTRUM)
        if protons and electrons:
            background = vstack([protons, electrons])
        elif protons:
            background = protons
        elif electrons:
            background = electrons
        else:
            background = None

        if source_gammas:
            gammas = load_mc(source_gammas, obstime, CRAB_MAGIC_JHEAP2015)
        else:
            gammas = None

        for feature in logx:
            plot_data[feature]["bins"] = pd.Series(dtype=np.float64)
            plot_data[feature]["values"] = pd.DataFrame()
            if feature not in combined.keys():
                log.debug(f"{feature} missing in keys: {combined.keys()}")
                continue

            if binsizes:
                if feature in binsizes:
                    min_, max_, n = binsizes[feature]
                else:
                    min_ = get_value(combined, feature).min()
                    max_ = get_value(combined, feature).max()
                    n = 30
            else:
                min_ = get_value(combined, feature).min()
                max_ = get_value(combined, feature).max()
                n = 30
            bins = np.logspace(
                np.log10(min_),
                np.log10(max_),
                n
            )
            plot_data[feature]["bins"] = pd.Series(bins)
            feature_df = pd.DataFrame()
            feature_df["observations"], _ = np.histogram(combined[feature], bins=bins)
            if background:
                feature_df["background"], _ = np.histogram(
                    background[feature],
                    weights = background["weights"],
                    bins=bins
                )
            if gammas:
                feature_df["gammas"], _ = np.histogram(
                    gammas[feature],
                    weights = gammas["weights"],
                    bins=bins
                )
            for run_data in runs:
                run_id = run_data[0]["obs_id"]
                log.info(f"Filling feature df for run {run_id}")
                # why would that happen?
                if feature not in run_data.keys():
                    log.debug(f"{feature} missing in keys: {run_data.keys()}")
                    continue
                feature_df[run_id], _ = np.histogram(
                    run_data[feature],
                    bins=bins
                )
            plot_data[feature]["values"] = feature_df

        for feature in linx:
            plot_data[feature]["bins"] = pd.Series(dtype=np.float64)
            plot_data[feature]["values"] = pd.DataFrame()
            if feature not in combined.keys():
                log.debug(f"{feature} missing in keys: {combined.keys()}")
                continue
            if binsizes:
                if feature in binsizes:
                    min_, max_, n = binsizes[feature]
                else:
                    min_ = get_value(combined, feature).min()
                    max_ = get_value(combined, feature).max()
                    n = 30
            else:
                min_ = get_value(combined, feature).min()
                max_ = get_value(combined, feature).max()
                n = 30
            bins = np.linspace(
                min_,
                max_,
                n
            )
            plot_data[feature]["bins"] = pd.Series(bins)
            feature_df = pd.DataFrame()
            feature_df["observations"], _ = np.histogram(combined[feature], bins=bins)
            if background:
                feature_df["background"], _ = np.histogram(
                    background[feature],
                    weights = background["weights"],
                    bins=bins
                )
            if gammas:
                feature_df["gammas"], _ = np.histogram(
                    gammas[feature],
                    weights = gammas["weights"],
                    bins=bins
                )
            for run_data in runs:
                run_id = run_data[0]["obs_id"]
                log.debug(f"{feature} missing in keys: {run_data.keys()}")
                # why would that happen?
                if feature not in run_data.keys():
                    log.warning(f"{feature} missing in keys: {run_data.keys()}")
                    continue
                feature_df[run_id], _ = np.histogram(
                    run_data[feature],
                    bins=bins
                )
            plot_data[feature]["values"] = feature_df

        save_plot_data(cache, plot_data)

    figs = []

    datamc = ["observations"]       
    if background:
        datamc.append("background")
    if gammas:
        datamc.append("gammas")
    log.info(f"datamc: {datamc}")
    log.info(f"all: {set(plot_data[feature]['values'].keys())}")
    log.info(f"runs: {set(plot_data[feature]['values'].keys()) - set(datamc)}")
    for feature in logx:
        log.info(feature)
        if not plot_data[feature]["values"].empty:
            figs.append(plot(plot_data, feature, datamc, logx=True))
            figs.append(
                plot(
                    plot_data,
                    feature,
                    set(plot_data[feature]["values"].keys()) - set(datamc),
                    logx=True
                )
            )

    for feature in linx:
        log.info(feature)
        if not plot_data[feature]["values"].empty:
            figs.append(plot(plot_data, feature, datamc, logx=False))
            figs.append(
                plot(
                    plot_data,
                    feature,
                    set(plot_data[feature]["values"].keys()) - set(datamc),
                    logx=False
                )
            )

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figs:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)


if __name__ == "__main__":
    main()
