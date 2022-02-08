from cta_tools.plotting import preliminary
import re
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
from lstchain.io import read_data_dl2_to_QTable, read_mc_dl2_to_QTable
from astropy.table import vstack, join
from astropy.time import Time
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
from cta_tools.plotting.features import compare_datasets
import matplotlib
from pyirf.spectral import (
    calculate_event_weights,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
    CRAB_MAGIC_JHEAP2015,
    PowerLaw,
)
from cta_tools.logging import setup_logging


log = setup_logging()
if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages
import click
plt.rcParams.update({'figure.max_open_warning': 0})


logx = [
    "reco_energy",
    "intensity",
]
linx = [
    "pointing_alt",
    "pointing_az",
    "reco_alt",
    "reco_az",
    "gh_score",
    "length",
    "width",
    "skewness",
    "kurtosis",
    "time_gradient"
]

data_structure = {k:{"bins":None, "values":None} for k in linx+logx}


def load_run(path):
    data = read_data_dl2_to_QTable(path)
    data["time"] = Time(data["dragon_time"], format="mjd", scale="tai")
    log.info(f"Loading of {path} finished")
    log.info(f"{len(data)} events")
    return data


def load_mc(path, obstime, spectrum):
    data, sim_info= read_mc_dl2_to_QTable(path)
    log.info(f"Loading of {path} finished")
    log.info(f"{len(data)} events")
    log.info(f"Reweighting from {sim_info.n_showers} showers to obstime: {obstime.to(u.min):.2f}")
    data["weights"] = calculate_event_weights(
        data["true_energy"],
        spectrum,
        PowerLaw.from_simulation(sim_info, obstime),
    )
    return data



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
            log.info(f"Observation time: {run_time.to(u.min):.2f}")
            obstime += run_time
        log.info(f"Combined observation time: {obstime.to(u.min):.2f}")

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

            log.info(feature)
            log.info(combined[feature])
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
            log.info(min_)
            log.info(max_)
            bins = np.logspace(
                np.log10(min_),
                np.log10(max_),
                n
            )
            plot_data[feature]["bins"] = pd.Series(bins)
            feature_df = pd.DataFrame()
            feature_df["observations"], _ = np.histogram(get_value(combined, feature), bins=bins)
            if background:
                feature_df["background"], _ = np.histogram(
                        get_value(background, feature),
                    weights = background["weights"],
                    bins=bins
                )
            if gammas:
                feature_df["gammas"], _ = np.histogram(
                    get_value(gammas, feature),
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
                    get_value(run_data, feature),
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
            feature_df["observations"], _ = np.histogram(get_value(combined, feature), bins=bins)
            if background:
                feature_df["background"], _ = np.histogram(
                        get_value(background, feature),
                    weights = background["weights"],
                    bins=bins
                )
            if gammas:
                feature_df["gammas"], _ = np.histogram(
                    get_value(gammas, feature),
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
                    get_value(run_data, feature),
                    bins=bins
                )
            plot_data[feature]["values"] = feature_df
        save_plot_data(cache, plot_data)

    figs = []

    datamc = ["observations"]       
    if protons or electrons:
        datamc.append("background")
    if source_gammas:
        datamc.append("gammas")
    for feature in logx:
        log.info(feature)
        if not plot_data[feature]["values"].empty:
            figs.append(compare_datasets(plot_data, feature, datamc, logx=True))
            figs.append(
                compare_datasets(
                    plot_data,
                    feature,
                    set(plot_data[feature]["values"].keys()) - set(datamc),
                    logx=True
                )
            )

    for feature in linx:
        log.info(feature)
        if not plot_data[feature]["values"].empty:
            figs.append(compare_datasets(plot_data, feature, datamc, logx=False))
            figs.append(
                compare_datasets(
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
