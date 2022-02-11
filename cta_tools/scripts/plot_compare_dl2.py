import pandas as pd
from astropy.table import vstack
import astropy.units as u
import click
import numpy as np
import yaml
import matplotlib
from pathlib import Path
import logging
from lstchain.io import read_mc_dl2_to_QTable
from pyirf.spectral import (
    calculate_event_weights,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
    CRAB_MAGIC_JHEAP2015,
    PowerLaw,
)
from cta_tools.plotting import preliminary
from cta_tools.cuts import create_mask_selection
from cta_tools.io import (
    save_plot_data,
    read_plot_data,
    read_lst_dl2_runs,
)
from cta_tools.utils import get_value
from cta_tools.plotting.features import compare_datasets
from cta_tools.logging import setup_logging

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages

log = logging.getLogger(__name__)

logx = [
    "reco_energy",
    "intensity",
]


def load_mc(path, obstime, spectrum):
    data, sim_info = read_mc_dl2_to_QTable(path)
    log.info(f"Loading of {path} finished")
    log.info(f"{len(data)} events")
    log.info(f"Weighting from {sim_info.n_showers} showers to: {obstime.to(u.min):.2f}")
    data["weights"] = calculate_event_weights(
        data["true_energy"],
        spectrum,
        PowerLaw.from_simulation(sim_info, obstime),
    )
    return data


@click.argument(
    "input_files",
    nargs=-1,
)
@click.option("--protons", "-p")
@click.option("--electrons", "-e")
@click.option("--source_gammas", "-g")
@click.option("--binsizes_config", "-b")
@click.option("--cuts", "-c")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
@click.command()
def main(
    input_files,
    protons,
    electrons,
    source_gammas,
    binsizes_config,
    cuts,
    verbose,
    output,
):
    setup_logging(verbose=verbose)
    observations = {}
    if binsizes_config:
        with open(binsizes_config) as f:
            binsizes = yaml.safe_load(f)
    else:
        binsizes = None
    if cuts:
        with open(cuts) as f:
            selection = yaml.safe_load(f).get("selection")
    else:
        selection = None

    cache = Path(output).with_suffix(".h5")
    if cache.exists():
        plot_data = read_plot_data(cache)
    else:
        plot_data = {}
        observations, obstime = read_lst_dl2_runs(input_files)
        combined = vstack(list(observations.values()))
        log.info(combined.keys())

        if protons:
            protons = load_mc(protons, obstime, IRFDOC_PROTON_SPECTRUM)
        if electrons:
            electrons = load_mc(electrons, obstime, IRFDOC_ELECTRON_SPECTRUM)
        if electrons and protons:
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

        if selection:
            mask = create_mask_selection(combined, selection)
            combined = combined[mask]
            if background:
                mask = create_mask_selection(background, selection)
                background = background[mask]
            if gammas:
                mask = create_mask_selection(gammas, selection)
                gammas = gammas[mask]

        for feature in combined.keys():
            plot_data[feature] = {}
            plot_data[feature]["bins"] = pd.Series(dtype=np.float64)
            plot_data[feature]["values"] = pd.DataFrame()

            log.info(feature)
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
            log.debug(min_)
            log.debug(max_)
            if feature in logx:
                bins = np.logspace(np.log10(min_), np.log10(max_), n)
            else:
                bins = np.linspace(min_, max_, n)

            plot_data[feature]["bins"] = pd.Series(bins)
            feature_df = pd.DataFrame()
            feature_df["observations"], _ = np.histogram(
                get_value(combined, feature), bins=bins
            )
            if background:
                feature_df["background"], _ = np.histogram(
                    get_value(background, feature),
                    weights=background["weights"],
                    bins=bins,
                )
            if gammas:
                feature_df["gammas"], _ = np.histogram(
                    get_value(gammas, feature), weights=gammas["weights"], bins=bins
                )
            for run_data in observations:
                run_id = run_data[0]["obs_id"]
                log.info(f"Filling feature df for run {run_id}")
                # why would that happen?
                if feature not in run_data.keys():
                    log.debug(f"{feature} missing in keys: {run_data.keys()}")
                    continue
                feature_df[run_id], _ = np.histogram(
                    get_value(run_data, feature), bins=bins
                )
            plot_data[feature]["values"] = feature_df
        save_plot_data(cache, plot_data)

    figs = []

    datamc = ["observations"]
    if protons or electrons:
        datamc.append("background")
    if source_gammas:
        datamc.append("gammas")
    for feature in combined.keys():
        log.debug(f"Plotting feature: {feature}")
        if not plot_data[feature]["values"].empty:
            log.debug("And indeed there is data")
            figs.append(
                compare_datasets(plot_data, feature, datamc, logx=feature in logx)
            )
            figs.append(
                compare_datasets(
                    plot_data,
                    feature,
                    set(plot_data[feature]["values"].keys()) - set(datamc),
                    logx=feature in logx,
                )
            )

    if output is None:
        matplotlib.pyplot.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figs:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)


if __name__ == "__main__":
    main()
