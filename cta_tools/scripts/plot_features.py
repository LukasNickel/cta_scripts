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
import asyncio
import logging
if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages
import asyncclick as click


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
    # delta_t
]


data_structure = {k:{"bins":None, "values":None} for k in linx+logx}


async def load_run(path):
    data = read_lst_dl1(path)
    log.info(f"Loading of {path} finished")
    print(f"Loading of {path} finished")
    return data


def plot(data, feature, keys, logx=True):
    fig, ax = plt.subplots()
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
    #from IPython import embed
    #import nest_asyncio
    #nest_asyncio.apply()
    #embed(using='asyncio')
    return fig


# make this a yaml file or smth for the file lists and spectras to weight to
@click.argument(
    "input_files",
    nargs=-1,
)
@click.option("--protons", "-p", type=click.Path(exists=True))
@click.option("--electrons", "-e", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
@click.command()
async def main(
    input_files,
    protons,
    electrons,
    output,
):
    observations = {}

    cache = Path(output).with_suffix(".h5")
    if cache.exists():
        plot_data = read_plot_data(cache, data_structure)
    else:
        plot_data = data_structure.copy()
        obstime = 0 * u.s
        runs = await asyncio.gather(*[asyncio.create_task(load_run(f)) for f in input_files])
        for run in runs:
            observations[run[0]["obs_id"]] = run
            run_time = (run["time"][-1] - run["time"][0]).to(u.s)
            obstime += run_time

        combined = vstack(list(observations.values()))

        proton_sim_info = read_sim_info(protons)
        protons = read_mc_dl1(protons)

        protons["weights"] = calculate_event_weights(
            protons["true_energy"],
            IRFDOC_PROTON_SPECTRUM,
            PowerLaw.from_simulation(proton_sim_info, obstime),
        )
        electron_sim_info = read_sim_info(electrons)
        electrons = read_mc_dl1(electrons)
        electrons["weights"] = calculate_event_weights(
            electrons["true_energy"],
            IRFDOC_ELECTRON_SPECTRUM,
            PowerLaw.from_simulation(electron_sim_info, obstime),
        )
        background = vstack([protons, electrons])

        for feature in logx:
            bins = np.logspace(
                np.log10(get_value(combined, feature).min()),
                np.log10(get_value(combined, feature).max()),
                30
            )
            plot_data[feature]["bins"] = pd.Series(bins)
            feature_df = pd.DataFrame()
            feature_df["observations"], _ = np.histogram(combined[feature], bins=bins)
            feature_df["background"], _ = np.histogram(
                background[feature],
                weights = background["weights"],
                bins=bins
            )
            for run_data in runs:
                run_id = run[0]["obs_id"]
                # why would that happen?
                if feature not in run_data.keys():
                    continue
                feature_df[run_id], _ = np.histogram(
                    run_data[feature],
                    bins=bins
                )
            plot_data[feature]["values"] = feature_df

        for feature in linx:
            bins = np.linspace(
                get_value(combined, feature).min(),
                get_value(combined, feature).max(),
                30
            )
            plot_data[feature]["bins"] = pd.Series(bins)
            feature_df = pd.DataFrame()
            feature_df["observations"], _ = np.histogram(combined[feature], bins=bins)
            feature_df["background"], _ = np.histogram(
                background[feature],
                weights = background["weights"],
                bins=bins
            )
            for run_data in runs:
                run_id = run[0]["obs_id"]
                # why would that happen?
                if feature not in run_data.keys():
                    continue
                feature_df[run_id], _ = np.histogram(
                    run_data[feature],
                    bins=bins
                )
            plot_data[feature]["values"] = feature_df

    save_plot_data(cache, plot_data)

    figs = []

    for feature in logx:
        datamc = ["observations", "background"]
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
        datamc = ["observations", "background"]
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
    #main()
    main(_anyio_backend="asyncio")
