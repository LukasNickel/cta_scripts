from cta_tools.plotting import preliminary
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
from astropy.table import vstack, Table
from cta_tools.io import read_lst_dl1, read_mc_dl1, read_plot_data, save_plot_data
from cta_tools.plotting.features import plot_binned_time_evolution
from cta_tools.utils import bin_df
import matplotlib
import asyncio
import logging
if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages
import asyncclick as click
import pandas as pd


log = logging.getLogger(__name__)
log.setLevel("INFO")

cols = ["hillas_intensity", "hillas_x", "hillas_y", "pointing_az", "pointing_alt", "hillas_length", "hillas_width", "morphology_num_pixels", "morphology_num_islands", "intensity_mean", "intensity_std"]

data_structure = {
    key: {"bins": None, "values": None} for key in cols
}


async def load_run(path):
    data = read_lst_dl1(path)
    log.info(f"Loading of {path} finished")
    print(f"Loading of {path} finished")
    return data


async def plot_feature(df, col):
    fig, ax = plt.subplots()
    plot_binned_time_evolution(df, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("time w.r.t. the first event [s]")
    return fig

def bin_(df, col, bins, output):
    try:
        return pd.read_hdf(output, key=col)
    except:
        binned = bin_df(df, "delta_t_sec", col, bins=bins)
        binned.to_hdf(output, key=col)
        return binned


# make this a yaml file or smth for the file lists and spectras to weight to
@click.argument(
    "input_files",
    nargs=-1,
)
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
@click.command()
async def main(
    input_files,
    output,
):
    cache = Path(output).with_suffix(".h5")
    figs = []

    if cache.exists():
        plot_data = read_plot_data(cache, data_structure)
    else:
        plot_data = data_structure.copy()
        obstime = 0 * u.s
        observations = {}
        runs = await asyncio.gather(*[asyncio.create_task(load_run(f)) for f in input_files])
        for run in runs:
            observations[run[0]["obs_id"]] = run
            run_time = (run["time"][-1] - run["time"][0]).to(u.s)
            #run["weights"] = 1 / run_time.to_value(u.s)
            obstime += run_time

        combined = vstack(list(observations.values()))
        #combined["weights"] = 1 / obstime.to_value(u.s)
        combined["delta_t_sec"] = (combined["time"] - combined["time"][0]).sec
        last = combined["delta_t_sec"].max()
        time_bins = np.linspace(0, last, 20)
        if isinstance(combined, Table):
            combined = combined.to_pandas()
            combined.set_index(["obs_id", "event_id"])        
        for col in cols:
            plot_data[col]["bins"] = pd.Series(time_bins)
            plot_data[col]["values"] = bin_df(combined, "delta_t_sec", col, bins=time_bins)
        save_plot_data(cache, plot_data)

    for col, data in plot_data.items():
        figs.append(asyncio.create_task(plot_feature(data["values"], col)))
    figs = await asyncio.gather(*figs)

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
