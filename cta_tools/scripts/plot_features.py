import click
import numpy as np
import matplotlib.pyplot as plt
import astropy
from aict_tools.io import read_data
from pathlib import Path
import astropy.units as u
from tqdm import tqdm
from astropy.table import vstack
from cta_tools.io import read_lst_dl1, read_mc_dl1, read_sim_info
from cta_tools.plotting.features import plot_dl1
import matplotlib
from pyirf.spectral import (
    calculate_event_weights,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
    PowerLaw,
    CRAB_MAGIC_JHEAP2015,
)

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


# make this a yaml file or smth for the file lists and spectras to weight to
@click.command()
@click.argument(
    "input_files",
    nargs=-1,
)
@click.option("--protons", "-p", type=click.Path(exists=True))
@click.option("--electrons", "-e", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
def main(
    input_files,
    protons,
    electrons,
    output,
):
    observations = {}

    obstime = 0 * u.s
    for f in tqdm(input_files):
        data = read_lst_dl1(f)
        observations[data[0]["obs_id"]] = data
        run_time = (data["time"][-1] - data["time"][0]).to(u.s)
        data["weights"] = 1 / run_time.to_value(u.s)
        obstime += run_time

    combined = vstack(list(observations.values()))
    combined["weights"] = 1 / obstime.to_value(u.s)

    proton_sim_info = read_sim_info(protons)
    protons = read_mc_dl1(protons)

    protons["weights"] = calculate_event_weights(
        protons["true_energy"],
        IRFDOC_PROTON_SPECTRUM,
        PowerLaw.from_simulation(proton_sim_info, 1 * u.s),
    )
    electron_sim_info = read_sim_info(electrons)
    electrons = read_mc_dl1(electrons)
    electrons["weights"] = calculate_event_weights(
        electrons["true_energy"],
        IRFDOC_ELECTRON_SPECTRUM,
        PowerLaw.from_simulation(electron_sim_info, 1 * u.s),
    )
    background = vstack([protons, electrons])
    figs = []
    figs += plot_dl1(observations)
    figs += plot_dl1({"observations": combined, "background mc": background})

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figs:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)
