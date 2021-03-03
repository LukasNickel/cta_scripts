from astropy.table import vstack
import h5py
from aict_tools.io import read_data
import matplotlib.pyplot as plt
import re
from pathlib import Path
import click
import numpy as np
import astropy.units as u
from cta_tools.plotting.theta2 import theta2
from cta_tools.io import read_lst_dl2
from astropy.coordinates import SkyCoord
from cta_tools.reco.theta import calc_wobble_thetas


def plot(data, ontime, ax):
    on = data["theta_on"].to_value(u.deg)
    off = []
    noff = 0
    for c in data.keys():
        if c.startswith("theta_off"):
            off.append(data[c].to_value(u.deg))
            noff += 1
    theta2(on, off, scaling=1 / noff, cut=1, ontime=ontime, ax=ax, bins=20)
    return ax


@click.command()
@click.argument(
    "input_files",
    nargs=-1,
)
@click.argument("output", type=click.Path(exists=False, dir_okay=False))
@click.option("--source_ra", default=83.63308333)
@click.option("--source_dec", default=22.0145)
def main(
    input_files,
    output,
    source_ra,
    source_dec,
):
    ontime = 0
    runs = []
    for f in input_files:
        print(f)
        data = read_lst_dl2(f)
        source = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg, frame="icrs")
        theta_on, off_thetas = calc_wobble_thetas(data, source=source)
        data["theta_on"] = theta_on
        for i, theta_off in enumerate(off_thetas):
            data[f"theta_off_{i}"] = theta_off
        ontime += (data["time"][-1] - data["time"][0]).to_value(u.s)
        ontime += 0
        runs.append(data)

    events = vstack(runs)
    figures = []
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot(data, ontime, ax)
    ax.set_title("All energies")

    energy_selection = [
        (0 * u.GeV, 50 * u.GeV),
        (50 * u.GeV, 100 * u.GeV),
        (100 * u.GeV, 200 * u.GeV),
        (200 * u.GeV, 500 * u.GeV),
        (500 * u.GeV, 1000 * u.GeV),
        (1 * u.TeV, 100 * u.TeV),
    ]
    for low, high in energy_selection:
        print(low, high)
        selection = (data["reco_energy"] > lower) & (data["reco_energy"] <= upper)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        plot(data, ontime, ax)
        ax.set_title(f"{lower} - {upper}")

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
                pdf.savefig(fig)


if __name__ == "__main__":
    main()
