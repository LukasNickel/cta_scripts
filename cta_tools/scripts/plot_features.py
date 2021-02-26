import click
import numpy as np
import matplotlib.pyplot as plt
import astropy
from aict_tools.io import read_data
from pathlib import Path
import astropy.units as u
from tqdm import tqdm
from astropy.table import vstack
from cta_tools.io import read_table, read_sim_info
import matplotlib
from pyirf.spectral import (
    calculate_event_weights,
    IRFDOC_PROTON_SPECTRUM,
    PowerLaw,
    CRAB_MAGIC_JHEAP2015,
)

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages

# make this a yaml file or smth for the file lists and spectras to weight to
@click.command()
@click.option(
    "--name",
    "-n",
    multiple=True,
    help="name for the observation to be included in the plot",
)
@click.option(
    "--pattern",
    "-p",
    multiple=True,
    type=str,
    help="Patern for files to the associated observation",
)
@click.option(
    "--weight",
    "-w",
    multiple=True,
    type=int,
    help="reweight to spectrum. 0: do nothing, 1: crab, 2: Proton CR",
)
@click.option("--feature_name", "-f", multiple=True, type=str)
@click.option("--cut_columns", "-c", multiple=True, type=str)
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
def main(
    name,
    pattern,
    feature_name,
    cut_columns,
    output,
    weight,
):
    observations = []
    assert len(name) == len(pattern)
    for n, p, w in zip(name, pattern, weight):
        files = Path(p).parent.glob(p.split("/")[-1])
        data_tables = []
        weights = []
        cols = list(feature_name)
        obstime = 0
        if w != 0:
            cols.append("gamma_energy_prediction")
        else:
            cols.append("dragon_time")
        for f in tqdm(files):
            data = read_table(f, columns=cols)
            if cut_columns:
                cuts = read_data(f, "cuts")
                mask = np.ones(len(data), dtype=bool)
                for c in cut_columns:
                    mask &= cuts[f"passed_{c}"].values
                data = data[mask]
            data_tables.append(data)
            if w == 0:
                obstime += data["dragon_time"].max() - data["dragon_time"].min()
        combined = vstack(data_tables)
        # calculate weights for event rate
        if w == 1:
            sim_info = read_sim_info(f)
            combined["weights"] = calculate_event_weights(
                u.Quantity(combined["gamma_energy_prediction"], u.TeV, copy=False),
                CRAB_MAGIC_JHEAP2015,
                PowerLaw.from_simulation(sim_info, 1 * u.s),
            )
        elif w == 2:
            sim_info = read_sim_info(f)
            combined["weights"] = calculate_event_weights(
                u.Quantity(combined["gamma_energy_prediction"], u.TeV, copy=False),
                IRFDOC_PROTON_SPECTRUM,
                PowerLaw.from_simulation(sim_info, 1 * u.s),
            )
        elif w == 0:
            combined["weights"] = np.ones(len(combined)) / obstime
        else:
            raise Exception("Not implemented")

        observations.append((n, w, combined))

    figures = []
    for feature in feature_name:
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        nbins = 50
        # set bins?
        for i, (name, weight, data) in enumerate(observations):
            # hack for nans
            if isinstance(data[feature], astropy.table.column.MaskedColumn):
                values = data[feature].data.data
            elif isinstance(data[feature], astropy.table.column.Column):
                values = data[feature]
            if i == 0:
                bins = np.linspace(
                    np.nanpercentile(values, 1),
                    np.nanpercentile(values, 99),
                    nbins,
                )
                if any([x in feature for x in ["energy", "intensity"]]):
                    bins = np.logspace(
                        np.log10(values.min()),
                        np.log10(values.max()),
                        nbins,
                    )
                    ax.set_xscale("log")
            ax.hist(
                values,
                histtype="step",
                label=name,
                bins=bins,
                weights=data["weights"],
            )

        ax.legend()
        title = feature
        ax.set_title(title)
        ax.set_ylabel("Event Rate / s")
        ax.set_yscale("log")

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)
