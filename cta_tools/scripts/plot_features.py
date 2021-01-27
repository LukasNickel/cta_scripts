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
    CRAB_MAGIC_JHEAP2015
)
if matplotlib.get_backend() == 'pgf':
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages

# make this a yaml file or smth for the file lists and spectras to weight to
@click.command()
@click.option(
    '--name',
    '-n',
    multiple=True,
    help='name for the observation to be included in the plot'
)
@click.option(
    '--pattern', '-p', multiple=True, type=str,
    help='Patern for files to the associated observation'
)
@click.option(
    '--weight',
    '-w',
    multiple=True,
    type=int,
    help='reweight to spectrum. 0: do nothing, 1: crab, 2: Proton CR'
)
@click.option('--feature_name', '-f', multiple=True, type=str)
@click.option('--output', '-o', type=click.Path(exists=False, dir_okay=False))
@click.option('--selection', '-s', default=False, is_flag=True)
@click.option('--gh_cuts', '-g', default=False, is_flag=True)
@click.option('--theta_cuts', '-t', default=False, is_flag=True)
@click.option('--t_obs', '-h', default=50)
def main(
        name,
        pattern,
        feature_name,
        output,
        weight,
        selection,
        gh_cuts,
        theta_cuts,
        t_obs
):
    observations = []
    assert len(name) == len(pattern)
    for n, p, w in zip(name, pattern, weight):
        files = Path(p).parent.glob(p.split('/')[-1])
        data_tables = []
        weights = []
        cols = list(feature_name)
        if w != 0:
            cols.append('gamma_energy_prediction')
        for f in tqdm(files):
            data = read_table(f, columns=cols)
            if w == 1:
                sim_info = read_sim_info(f)
                weights = calculate_event_weights(
                    u.Quantity(data['gamma_energy_prediction'], u.TeV, copy=False),
                    PowerLaw.from_simulation(sim_info, t_obs*u.h),
                    CRAB_MAGIC_JHEAP2015,
                )
                data['weights'] = weights
            elif w == 2:
                sim_info = read_sim_info(f)
                weights = calculate_event_weights(
                    u.Quantity(data['gamma_energy_prediction'], u.TeV, copy=False),
                    PowerLaw.from_simulation(sim_info, t_obs*u.h),
                    IRFDOC_PROTON_SPECTRUM,
                )
                data['weights'] = weights
            else:
                data['weights'] = 1

            if theta_cuts or gh_cuts:
                cuts = read_data(f, 'cuts')
                if theta_cuts:
                    data = data[cuts['passed_theta']]
                if gh_cuts:
                    data = data[cuts['passed_gh']]
            data_tables.append(data)

        observations.append((
            n,
            w,
            vstack(data_tables)
        ))

    figures = []
    for feature in feature_name:
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
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
                    50,
                )
                if 'energy' in feature:
                    bins = np.logspace(
                        np.log10(values.min()),
                        np.log10(values.max()),
                        50,
                    )
                    ax.set_xscale('log')

            if (data['weights'] != 1).any():
                name += f' (reweighted to {t_obs}h)'
            ax.hist(
                values,
                histtype='step',
                label=name,
                bins=bins,
                weights=data['weights'],
            )

        ax.legend()
        title = feature
        ax.set_title(title)
        ax.set_yscale('log')

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)
