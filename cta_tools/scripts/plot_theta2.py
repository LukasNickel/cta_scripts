from cta_tools.plotting import preliminary
from astropy.time import Time
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
from lstchain.io import read_data_dl2_to_QTable, read_mc_dl2_to_QTable
from astropy.table import Table
from cta_tools.io import read_lst_dl2, read_dl3, save_plot_data, read_plot_data
from astropy.coordinates import SkyCoord
from cta_tools.reco.theta import calc_wobble_thetas
import logging
import matplotlib
if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages

log = logging.getLogger(__name__)

def plot(data, ontime, ax):
    on = data['theta_on']
    off = []
    noff = 0
    for c in data.keys():
        if c.startswith('theta_off'):
            off.append(data[c])
            noff += 1
    theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.1, ontime=ontime, ax=ax, bins=30, window=(0, 0.1))
    return ax

@click.command()
@click.argument('input_files', nargs=-1,)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--cuts", "-c")
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
@click.option("--source_ra", default=83.63308333)
@click.option("--source_dec", default=22.0145)
def main(input_files, verbose, cuts, output, source_ra, source_dec,):
    setup_logging(verbose=verbose)
    if cuts:
        with open(cuts) as f:
            selection = yaml.safe_load(f).get("selection")
    else:
        selection = None

    cache = Path(output).with_suffix(".h5")
    if cache.exists():
        events = Table.read(cache)
        ontime = events.meta["ontime"]
    else:
        plot_data = {}
        observations, obstime = read_lst_dl2_runs(input_files)
        log.info(combined.keys())
        source=SkyCoord(ra=source_ra*u.deg, dec=source_dec*u.deg, frame='icrs')
        for run_id, run_data in observations::
            if selection:
                mask = create_mask_selection(run_data, selection)
                run_data = combined[mask]
            theta_on, off_thetas = calc_wobble_thetas(run_data, source=source)
            thetas = QTable()
            thetas['theta_on'] = theta_on.to_value(u.deg)
            for i, theta_off in enumerate(off_thetas):
                thetas[f'theta_off_{i}'] = theta_off.to_value(u.deg)
            thetas["energy"] = run_data["reco_energy"]
        events = vstack(thetas)
        events.meta["obstime"] = ontime
        events.write(cache, serialize_meta=True)

    log.info(f"Loaded {len(events)} events")
    figures = []
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot(events, ontime, ax)
    ax.set_title('All energies')

    energy_selection = [
        (0*u.GeV, 50*u.GeV),
        (50*u.GeV, 100*u.GeV),
        (100*u.GeV, 200*u.GeV),
        (200*u.GeV, 500*u.GeV),
        (500*u.GeV, 1000*u.GeV),
        (1*u.TeV, 100*u.TeV),
        (100*u.GeV, 100*u.TeV),
    ]
    for low, high in energy_selection:
        selection = (events[energy] > low) & (events[energy] <= high)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        plot(events[selection], ontime, ax)
        ax.set_title(f'{low} - {high}')        
    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
                pdf.savefig(fig)


if __name__ == "__main__":
    main()
