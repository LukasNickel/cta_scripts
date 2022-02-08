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

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel("INFO")

def plot(data, ontime, ax):
#    from IPython import embed;embed()
    on = data['theta_on'].to_value(u.deg)
    off = []
    noff = 0
    for c in data.keys():
        if c.startswith('theta_off'):
            off.append(data[c].to_value(u.deg))
            noff += 1
    theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.1, ontime=ontime, ax=ax, bins=30, window=(0, 0.1))
    return ax
        

@click.command()
@click.argument('input_files', nargs=-1,)
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
@click.option("--source_ra", default=83.63308333)
@click.option("--source_dec", default=22.0145)
def main(input_files, output, source_ra, source_dec,):
    cache = Path(output).with_suffix(".h5")
    if cache.exists():
        events = Table.read(cache)
        ontime = events.meta["ontime"]
        energy_key = events.meta["energy_key"]
    else:
        ontime = 0
        runs = []
        for f in input_files:
            print(f)
            if f.endswith('.h5'):
                data = read_data_dl2_to_QTable(f)
                log.info(data.keys())
#                 data = read_lst_dl2(f)
                time_key = 'time'
                energy_key = 'reco_energy'
                data["time"] = Time(data["dragon_time"], format="unix", scale="utc")
                log.info(data["time"].to_datetime())
            else:
                # adding a rename option would be more in line with what i do elsewhere....
                data = read_dl3(f)
                time_key = 'TIME'
                energy_key = 'ENERGY'

            source=SkyCoord(ra=source_ra*u.deg, dec=source_dec*u.deg, frame='icrs')

            theta_on, off_thetas = calc_wobble_thetas(data, source=source)
            data['theta_on'] = theta_on
            for i, theta_off in enumerate(off_thetas):
                data[f'theta_off_{i}'] = theta_off
            
            ontime += (data[time_key][-1] - data[time_key][0])#.to_value(u.s)
            ontime += 0
            runs.append(data)
        events = vstack(runs)
        keepers = []
        substrings = [
                "energy",
                "time",
                "theta",
                "gh_score",
                ]
        for c in events.keys():
            for k in substrings:
                if k in c.lower():
                    keepers.append(c)
                    break
        events.keep_columns(keepers)
        events.meta["ontime"] = ontime
        events.meta["energy_key"] = energy_key
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
        print(low, high)
        selection = (events[energy_key] > low) & (events[energy_key] <= high)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        plot(events[selection], ontime, ax)
        ax.set_title(f'{low} - {high}')        

    if 'gh_score' in events.keys():
        selection = (events['gh_score'] > 0.6)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events[selection]['theta_on'].to_value(u.deg)
        off = []
        noff = 0
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[selection][c].to_value(u.deg))
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.04, ontime=ontime, ax=ax, bins=100, window=(0, .3))
        ax.set_title("gh > 0.6")        

    if 'gh_score' in events.keys():
        selection = (events['gh_score'] > 0.7)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events[selection]['theta_on'].to_value(u.deg)
        off = []
        noff = 0
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[selection][c].to_value(u.deg))
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.04, ontime=ontime, ax=ax, bins=100, window=(0, 0.3))  
        ax.set_title("gh > 0.7")        

    if 'gh_score' in events.keys():
        selection = (events['gh_score'] > 0.8)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events[selection]['theta_on'].to_value(u.deg)
        off = []
        noff = 0
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[selection][c].to_value(u.deg))
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.04, ontime=ontime, ax=ax, bins=100, window=(0, 0.3))  
        ax.set_title("gh > 0.8")        
        
    if 'gh_score' in events.keys():
        selection = (events['gh_score'] > 0.85)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events[selection]['theta_on'].to_value(u.deg)
        off = []
        noff = 0
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[selection][c].to_value(u.deg))
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.04, ontime=ontime, ax=ax, bins=100, window=(0, 0.3))    
        ax.set_title("gh > 0.85")        

    if 'gh_score' in events.keys():
        selection = (events['gh_score'] > 0.6) & (events['reco_energy'] > 100 *u.GeV)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events[selection]['theta_on'].to_value(u.deg)
        off = []
        noff = 0
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[selection][c].to_value(u.deg))
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.04, ontime=ontime, ax=ax, bins=100, window=(0, .3))
        ax.set_title("gh > 0.6; E>100GeV")        

    if 'gh_score' in events.keys():
        selection = (events['gh_score'] > 0.7) & (events['reco_energy'] > 100 *u.GeV)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events[selection]['theta_on'].to_value(u.deg)
        off = []
        noff = 0
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[selection][c].to_value(u.deg))
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.04, ontime=ontime, ax=ax, bins=100, window=(0, 0.3))  
        ax.set_title("gh > 0.6; E>100GeV")        

    if 'gh_score' in events.keys():
        selection = (events['gh_score'] > 0.8) & (events['reco_energy'] > 100 *u.GeV)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events[selection]['theta_on'].to_value(u.deg)
        off = []
        noff = 0
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[selection][c].to_value(u.deg))
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.04, ontime=ontime, ax=ax, bins=100, window=(0, 0.3))  
        ax.set_title("gh > 0.8; E>100GeV")        

    if 'gh_score' in events.keys():
        selection = (events['gh_score'] > 0.85) & (events['reco_energy'] > 100 *u.GeV)
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events[selection]['theta_on'].to_value(u.deg)
        off = []
        noff = 0
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[selection][c].to_value(u.deg))
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, scaling=1/noff, cut=.04, ontime=ontime, ax=ax, bins=100, window=(0, 0.3))  
        ax.set_title("gh > 0.85; E>100GeV")        
    

    if 'PASSED_THETA_CUT' in events.keys():
        selection = list(events['PASSED_THETA_CUT'])
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events['theta_on'].to_value(u.deg)
        w_on = selection
        off = []
        w_off = []
        noff = 0
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[c].to_value(u.deg))
                w_off += selection
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, cut=0.04, scaling=1/noff, ontime=ontime, ax=ax, bins=30, window=(0, 1), on_weights=w_on, off_weights=w_off)  
        ax.set_title("Passed theta")        

    if 'PASSED_THETA_CUT' in events.keys():
        selection = events['PASSED_THETA_CUT']
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        on = events[selection]['theta_on'].to_value(u.deg)
        noff = 0
        off = []
        for c in events.keys():
            if c.startswith('theta_off'):
                off.append(events[selection][c].to_value(u.deg))
                noff += 1
        theta2(on**2, np.array(off).ravel()**2, cut=0.04,  scaling=1/noff, ontime=ontime, ax=ax, bins=30, window=(0, 1))  
        ax.set_title("Passed theta")        
        
    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
                pdf.savefig(fig)


if __name__ == "__main__":
    main()
