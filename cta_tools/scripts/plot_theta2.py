from astropy.table import vstack
from aict_tools.io import read_data
import matplotlib.pyplot as plt
import re
from pathlib import Path
import click
import numpy as np
import astropy.units as u
from cta_tools.plotting.theta2 import theta2
from cta_tools.io import read_table


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, dir_okay=True))
@click.argument('output', type=click.Path(exists=False, dir_okay=False))
@click.option('--offs', '-w', multiple=True, default=[1])
@click.option('--runs', type=str, default='*')
@click.option('--min_energy', type=int, default=1)
def main(input_folder, output, offs, runs, min_energy, theta_cut=False):
    reg = re.compile(runs)
    input_files = Path(input_folder).glob("**/*")
    run_tables = []
    live_time = 0
    runs = []
    for f in input_files:
        
        if reg.match(str(f)):
            print(f)
            t = read_table(f)
            cuts = read_data(f, 'cuts')
            emask = t['gamma_energy_prediction'] > min_energy/100#*u.GeV
            gh_mask = cuts['passed_gh']
            theta_mask = cuts['passed_theta']
            selected = emask & gh_mask #& theta_mask
            if theta_cut:
                selected &= theta_mask
            run_tables.append(t[selected.values])

            number = re.search(r'([0-9]{5})', f.stem).group(0)
            live_time += 0#t.meta["LIVETIME"]
            runs.append(int(number))

    runs.sort()
    events = vstack(run_tables)
    #theta_cuts = events['THETA_CUT_VALUE']
    theta_on = events['theta_on']#[events['THETA_ON'] < theta_cuts]
    print('On events:', len(theta_on))
    off_thetas = []
    alpha = 1/len(offs)
    for i in offs:
        off = events[f"theta_off{i}"]#[events[f"THETA_OFF{i}"] < theta_cuts]
        print(len(off))
        off_thetas.append(off.data)

    theta_off = np.concatenate(off_thetas)


    plt.figure()
    ax = plt.gca()
    xmax = 0.2
    theta2(
        theta_on**2,
        theta_off**2,
        scaling=alpha,
        cut=0.5,  # dummy
        ontime=live_time*u.s,
        ax=ax,
        window=[0, xmax],
        bins=30,
    )
    plt.title(f'Runs: {runs[0]} - {runs[-1]}')
    plt.savefig(output)


if __name__ == '__main__':
    main()
