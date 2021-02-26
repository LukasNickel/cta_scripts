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
from cta_tools.io import read_table, read_cuts


@click.command()
@click.argument("input_pattern", type=str)
@click.argument("output", type=click.Path(exists=False, dir_okay=False))
@click.option("--min_energy", type=float, default=0, help="TEV!")
@click.option("--max_energy", type=float, default=1e100)
def main(input_pattern, output, min_energy, max_energy):
    print(input_pattern, input_pattern.split("/"[-1]))
    input_files = Path(input_pattern).parent.glob(input_pattern.split("/")[-1])
    energy_key = (
        "ENERGY" if input_pattern.endswith("fits.gz") else "gamma_energy_prediction"
    )
    run_tables = []
    live_time = 0
    runs = []
    for f in input_files:
        print(f)
        t = read_table(f)
        mask = (t[energy_key].to_value(u.TeV) > min_energy) & (
            t[energy_key].to_value(u.TeV) < max_energy
        )
        cuts = read_cuts(f)
        if cuts is not None:
            for selection in cuts.keys():
                if "theta" not in selection:
                    print(selection)
                    mask &= cuts[selection].values
        run_tables.append(t[mask])
        # number = re.search(r'([0-9]{5})', f.stem).group(0)
        live_time += 0  # t.meta["LIVETIME"]
        runs.append(0)
        # runs.append(int(number))

    runs.sort()
    events = vstack(run_tables)
    print(events.keys())
    theta_on = events["THETA_ON"]
    print("On events:", len(theta_on))
    off_thetas = []
    for c in [col for col in events.keys() if "THETA_OFF" in col]:
        off = events[c]
        print("wobble position ", c, len(off))
        off_thetas.append(off.data)

    alpha = 1 / len(off_thetas)
    theta_off = np.concatenate(off_thetas)

    plt.figure()
    ax = plt.gca()
    xmax = 0.2
    theta2(
        theta_on ** 2,
        theta_off ** 2,
        scaling=alpha,
        cut=0.5,  # dummy
        ontime=live_time * u.s,
        ax=ax,
        window=[0, xmax],
        bins=30,
    )
    plt.title(f"Runs: {runs[0]} - {runs[-1]}")
    plt.savefig(output)


if __name__ == "__main__":
    main()
