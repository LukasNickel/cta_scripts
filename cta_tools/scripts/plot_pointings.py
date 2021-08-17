from cta_tools.plotting import preliminary
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aict_tools.io import read_data
from pathlib import Path
from tqdm import tqdm


@click.command()
@click.argument("pattern")
@click.argument("output")
@click.option("--mc_alt", type=float)
@click.option("--mc_az", type=float)
def main(pattern, output, mc_alt, mc_az):
    files = Path(pattern).parent.glob(pattern.split("/")[-1])
    dfs = []
    for f in tqdm(files):
        dfs.append(read_data(f, "events", columns=["alt_tel", "az_tel"]))

    pointings = pd.concat(dfs)
    fig, (ax_alt, ax_az) = plt.subplots(1, 2)
    ax_alt.hist(
        np.rad2deg(pointings["alt_tel"].values.flatten()),
        bins=30,
        histtype="stepfilled",
        alpha=0.5,
        label="Pointings",
    )
    if mc_alt:
        ax_alt.axvline(mc_alt, linestyle="--", color="black", label="Monte Carlo")
    ax_alt.set_title("Altitude")
    ax_alt.legend()

    ax_az.hist(
        np.rad2deg(pointings["az_tel"].values.flatten()),
        bins=30,
        histtype="stepfilled",
        alpha=0.5,
        label="Pointings",
    )
    if mc_az:
        ax_az.axvline(mc_az, linestyle="--", color="black", label="Monte Carlo")
    ax_az.set_title("Azimuth")
    ax_az.legend()

    fig.savefig(output)


if __name__ == "__main__":
    main()
