from cta_tools.plotting import preliminary
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
from regions import CircleSkyRegion
from gammapy.modeling import Fit
from gammapy.data import EventList
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import click
from cta_tools.logging import setup_logging


log = setup_logging()

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages

@click.command()
@click.argument('input_files', nargs=-1,)
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False))
def main(input_files, output):
    eventlist_list = []
    for f in input_files:
        eventlist_list.append(EventList.read(f))
    events = eventlist_list[0]
    for e in eventlist_list[1:]:
        events.stack(e)
    figures = []
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    events.plot_energy(ax=ax)

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    events.plot_energy_offset(ax=ax)

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    events.plot_offset2_distribution(ax=ax)

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    events.plot_time(ax=ax)
    
    #figures.append(events.plot_image())

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
                pdf.savefig(fig)



if __name__ == "__main__":
    main()
