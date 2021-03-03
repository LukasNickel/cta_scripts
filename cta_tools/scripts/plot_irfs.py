from cta_tools.plotting.irfs import (
    plot_aeff,
    plot_edisp,
    plot_sensitivity,
    plot_angular_resolution,
    plot_energy_bias_resolution,
    plot_background,
    plot_theta_cuts,
    plot_gh_cuts,
)
import matplotlib.pyplot as plt
from astropy.table import QTable
import click
import matplotlib

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


@click.command()
@click.argument("infile", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False))
def main(infile, output):

    figures = []

    sens = QTable.read(infile, hdu="SENSITIVITY")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_sensitivity(sens[1:-1], ax)

    aeff = QTable.read(infile, hdu="EFFECTIVE_AREA")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_aeff(aeff[0], ax)

    edisp = QTable.read(infile, hdu="ENERGY_DISPERSION")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_edisp(edisp[0], ax)

    angres = QTable.read(infile, hdu="ANGULAR_RESOLUTION")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_angular_resolution(angres, ax)

    energres = QTable.read(infile, hdu="ENERGY_BIAS_RESOLUTION")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_energy_bias_resolution(energres, ax)

    bkg = QTable.read(infile, hdu="BACKGROUND")[0]
    rad_max = QTable.read(infile, hdu="RAD_MAX")[0]
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_background(bkg, rad_max, ax)

    thetacuts = QTable.read(infile, hdu="THETA_CUTS")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_theta_cuts(thetacuts, ax)

    ghcuts = QTable.read(infile, hdu="GH_CUTS")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_gh_cuts(ghcuts, ax)

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
                pdf.savefig(fig)


if __name__ == "__main__":
    main()
