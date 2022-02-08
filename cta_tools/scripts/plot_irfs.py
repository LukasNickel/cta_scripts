from cta_tools.plotting import preliminary
from cta_tools.plotting.irfs import (
    plot_aeff,
    plot_edisp,
    plot_sensitivity,
    plot_sensitivity_gp,
    plot_angular_resolution,
    plot_energy_bias_resolution,
    plot_background,
    plot_efficiency,
    plot_theta_cuts,
    plot_gh_cuts,
)
import matplotlib.pyplot as plt
from astropy.table import QTable
import click
import matplotlib
from cta_tools.logging import setup_logging


log = setup_logging()

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


@click.command()
@click.argument("infile", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False))
def main(infile, output):

    figures = []
    try:
        magic = QTable.read('magic_sensitivity_2014.ecsv')
    except:
        print("Couldnt read reference")
        magic = None


    sens = QTable.read(infile, hdu="SENSITIVITY")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_sensitivity(sens[1:-1], ax, label='LST-1')
    
    if magic:
        for k in filter(lambda k: k.startswith('sensitivity_') or k.startswith('e_'), magic.colnames):
            magic[k].info.format = '.3g'
        magic['reco_energy_low'] = magic['e_min']
        magic['reco_energy_high'] = magic['e_max']
        magic['reco_energy_center'] = magic['e_center']
        magic['flux_sensitivity'] = magic['sensitivity_lima_5off']

        plot_sensitivity(magic, ax, label='MAGIC')
    ax.legend()
   
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_sensitivity_gp(sens[1:-1], ax)

    aeff = QTable.read(infile, hdu="EFFECTIVE_AREA")
    aeff_only_gh = QTable.read(infile, hdu="EFFECTIVE_AREA_ONLY_GH")
    aeff_no_cuts = QTable.read(infile, hdu="EFFECTIVE_AREA_NO_CUTS")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_aeff(aeff[0], ax, label='gh + theta')
    plot_aeff(aeff_only_gh[0], ax, label='Only GH')
    plot_aeff(aeff_no_cuts[0], ax, label='No Cuts')
    ax.legend()

    edisp = QTable.read(infile, hdu="ENERGY_DISPERSION")
    edisp_only_gh = QTable.read(infile, hdu="ENERGY_DISPERSION_ONLY_GH")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_edisp(edisp[0], ax, label='gh + theta')
    plot_edisp(edisp_only_gh[0], ax, label='only gh')

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

    signal = QTable.read(infile, hdu="SIGNAL")
    signal_gh = QTable.read(infile, hdu="SIGNAL_GH")
    signal_cuts = QTable.read(infile, hdu="SIGNAL_CUTS")
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_efficiency({'NO CUTS': signal,'ONLY GH': signal_gh,'CUTS': signal_cuts} , ax)
    ax.legend()
    
    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
                pdf.savefig(fig)


if __name__ == "__main__":
    main()
