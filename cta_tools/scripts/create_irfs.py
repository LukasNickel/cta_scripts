import numpy as np
import click
import logging
import operator
import yaml
from astropy import table
from pathlib import Path
import astropy.units as u
from astropy.io import fits
from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
)
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution
from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_MAGIC_JHEAP2015,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
    background_2d,
)
from pyirf.gammapy import (   
    create_psf_3d,
    create_energy_dispersion_2d,
    create_effective_area_table_2d,
)
from pyirf.io import (
    create_aeff2d_hdu,
    create_psf_table_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
    create_background_2d_hdu,
)
from cta_tools.io import read_to_pyirf
from cta_tools.logging import setup_logging


log = setup_logging()


def get_global_cut_table(cut_value):
    bins = u.Quantity([1, 1e6], u.GeV)
    cut_table = QTable()
    cut_table["low"] = bins[:-1]
    cut_table["high"] = bins[1:]
    cut_table["center"] = bin_center(bins)
    cut_table["cut"] = cut_value
    return cut_table


@click.command()
@click.option("-g", "--gamma", type=click.Path(exists=True, dir_okay=False))
@click.option("-p", "--proton", type=click.Path(exists=True, dir_okay=False))
@click.option("-e", "--electron", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--irfoutput", type=click.Path())
@click.option("-c", "--config-file", type=str)
def main(gamma, proton, electron, irfoutput, config_file):
    irf_output = Path(irfoutput)
    gammapy_output = (irf_output.parent / irf_output.stem.partition(".")[0]).with_suffix(".gammapy.h5")

    with Path(config_file).open() as f:
        config = yaml.safe_load(f)
    T_OBS = config["obstime"] * u.hour

    # scaling between on and off region.
    # Make off region 2 times larger than on region for better
    # background statistics
    ALPHA = config["alpha"]

    # Radius to use for calculating bg rate
    MAX_BG_RADIUS = config["max_bg_radius"] * u.deg
    # gh cut used for first calculation of the binned theta cuts
    emin = 5 * u.GeV
    emax = 50 * u.TeV

    particles = {
        "gamma": {
            "file": gamma,
            "target_spectrum": CRAB_MAGIC_JHEAP2015,
        },
        "proton": {
            "file": proton,
            "target_spectrum": IRFDOC_PROTON_SPECTRUM,
        },
        "electron": {
            "file": electron,
            "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
        },
    }

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pyirf").setLevel(logging.DEBUG)

    for particle_type, p in particles.items():
        log.info(f"Simulated {particle_type.title()} Events:")
        p["events"], p["simulation_info"] = read_to_pyirf(p["file"])
        p["events"]["particle_type"] = particle_type

        p["simulated_spectrum"] = PowerLaw.from_simulation(p["simulation_info"], T_OBS)
        p["events"]["weight"] = calculate_event_weights(
            p["events"]["true_energy"], p["target_spectrum"], p["simulated_spectrum"]
        )
        for prefix in ("true", "reco"):
            k = f"{prefix}_source_fov_offset"
            p["events"][k] = calculate_source_fov_offset(p["events"], prefix=prefix)

        p["events"]["theta"] = calculate_theta(
            p["events"],
            assumed_source_az=p["events"]["true_az"],
            assumed_source_alt=p["events"]["true_alt"],
        )
        log.info(p["simulation_info"])
        log.info("")

    gammas = particles["gamma"]["events"]

    # background table composed of both electrons and protons
    background = table.vstack(
        [particles["proton"]["events"], particles["electron"]["events"]]
    )

    sensitivity_bins = add_overflow_bins(
        create_bins_per_decade(emin, emax, bins_per_decade=5)
    )

    #sensitivity_bins[-1] = 100 * u.TeV
    #sensitivity_bins[0] = 1 * u.GeV
    #theta_bins = sensitivity_bins
    gh_cuts = calculate_percentile_cut(
        gammas["gh_score"],
        gammas["reco_energy"],
        bins=sensitivity_bins,
        min_value=config["gh_cuts"]["min"],
        max_value=config["gh_cuts"]["max"],
        fill_value=config["gh_cuts"]["fill"],
        percentile=config["gh_cuts"]["percentile"],
    )
    for tab in (gammas, background):
        tab["selected_gh"] = evaluate_binned_cut(
            tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
        )

    theta_bins = sensitivity_bins[[0,-1]] if config["theta_cuts"]["global"] else sensitivity_bins
    theta_cuts = calculate_percentile_cut(
        gammas[gammas["selected_gh"]]["theta"],
        gammas[gammas["selected_gh"]]["reco_energy"],
        bins=theta_bins,
        percentile=config["theta_cuts"]["percentile"],
        min_value=config["theta_cuts"]["min"] * u.deg,
        max_value=config["theta_cuts"]["max"] * u.deg,
        fill_value=config["theta_cuts"]["fill"] * u.deg,
    )
    gammas["selected_theta"] = evaluate_binned_cut(
        gammas["theta"], gammas["reco_energy"], theta_cuts, operator.le
    )
    gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]


    # SENSITIVITY
    signal_hist_no_cuts = create_histogram_table(
        gammas, bins=sensitivity_bins
    )
    signal_hist_only_gh = create_histogram_table(
        gammas[gammas["selected_gh"]], bins=sensitivity_bins
    )
    # calculate sensitivity
    signal_hist = create_histogram_table(
        gammas[gammas["selected"]], bins=sensitivity_bins
    )
    # selected gh because a larger background region is used and then scaled 
    background_hist = estimate_background(
        background[background["selected_gh"]],
        reco_energy_bins=sensitivity_bins,
        theta_cuts=theta_cuts,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )
    sensitivity = calculate_sensitivity(signal_hist, background_hist, alpha=ALPHA)

    # scale relative sensitivity by Crab flux to get the flux sensitivity
    spectrum = particles["gamma"]["target_spectrum"]
    for s in (sensitivity,):
        s["flux_sensitivity"] = s["relative_sensitivity"] * spectrum(
            s["reco_energy_center"]
        )
    log.info("Calculating IRFs")
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
        fits.BinTableHDU(theta_cuts, name="THETA_CUTS"),
        fits.BinTableHDU(gh_cuts, name="GH_CUTS"),
    ]

    gammapy_hdus = []
    gammapy_irfs = {}

    # IRFS 
    masks = {
        "": gammas["selected"],
        "_NO_CUTS": slice(None),
        "_ONLY_GH": gammas["selected_gh"],
    }

    # binnings for the irfs
    true_energy_bins = add_overflow_bins(create_bins_per_decade(emin, emax, 10))
#    true_energy_bins[-1] = 100 * u.TeV
#    true_energy_bins[0] = 1 * u.GeV

    reco_energy_bins = add_overflow_bins(create_bins_per_decade(emin, emax, 5))
#    reco_energy_bins[-1] = 100 * u.TeV
#    reco_energy_bins[0] = 1 * u.GeV

    wobble_offset = config["wobble_offset"] * u.deg
    fov_offset_bins = u.Quantity([wobble_offset - 0.01*u.deg, wobble_offset + 0.01 * u.deg])
    
    #fov_offset_bins = [0, 0.5] * u.deg
    source_offset_bins = np.arange(0, 1 + 1e-4, 1e-3) * u.deg
    energy_migration_bins = np.geomspace(0.2, 5, 100)

    for label, mask in masks.items():
        # AEFF
        effective_area = effective_area_per_energy(
            gammas[mask],
            particles["gamma"]["simulation_info"],
            true_energy_bins=true_energy_bins,
        )
        hdus.append(
            create_aeff2d_hdu(
                effective_area[..., np.newaxis],  # add one d for FOV offset
                true_energy_bins,
                fov_offset_bins,
                extname="EFFECTIVE_AREA" + label,
                TELESCOP="LST",  # gammapy????
            )
        )
        effective_area_gammapy = create_effective_area_table_2d(
                # add a new dimension for the single fov offset bin
                effective_area=effective_area[..., np.newaxis],
                true_energy_bins=true_energy_bins,
                fov_offset_bins=fov_offset_bins,
        )
        gammapy_irfs["EFFECTIVE_AREA" + label] = effective_area_gammapy

        # EDISP
        edisp = energy_dispersion(
            gammas[mask],
            true_energy_bins=true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            migration_bins=energy_migration_bins,
        )
        hdus.append(
            create_energy_dispersion_hdu(
                edisp,
                true_energy_bins=true_energy_bins,
                migration_bins=energy_migration_bins,
                fov_offset_bins=fov_offset_bins,
                extname="ENERGY_DISPERSION" + label,
            )
        )
        edisp_gammapy = create_energy_dispersion_2d(
            edisp,
            true_energy_bins,
            energy_migration_bins,
            fov_offset_bins
        )
        gammapy_irfs["ENERGY_DISPERSION" + label] = edisp_gammapy

    bias_resolution = energy_bias_resolution(
        gammas[gammas["selected"]],
        true_energy_bins,
    )
    ang_res = angular_resolution(
        gammas[gammas["selected_gh"]],
        true_energy_bins,
    )
    psf = psf_table(
        gammas[gammas["selected_gh"]],
        true_energy_bins,
        fov_offset_bins=fov_offset_bins,
        source_offset_bins=source_offset_bins,
    )
    psf_gammapy = create_psf_3d(psf, true_energy_bins, source_offset_bins,  fov_offset_bins)
    gammapy_irfs["PSF"] = psf_gammapy

    background_rate = background_2d(
        background[background["selected_gh"]],
        reco_energy_bins,
        fov_offset_bins=np.arange(0, 11) * u.deg,
        t_obs=T_OBS,
    )

    hdus.append(
        create_background_2d_hdu(
            background_rate,
            reco_energy_bins,
            fov_offset_bins=np.arange(0, 11) * u.deg,
        )
    )
    hdus.append(
        create_psf_table_hdu(
            psf,
            true_energy_bins,
            source_offset_bins,
            fov_offset_bins,
        )
    )
    hdus.append(
        create_rad_max_hdu(
            theta_cuts["cut"][:, np.newaxis], theta_bins, fov_offset_bins
        )
    )
    hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))
    hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))
    hdus.append(fits.BinTableHDU(signal_hist_no_cuts, name="SIGNAL"))
    hdus.append(fits.BinTableHDU(signal_hist_only_gh, name="SIGNAL_GH"))
    hdus.append(fits.BinTableHDU(signal_hist, name="SIGNAL_CUTS"))


    log.info("Writing outputfile")
#    for name, component in gammapy_irfs.items()#:
#        component.to_table().write(gammapy_output, name)
#        gammapy_hdus.append(component.to_table_hdu())

    fits.HDUList(hdus).writeto(irf_output, overwrite=True)


if __name__ == "__main__":
    main()
