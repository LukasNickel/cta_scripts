from pyirf.utils import cone_solid_angle
import numpy as np
import astropy.units as u


def plot_sensitivity(sensitivity, ax=None, label=None):
    unit = u.Unit("erg cm-2 s-1")
    ax.errorbar(
        sensitivity["reco_energy_center"].to_value(u.TeV),
        (
            (sensitivity["reco_energy_center"].to(u.TeV)) ** 2
            * sensitivity["flux_sensitivity"]
        ).to_value(unit),
        xerr=(
            sensitivity["reco_energy_high"] - sensitivity["reco_energy_low"]
        ).to_value(u.TeV)
        / 2,
        ls="",
        label=label,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reconstructed energy / TeV")
    ax.set_ylabel(
        rf"$(E^2 \cdot \mathrm{{Flux Sensitivity}}) /$ ({unit.to_string('latex')})"
    )
    ax.set_title("Sensitivity")
    return 0


def plot_aeff(area, ax=None, label=None):
    ax.errorbar(
        0.5 * (area["ENERG_LO"] + area["ENERG_HI"]).to_value(u.TeV)[1:-1],
        area["EFFAREA"].to_value(u.m ** 2).T[1:-1, 0],
        xerr=0.5 * (area["ENERG_LO"] - area["ENERG_HI"]).to_value(u.TeV)[1:-1],
        ls="",
        label=label,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True energy / TeV")
    ax.set_ylabel("Effective collection area / m²")
    ax.set_title("Effective Area")
    return 0


def plot_edisp(edisp, ax=None, label=None):
    e_bins = edisp["ENERG_LO"][1:]
    migra_bins = edisp["MIGRA_LO"][1:]
    ax.pcolormesh(
        e_bins.to_value(u.TeV),
        migra_bins,
        edisp["MATRIX"].T[1:-1, 1:-1, 0].T,
        cmap="inferno",
        label=label,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_\mathrm{True} / \mathrm{TeV}$")
    ax.set_ylabel(r"$E_\mathrm{Reco} / E_\mathrm{True}$")
    ax.set_title("Energy Dispersion")
    return 0


def plot_angular_resolution(ang_res, ax=None, label=None):
    ax.errorbar(
        0.5
        * (ang_res["true_energy_low"] + ang_res["true_energy_high"]).to_value(u.TeV),
        ang_res["angular_resolution"].to_value(u.deg),
        xerr=0.5
        * (ang_res["true_energy_high"] - ang_res["true_energy_low"]).to_value(u.TeV),
        ls="",
        label=label,
    )

    ax.set_xscale("log")
    ax.set_xlabel("True energy / TeV")
    ax.set_ylabel("Angular Resolution / deg")
    ax.set_title("Angular Resolution")
    return 0


def plot_energy_bias_resolution(bias_resolution, ax=None):
    ax.errorbar(
        0.5
        * (
            bias_resolution["true_energy_low"] + bias_resolution["true_energy_high"]
        ).to_value(u.TeV),
        bias_resolution["resolution"],
        xerr=0.5
        * (
            bias_resolution["true_energy_high"] - bias_resolution["true_energy_low"]
        ).to_value(u.TeV),
        ls="",
    )
    ax.set_xscale("log")
    ax.set_xlabel("True energy / TeV")
    ax.set_ylabel("Energy Resolution")
    return 0


def plot_background(bg_rate, rad_max, ax=None, label=None):
    # pyirf data
    reco_bins = np.append(bg_rate["ENERG_LO"], bg_rate["ENERG_HI"][-1])

    # first fov bin, [0, 1] deg
    fov_bin = 0
    rate_bin = bg_rate["BKG"].T[:, fov_bin]

    # interpolate theta cut for given e reco bin
    e_center_bg = 0.5 * (bg_rate["ENERG_LO"] + bg_rate["ENERG_HI"])
    e_center_theta = 0.5 * (rad_max["ENERG_LO"] + rad_max["ENERG_HI"])
    theta_cut = np.interp(e_center_bg, e_center_theta, rad_max["RAD_MAX"].T[:, 0])

    # undo normalization
    rate_bin *= cone_solid_angle(theta_cut)
    rate_bin *= np.diff(reco_bins)

    ax.errorbar(
        0.5 * (bg_rate["ENERG_LO"] + bg_rate["ENERG_HI"]).to_value(u.TeV)[1:-1],
        rate_bin.to_value(1 / u.s)[1:-1],
        xerr=np.diff(reco_bins).to_value(u.TeV)[1:-1] / 2,
        ls="",
        label=label,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True energy / TeV")
    ax.set_ylabel("Background rate [1/s]")
    return 0


def plot_theta_cuts(theta_cuts, ax=None):
    ax.errorbar(
        0.5 * (theta_cuts["low"] + theta_cuts["high"]).to_value(u.TeV),
        theta_cuts["cut"].to_value(u.deg),
        xerr=0.5 * (theta_cuts["low"] - theta_cuts["high"]).to_value(u.TeV),
        ls="",
    )
    ax.set_xscale("log")
    ax.set_xlabel("True energy / TeV")
    ax.set_ylabel("θ-cut / deg²")
    return 0


def plot_gh_cuts(gh_cuts, ax=None):
    ax.errorbar(
        0.5 * (gh_cuts["low"] + gh_cuts["high"]).to_value(u.TeV),
        gh_cuts["cut"],
        xerr=0.5 * (gh_cuts["low"] - gh_cuts["high"]).to_value(u.TeV),
        ls="",
    )
    ax.set_xscale("log")
    ax.set_xlabel("True energy / TeV")
    ax.set_ylabel("G/H-Cut")
    return 0
