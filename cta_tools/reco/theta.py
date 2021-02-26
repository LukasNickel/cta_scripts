from ctapipe.coordinates import CameraFrame
import numpy as np
from astropy.io import fits
from pathlib import Path
from astropy.table import QTable
import pandas as pd
from pyirf.cuts import evaluate_binned_cut
from pyirf.binning import calculate_bin_indices
import operator
from aict_tools.io import read_data
import click
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, SkyOffsetFrame
from astropy.time import Time
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator

erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))
from cta_tools.coords.transform import (
    get_icrs_pointings,
    get_altaz_prediction,
    get_icrs_prediction,
)


location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)


def calc_theta(data, source=SkyCoord.from_name("CRAB_NEBULA")):
    altaz = AltAz(location=location, obstime=data["time"])
    return get_icrs_prediction(data).separation(source).to_value(u.deg)


def calc_mc_theta(data):
    altaz = AltAz(location=location, obstime=data["time"])
    source = SkyCoord(
        alt=data["true_alt"],
        az=data["true_az"],
        frame=altaz,
    )
    return calc_theta(data, source)


def calc_wobble_thetas(data, source=SkyCoord.from_name("CRAB_NEBULA"), n_off=5):
    altaz = AltAz(location=location, obstime=data["time"])
    pointing = SkyCoord(
        alt=data["pointing_alt"],
        az=data["pointing_az"],
        frame=altaz,
    )
    camera_frame = CameraFrame(
        telescope_pointing=pointing,
        focal_length=data["focal_length"],
        obstime=data["time"],
        location=location,
    )
    cam_coords = SkyCoord(
        x=data["source_x_pred"], y=data["source_y_pred"], frame=camera_frame
    )
    icrs_preds = cam_coords.transform_to("icrs")
    source_cam = source.transform_to(camera_frame)

    r = np.sqrt((source_cam.x) ** 2 + (source_cam.y) ** 2)
    phi0 = np.arctan2(source_cam.y.to_value(u.m), source_cam.x.to_value(u.m))

    theta_on = icrs_preds.separation(source.transform_to("icrs")).to_value(u.deg)
    theta_offs = []
    for off in range(1, n_off + 1):
        new_phi = phi0 + 2 * np.pi * off / (n_off + 1)
        off_pos = SkyCoord(
            x=r * np.cos(new_phi),
            y=r * np.sin(new_phi),
            frame=camera_frame,
        ).transform_to("icrs")
        theta_offs.append(off_pos.separation(icrs_pred).to_value(u.deg))
    return theta_on, theta_offs
