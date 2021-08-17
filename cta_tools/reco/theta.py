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
crab_nebula = SkyCoord(ra=83.63308333 * u.deg, dec=22.0145 * u.deg, frame="icrs")


def calc_theta(data, source=crab_nebula):
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


def calc_wobble_thetas(data, source=crab_nebula, n_off=5):
    if 'RA_PNT' in data.keys():
        icrs_pointings = SkyCoord(
            ra=data["RA_PNT"],
            dec=data["DEC_PNT"],
            frame='icrs',
        )
        icrs_preds = SkyCoord(
            ra=data["RA"],
            dec=data["DEC"],
            frame='icrs',
        )
    else:
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
        icrs_pointings = pointing.transform_to('icrs')
    icrs_source = source.transform_to('icrs')
    
    wobble_offset = icrs_pointings.separation(icrs_source)
 
    source_angle = icrs_pointings.position_angle(icrs_source)
    off_positions = icrs_pointings.directional_offset_by(
        separation=wobble_offset,
        position_angle=source_angle + np.arange(360 / (n_off + 1), 360, 360 / (n_off + 1))[:, None] * u.deg
    )

    theta_on = icrs_preds.separation(icrs_source).to(u.deg)
    theta_offs = []
    for off_pos in off_positions:
        theta_offs.append(off_pos.separation(icrs_preds).to(u.deg))

    return theta_on, theta_offs
