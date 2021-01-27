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
from cta_tools.reco.theta import calc_theta_off
from cta_tools.reco.theta import calc_theta_off_cam


def wobble_predictions_lst(df, source='crab', n_off=5):

    location = EarthLocation.from_geodetic(
        -17.89139 * u.deg,
        28.76139 * u.deg,
        2184 * u.m
    )
    if 'dragon_time' in df.keys():
        obstime = Time(df.dragon_time.values, format='unix')
    else:
        obstime = Time('2013-11-01T03:00')
        print('Using MC default time')
    altaz = AltAz(location=location, obstime=obstime)

    pointing = SkyCoord(
        alt=u.Quantity(df.alt_tel.values, u.rad, copy=False),
        az=u.Quantity(df.az_tel.values*u.rad, copy=False),
        frame=altaz,
    )
    pointing_icrs = pointing.transform_to('icrs')

    camera_frame = CameraFrame(
        telescope_pointing=pointing,
        focal_length=28*u.m,
        obstime=obstime,
        location=location,
    )
    cam_pred = SkyCoord(
        x=df.source_x_prediction.values*u.m,
        y=df.source_y_prediction.values*u.m,
        frame=camera_frame
    )
    icrs_pred = cam_pred.transform_to('icrs')

    src = SkyCoord.from_name(source)
    theta_on, theta_offs = calc_theta_off(src, icrs_pred, pointing_icrs, n_off=n_off)
    #theta_on, theta_offs = calc_theta_off_cam(
    #    src.transform_to(camera_frame),
    #    cam_pred,
    #    n_off=n_off
    #)
    return (
        icrs_pred.ra.to_value(u.deg),
        icrs_pred.dec.to_value(u.deg),
        theta_on,
        theta_offs,
        pointing_icrs.ra,
        pointing_icrs.dec,
    )

