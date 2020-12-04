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



def calc_theta_off(
        source_coord: SkyCoord,
        reco_coord: SkyCoord,
        pointing_coord: SkyCoord,
        n_off=5
):
    fov_frame = SkyOffsetFrame(origin=pointing_coord)
    source_fov = source_coord.transform_to(fov_frame)
    reco_fov = reco_coord.transform_to(fov_frame)

    r = source_coord.separation(pointing_coord)
    phi0 = np.arctan2(source_fov.lat, source_fov.lon).to_value(u.rad)

    theta_offs = []
    for off in range(1, n_off + 1):

        off_pos = SkyCoord(
            lon=r * np.sin(phi0 + 2 * np.pi * off / (n_off + 1)),
            lat=r * np.cos(phi0 + 2 * np.pi * off / (n_off + 1)),
            frame=fov_frame,
        )

        theta_offs.append(off_pos.separation(reco_fov).to_value(u.deg))

    return reco_coord.separation(source_coord).to_value(u.deg), theta_offs
