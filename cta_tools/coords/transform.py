from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord, AltAz, ICRS
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.coordinates import (
    GroundFrame,
    TiltedGroundFrame,
    NominalFrame,
    TelescopeFrame,
    CameraFrame,
)
import astropy.units as u

from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator

erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))


location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)


def get_icrs_pointings(data):
    obstime = data["time"]
    altaz = AltAz(location=location, obstime=obstime)
    pointing = SkyCoord(
        alt=data["pointing_alt"],
        az=data["pointing_az"],
        frame=altaz,
    )
    return pointing.transform_to("icrs")


def get_altaz_prediction(data):
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
    return cam_coords.transform_to(altaz)


def get_icrs_prediction(data):
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
        alt=data["reco_alt"], az=data["reco_az"], frame=altaz
    )
    return cam_coords.transform_to("icrs")
