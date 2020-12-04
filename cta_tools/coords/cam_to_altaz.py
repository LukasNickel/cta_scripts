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
import astropy.units as u
erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))


def transform_predictions(data):
    location = EarthLocation.of_site('Roque de los Muchachos')

    # is this the mc time? does it even matter?
    obstime = Time('2013-11-01T03:00')
    #obstime = Time(data.dragon_time.values, scale='utc', format='unix')
    altaz = AltAz(location=location, obstime=obstime)
    array_pointing = SkyCoord(
        alt=data.columns['pointing_alt'], 
        az=data.columns['pointing_az'],
        frame=altaz,
    )
    camera_frame = CameraFrame(
        telescope_pointing=array_pointing,
        focal_length=data.columns['focal_length'],
        obstime=obstime,
        location=location,
    )
    cam_coords = SkyCoord(
        x=data.columns['source_x_prediction']*u.m,
        y=data.columns['source_y_prediction']*u.m,
        frame=camera_frame)
    source_prediction = cam_coords.transform_to(altaz)
    return source_prediction.alt, source_prediction.az




