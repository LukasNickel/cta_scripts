import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz
from astropy.time import Time
import logging
import warnings

from astropy.coordinates import SkyCoord, AltAz
from ctapipe.coordinates import (
        NominalFrame,
        CameraFrame,
        TiltedGroundFrame,
        project_to_ground,
        GroundFrame,
        MissingFrameAttributeWarning
)


def camera_to_horizontal(x, y, alt_pointing, az_pointing, focal_length):
    ''' Everything should be quantities!
    Only for MC right now, as the frame is fixed in time 
    '''

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=MissingFrameAttributeWarning)
        altaz = AltAz()

        tel_pointing = SkyCoord(
            alt=alt_pointing,
            az=az_pointing,
            frame=altaz,
        )

        frame = CameraFrame(
            focal_length = focal_length,
            telescope_pointing=tel_pointing,
        )


        cam_coords = SkyCoord(
            x=x,
            y=y,
            frame=frame,
        )

        source_altaz = cam_coords.transform_to(altaz)

        # rad verwenden? 
        return source_altaz.alt.to_value(u.deg), source_altaz.az.to_value(u.deg)
