import tables
import astropy.units as u
from astropy.coordinates import (
    SkyCoord,
    CameraFrame,
    AltAz
)


def camera_to_horizontal(x, y, alt_pointing, az_pointing, focal_length):
    altaz = AltAz()
    tel_pointing = SkyCoord(
        alt=u.Quantity(alt_pointing, u.deg, copy=False),
        az=u.Quantity(az_pointing, u.deg, copy=False),
        frame=altaz,
    )

