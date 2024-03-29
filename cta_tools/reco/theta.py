import numpy as np
import astropy.units as u
import logging
from astropy.coordinates import SkyCoord, AltAz, EarthLocatione
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator

erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))

log = logging.getLogger(__name__)

location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)
crab_nebula = SkyCoord(ra=83.63308333 * u.deg, dec=22.0145 * u.deg, frame="icrs")


def calc_wobble_thetas(data, source=crab_nebula, n_off=5):
    if "RA_PNT" in data.keys():
        icrs_pointings = SkyCoord(
            ra=data["RA_PNT"],
            dec=data["DEC_PNT"],
            frame="icrs",
        )
        icrs_preds = SkyCoord(
            ra=data["RA"],
            dec=data["DEC"],
            frame="icrs",
        )
    else:
        altaz = AltAz(location=location, obstime=data["time"])
        pointing = SkyCoord(
            alt=data["pointing_alt"],
            az=data["pointing_az"],
            frame=altaz,
        )
        reco_coords = SkyCoord(alt=data["reco_alt"], az=data["reco_az"], frame=altaz)
        icrs_preds = reco_coords.transform_to("icrs")
        icrs_pointings = pointing.transform_to("icrs")
    log.info(f"Pointings are at icrs {icrs_pointings}")
    log.info(f"Predictions are at {reco_coords}")
    log.info(f"Predictions are at icrs {icrs_preds}")
    icrs_source = source.transform_to("icrs")
    log.info(f"Source is at icrs location: {icrs_source}")
    wobble_offset = icrs_pointings.separation(icrs_source)
    log.info(f"Wobble offset set to {wobble_offset}")

    source_angle = icrs_pointings.position_angle(icrs_source)
    off_positions = icrs_pointings.directional_offset_by(
        separation=wobble_offset,
        position_angle=source_angle
        + np.arange(360 / (n_off + 1), 360, 360 / (n_off + 1))[:, None] * u.deg,
    )

    theta_on = icrs_preds.separation(icrs_source).to(u.deg)
    theta_offs = []
    for off_pos in off_positions:
        theta_offs.append(off_pos.separation(icrs_preds).to(u.deg))

    return theta_on, theta_offs
