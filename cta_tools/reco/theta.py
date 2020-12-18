import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
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

    # maybe this is wrong?
    r = source_coord.separation(pointing_coord)
    phi0 = np.arctan2(source_fov.lat, source_fov.lon).to_value(u.rad)

    theta_offs = []
    for off in range(1, n_off + 1):
        new_phi = phi0 + 2 * np.pi * off / (n_off + 1)
        off_pos = SkyCoord(
            lon=r * np.sin(new_phi),
            lat=r * np.cos(new_phi),
            frame=fov_frame,
        )
        theta_offs.append(off_pos.separation(reco_fov).to_value(u.deg))

    return reco_coord.separation(source_coord).to_value(u.deg), theta_offs


def calc_theta_off_cam(
        source_coord: SkyCoord,
        reco_coord: SkyCoord,
        n_off=5
):
    "Only use cam frames here"
    dist_on = (
        (source_coord.x.to_value(u.m) - reco_coord.x.to_value(u.m))**2
        + (source_coord.y.to_value(u.m) - reco_coord.y.to_value(u.m))**2
    )
    r = np.sqrt(source_coord.x.to_value(u.m)**2 + source_coord.y.to_value(u.m)**2)
    phi = np.arctan2(source_coord.y.to_value(u.m), source_coord.x.to_value(u.m))

    theta_on = np.rad2deg(np.sqrt(dist_on) / 28)

    theta_offs = []
    for i in range(1, n_off + 1):
        x_off = r * np.cos(phi + i * 2 * np.pi / (n_off + 1))
        y_off = r * np.sin(phi + i * 2 * np.pi / (n_off + 1))
        dist_off = (
            (x_off.to_value(u.m) - reco_coord.x.to_value(u.m))**2
            + (y_off.to_value(u.m) - reco_coord.y.to_value(u.m))**2
        )
        theta_offs.append(
            np.rad2deg(np.sqrt(dist_off) / 28)
        )

    return theta_on, theta_offs
