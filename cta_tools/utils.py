import numpy as np
import astropy.units as u


PYIRF_COLUMN_MAP = {
    "gamma_energy_prediction": "reco_energy",
    "altitude": "pointing_alt",
    "azimuth": "pointing_az",
    "gammaness": "gh_score",
}


OGADF_COLUMN_MAP = {
    "gamma_energy_prediction": "ENERGY",
    "reco_energy": "ENERGY",
    "altitude": "ALT_PNT",
    "azimuth": "AZ_PNT",
    "reco_az": "AZ",
    "reco_alt": "ALT",
    "pointing_alt": "AZ_PNT",
    "pointing_az": "ALT_PNT",
    "reco_energy": "ENERGY",
    "time": "TIME",
}


UNITS = {
    "true_az": u.deg,
    "true_alt": u.deg,
    "reco_az": u.deg,
    "reco_alt": u.deg,
    "pointing_alt": u.rad,
    "pointing_az": u.rad,
    "altitude": u.rad,
    "azimuth": u.rad,
    "true_energy": u.TeV,
}


def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return arr[idx]


def remove_nans(astro_table):
    is_nan = np.zeros(len(astro_table), dtype=bool)
    for col in astro_table.itercols():
        if col.info.dtype.kind == "f":
            is_nan |= np.isnan(col)
    return astro_table[~is_nan]


def rename_columns(astro_table, map_type="pyirf"):
    if map_type == "pyirf":
        col_map = PYIRF_COLUMN_MAP
    elif map_type == "ogadf":
        col_map = OGADF_COLUMN_MAP
    else:
        raise Exception("no such map", map_type)
    for old, new in col_map.items():
        if old in astro_table.columns:
            astro_table.rename_column(old, new)
    return astro_table


def add_units(astro_table):
    for key, value in astro_table.meta.items():
        if "UNIT" in key and key.split("_UNIT")[0] in astro_table.columns:
            astro_table[key.split("_UNIT")[0]].unit = value
    # ghetto hack to fix missing metadata
    for key, value in UNITS.items():
        try:
            astro_table[key].unit = value
        except Exception as e:
            print(e)
    return astro_table
