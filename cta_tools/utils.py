import numpy as np

PYIRF_COLUMN_MAP = {
    "gamma_energy_prediction": "reco_energy",
    "altitude": "pointing_alt",
    "azimuth": "pointing_az",
    "gammaness": "gh_score",
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
    else:
        raise Exception("no such map", map_type)
    for old, new in PYIRF_COLUMN_MAP.items():
        if old in astro_table.columns:
            astro_table.rename_column(old, new)
    return astro_table


def add_units(astro_table):
    for key, value in astro_table.meta.items():
        if "UNIT" in key and key.split("_UNIT")[0] in astro_table.columns:
            astro_table[key.split("_UNIT")[0]].unit = value
    return astro_table
