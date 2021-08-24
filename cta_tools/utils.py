import numpy as np
import astropy.units as u
from astropy.table import Table, QTable
import pandas as pd


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


def bin_df(df, bin_column, target_column, bins=None, log=True):
    if isinstance(df, Table):
        df = df.to_pandas()
    else:
        df = df.copy()
    if bins is None:
        if log:
            bins = np.logspace(np.log10(1), np.log10(np.nanmax(df[bin_column])), 50)
        else:
            bins = np.linspace(np.nanmin(df[bin_column]), np.nanmax(df[bin_column]), 50)
    df["bin"] = np.digitize(df[bin_column], bins)
    grouped = df.groupby("bin")
    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned["center"] = 0.5 * (bins[:-1] + bins[1:])
    binned["width"] = np.diff(bins)
    binned["mean"] = grouped[target_column].mean()
    binned["std"] = grouped[target_column].std()
    return binned


def get_value(df, column):
    values = df[column]
    if isinstance(df, QTable):
        if values.unit:
            values = values.value
        else:
            values = values.data
    if isinstance(df, Table):
        values = values.data

    return values