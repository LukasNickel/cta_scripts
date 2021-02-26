import numpy as np
import h5py
import pandas as pd
from aict_tools.io import read_data
from astropy.table import QTable
import astropy.units as u
from cta_tools.coords.cam_to_altaz import transform_predictions
from cta_tools.utils import ffill, remove_nans, add_units, rename_columns
from astropy.table.column import Column, MaskedColumn
from pyirf.simulations import SimulatedEventsInfo
import logging


# wie in pyirf für event display
def read_to_pyirf(path):

    events = read_mc_dl2(path)
    dist_cog = dist_pred = np.sqrt(
        (df["x"] - df["src_x"]) ** 2 + (df["y"] - df["src_y"]) ** 2
    )
    dist_pred = dist_pred = np.sqrt(
        (df["source_x_prediction"] - df["src_x"]) ** 2
        + (df["source_y_prediction"] - df["src_y"]) ** 2
    )
    df["sign_correct"] = dist_pred < dist_cog

    events.columns["reco_alt"], events.columns["reco_az"] = transform_predictions(
        events
    )

    sim_info = read_sim_info(infile)
    return events, sim_info


def read_mc_dl2(path, drop_nans=True, rename=True):
    # dont rename yet to join tables together
    events = read_lst_dl2(path, drop_nans=drop_nans, rename=False)
    mc = QTable.read(path, "/simulation/event/subarray/shower")
    mc = add_units(mc)
    table = join(events, mc, join_type="left", keys=["obs_id", "event_id"])
    if rename:
        return rename_columns(table)
    return table


def read_lst_dl2(path, drop_nans=True, rename=True):
    """
    lst1 only right now. could loop over tels or smth
    """
    energy = QTable.read(path, f"/dl2/event/telescope/tel_001/gamma_energy_prediction")
    gh = QTable.read(path, f"/dl2/event/telescope/tel_001/gammaness")
    disp = QTable.read(path, f"/dl2/event/telescope/tel_001/disp_predictions")
    pointing = QTable.read(path, f"/dl1/monitoring/telescope/pointing/tel_001")
    trigger = QTable.read(path, f"/dl1/event/telescope/trigger")
    events = join(energy, gh, keys=["obs_id", "event_id"])
    events = join(events, disp, keys=["obs_id", "event_id"])
    # there are no magic numbers here. move on
    events["tel_id"] = 1
    events = join(
        events, trigger, keys=["obs_id", "event_id", "tel_id"], join_type="left"
    )
    time_key = "time" if "time" in trigger.keys() else "telescopetrigger_time"
    events = join(events, pointing, keys=time_key, join_type="left")
    # masked columns make everything harder
    events = events.filled(np.nan)
    events["azimuth"] = ffill(events["azimuth"])
    events["altitude"] = ffill(events["altitude"])
    events = add_units(events)

    events["source_x_pred"].unit = u.m
    events["source_y_pred"].unit = u.m
    events["gamma_energy_prediction"].unit = u.TeV
    events["time"] = Time(events[time_key], format="mjd", scale="tai")

    subarray = SubarrayDescription.from_hdf(f)
    events["focal_length"] = subarray.tels[1].optics.equivalent_focal_length

    if drop_nans:
        events = remove_nans(events)
    if rename:
        return rename_columns(events)
    return events


def read_sim_info(path):
    run_info = QTable.read(path, "/configuration/simulation/run")
    e_min = np.unique(run_info.columns["energy_range_min"])
    assert len(e_min) == 1
    e_max = np.unique(run_info.columns["energy_range_max"])
    assert len(e_max) == 1
    max_impact = np.unique(run_info.columns["max_scatter_range"])
    assert len(max_impact) == 1
    index = np.unique(run_info.columns["spectral_index"])
    assert len(index) == 1
    view_cone = np.unique(run_info.columns["max_viewcone_radius"])
    assert len(view_cone) == 1

    sim_info = SimulatedEventsInfo(
        n_showers=(
            run_info.columns["shower_reuse"] * run_info.columns["num_showers"]
        ).sum(),
        energy_min=u.Quantity(e_min[0], u.TeV),
        energy_max=u.Quantity(e_max[0], u.TeV),
        max_impact=u.Quantity(max_impact[0], u.m),
        spectral_index=index[0],
        viewcone=u.Quantity(view_cone[0], u.deg),
    )

    return sim_info


def read_table(path, columns=None, key="events"):
    """
    Internal helper to use for files before and after converting to fits.
    """
    try:
        df = read_data(
            path,
            key=key,
            columns=columns,
        )
        table = QTable.from_pandas(df)
        if "gamma_energy_prediction" in table.keys():
            table["gamma_energy_prediction"].unit = u.TeV
        return table
    except:
        t = QTable.read(path, "EVENTS")
        if columns:
            return t[columns]
        else:
            return t
