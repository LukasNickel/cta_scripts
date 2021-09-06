import numpy as np
import pandas as pd
from astropy.table import join, hstack
import astropy.units as u
from cta_tools.utils import ffill, remove_nans, add_units, rename_columns, QTable
from ctapipe.io import read_table
from pyirf.simulations import SimulatedEventsInfo
from ctapipe.instrument.subarray import SubarrayDescription
from astropy.time import Time
import logging


# wie in pyirf für event display
def read_to_pyirf(path):
    events = QTable(read_mc_dl2(path))
    sim_info = read_sim_info(path)
    return events, sim_info


def read_mc_dl2(path, drop_nans=True, rename=True):
    # dont rename yet to join tables together
    events = read_lst_dl2(path, drop_nans=drop_nans, rename=False)
    mc = read_table(path, "/simulation/event/subarray/shower")
    mc = add_units(mc)
    table = join(events, mc, join_type="left", keys=["obs_id", "event_id"])
    if rename:
        return rename_columns(table)
    return table


def read_mc_dl1(path, drop_nans=True, rename=True, images=False, tel_id=1):
    events = read_dl1(path, images=images, tel_id=tel_id, root="simulation")
    if drop_nans:
        events = remove_nans(events)
    mc = read_table(path, "/simulation/event/subarray/shower")
    mc = add_units(mc)
    events = join(events, mc, join_type="left", keys=["obs_id", "event_id"])
    if rename:
        return rename_columns(events)
    return events


def read_lst_dl1(path, drop_nans=True, rename=True, images=False, tel_id=1):
    """
    drop_nans and images dont play nicely together
    """
    events = read_dl1(path, images=images, tel_id=tel_id, root="dl1")
    if drop_nans:
        events = remove_nans(events)
    if rename:
        return rename_columns(events)
    return events


def read_dl1(path, images=False, tel_id=1, root="dl1"):
    tel = f"tel_{tel_id:03d}"
    events = read_table(path, f"/dl1/event/telescope/parameters/{tel}")
    pointing = read_table(path, f"/dl1/monitoring/telescope/pointing/{tel}")
    trigger = read_table(path, "/dl1/event/telescope/trigger")
    if images:
        images = read_table(path, f"/dl1/event/telescope/images/{tel}")
        assert len(events) == len(images)
        events = join(events, images, keys=["obs_id", "event_id"], join_type="left")
      # there are no magic numbers here. move on
    events["tel_id"] = tel_id
    events = join(
        events, trigger, keys=["obs_id", "event_id", "tel_id"], join_type="left"
    )
    # that changed at some point
    time_key = "time" if "time" in trigger.keys() else "telescopetrigger_time"
    #events = join(events, pointing, keys=time_key, join_type="left")
    # masked columns make everything harder
    events = events.filled(np.nan)
    events["azimuth"] = np.interp(
        events[time_key].mjd,
        pointing[time_key].mjd,
        pointing["azimuth"].quantity.to_value(u.deg)
    ) * u.deg
    events["altitude"] = np.interp(
        events[time_key].mjd,
        pointing[time_key].mjd,
        pointing["altitude"].quantity.to_value(u.deg)
    ) * u.deg
    #events["azimuth"] = ffill(events["azimuth"])
    #events["altitude"] = ffill(events["altitude"])
    
    # done in ctapipe?
    #events = add_units(events)

    events["time"] = Time(events[time_key], format="mjd", scale="tai")

    # this is failing because of broken header info for some reason
    subarray = SubarrayDescription.from_hdf(path)
    events["focal_length"] = subarray.tels[1].optics.equivalent_focal_length
    #events["focal_length"] = 28 * u.m
    return events


def read_lst_dl2(path, drop_nans=True, rename=True, tel_id=1):
    """
    lst1 only right now. could loop over tels or smth
    """
    tel = f"tel_{tel_id:03d}"
    energy = read_table(path, f"/dl2/event/telescope/gamma_energy_prediction/{tel}")
    gh = read_table(path, f"/dl2/event/telescope/gammaness/{tel}")
    disp = read_table(path, f"/dl2/event/telescope/disp_predictions/{tel}")
    pointing = read_table(path, "/dl1/monitoring/telescope/pointing/tel_001")
    trigger = read_table(path, "/dl1/event/telescope/trigger")
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

    events["azimuth"] = np.interp(
        events[time_key].mjd,
        pointing[time_key].mjd,
        pointing["azimuth"].quantity.to_value(u.rad)
    ) * u.rad
    events["altitude"] = np.interp(
        events[time_key].mjd,
        pointing[time_key].mjd,
        pointing["altitude"].quantity.to_value(u.rad)
    ) * u.rad
 

#    events["azimuth"] = ffill(events["azimuth"])
#    events["altitude"] = ffill(events["altitude"])
    events = add_units(events)

    events["x_prediction"].unit = u.m
    events["y_prediction"].unit = u.m
    events["alt_prediction"].unit = u.deg
    events["az_prediction"].unit = u.deg
    events["gamma_energy_prediction"].unit = u.TeV
    events["time"] = Time(events[time_key], format="mjd", scale="tai")

    # this is failing because of broken header info for some reason
    # subarray = SubarrayDescription.from_hdf(path)
    # events["focal_length"] = subarray.tels[1].optics.equivalent_focal_length
    events["focal_length"] = 28 * u.m

    if drop_nans:
        events = remove_nans(events)
    if rename:
        return rename_columns(events)
    return events


def read_dl3(path):
    events = read_table(path, 'EVENTS')
    pointings = read_table(path, 'POINTING')
    return join(events, pointings, keys='TIME')


def read_sim_info(path):
    run_info = read_table(path, "/configuration/simulation/run")
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


def save_plot_data(path, data_structure):
    with pd.HDFStore(path) as store:
        for plot_key, plot_dict in data_structure.items():
            for data_key, data in plot_dict.items():
                key = f"{plot_key}/{data_key}"
                store.put(key, data)


def read_plot_data(path, data_structure):
    result = {}
    with pd.HDFStore(path) as store:
        for plot_key, plot_dict in data_structure.items():
            result[plot_key] = {}
            for data_key, data in plot_dict.items():
                key = f"/{plot_key}/{data_key}"
                # structure does not need to be completely saved, eg gamma energy is only in dl2
                if key not in store.keys():
                    continue
                result[plot_key][data_key] = store.get(key)
    return result
