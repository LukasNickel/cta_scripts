import numpy as np
from lstchain.reco.utils import get_effective_time
import pandas as pd
from astropy.table import join, hstack
import astropy.units as u
from cta_tools.utils import ffill, remove_nans, add_units, rename_columns, QTable
from ctapipe.io import read_table
from pyirf.simulations import SimulatedEventsInfo
from ctapipe.instrument.subarray import SubarrayDescription
from lstchain.io.io import read_dl2_params, read_mc_dl2_to_QTable
from astropy.time import Time
import logging


log = logging.getLogger(__name__)


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


def read_mc_dl1(path, drop_nans=True, rename=True, images=False, tel_id="LST_LSTCam"):
    events = read_dl1(path, images=images, tel_id=tel_id, root="simulation")
    log.info(events.keys())
    if drop_nans:
        events = remove_nans(events)
    if not "mc_energy" in events.keys():
        mc = read_table(path, "/simulation/event/subarray/shower")
        mc = add_units(mc)
        events = join(events, mc, join_type="left", keys=["obs_id", "event_id"])
    if rename:
        return rename_columns(events)
    return events


def read_lst_dl1(path, drop_nans=True, rename=True, images=False, tel_id="LST_LSTCam"):
    """
    drop_nans and images dont play nicely together
    """
    events = read_dl1(path, images=images, tel_id=tel_id, root="dl1")
    if drop_nans:
        events = remove_nans(events)
    if rename:
        return rename_columns(events)
    return events


def read_dl1(path, images=False, tel_id="LST_LSTCam", root="dl1"):
    if isinstance(tel_id, int):
        tel = f"tel_{tel_id:03d}"
    else:
        tel = tel_id
    events = read_table(path, f"/dl1/event/telescope/parameters/{tel}")#, start=0, stop=1000000)
    # lstchain has a different scheme, lets just not use these for now
#     pointing = read_table(path, f"/dl1/monitoring/telescope/pointing/{tel}")
#     trigger = read_table(path, "/dl1/event/telescope/trigger")
    if images:
        images = read_table(path, f"/dl1/event/telescope/images/{tel}")
        assert len(events) == len(images)
        events = join(events, images, keys=["obs_id", "event_id"], join_type="left")
      # there are no magic numbers here. move on
#     events["tel_id"] = tel_id
#     events = join(
#         events, trigger, keys=["obs_id", "event_id", "tel_id"], join_type="left"
#     )
    # that changed at some point
    if "time" in events.keys():
        time_key = "time"
    elif "dragon_time" in events.keys():
        time_key = "dragon_time"
        events["time"] = Time(events[time_key], format="unix")
    else:
        time_key = "trigger_time"
        events["time"] = Time(events[time_key], format="unix")
    time_key = "time"
    return events
#     events = join(events, pointing, keys=time_key, join_type="left")
    # masked columns make everything harder
    events = events.filled(np.nan)
    alt_key = "altitude" if "altitude" in events.keys() else "alt_tel"
    az_key = "azimuth" if "azimuth" in events.keys() else "az_tel"
    events["azimuth"] = np.interp(
        events[time_key].mjd,
        pointing[time_key].mjd,
        pointing["az_key"].quantity.to_value(u.deg)
    ) * u.deg
    events["alt_key"] = np.interp(
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


def read_lst_dl2_runs(paths):
    observations = {}
    obstime = 0 * u.s
    for p in paths:
        data = read_data_dl2_to_QTable(path)
        data["time"] = Time(data["dragon_time"], format="mjd", scale="tai")
        log.info(f"Loading of {p} finished")
        log.info(f"{len(data)} events")
        observations[data[0]["obs_id"]] = data
        t_eff, t_elapsed = get_effective_time(data)
        log.info(f"Effective Observation time: {t_eff}")
        obstime += t_eff 
    log.info(f"Combined observation time: {obstime.to(u.min):.2f}")
    return observations


def read_dl3(path):
    events = read_table(path, 'EVENTS')
    pointings = read_table(path, 'POINTING')
    return join(events, pointings, keys='TIME')


def read_sim_info(path):
    key = "/simulation/run_config"
    log.info(f"Reading sim info for file {path} and key {key}")
    run_info = read_table(path, key)
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
        for feature in store.keys():
            k = ["bins", "values"]
            result[plot_key] = {}
            for data_key in k:
                key = f"/{feature}/{data_key}"
                result[feature][data_key] = store.get(key)
    return result

