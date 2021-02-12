import tables
from aict_tools.cta_helpers import horizontal_to_camera_cta_simtel
import numpy as np
import pandas as pd
from aict_tools.io import read_data
from astropy.table import QTable
import astropy.units as u
from cta_tools.coords.cam_to_altaz import transform_predictions
from pyirf.simulations import SimulatedEventsInfo

# wie in pyirf für event display
def read_to_pyirf(infile):

    COLUMN_MAP = {
        "true_energy": ("mc_energy", u.TeV),
        "reco_energy": ("gamma_energy_prediction", u.TeV),
        "true_alt": ("mc_alt", u.rad),
        "true_az": ("mc_az", u.rad),
        "pointing_alt": ("alt_tel", u.rad),
        "pointing_az": ("az_tel", u.rad),
        "focal_length": ("focal_length", u.m),
        "gh_score": ("gammaness", ),
    }

    df = read_data(infile, 'events').dropna(subset=['source_x_prediction', 'source_y_prediction', 'gamma_energy_prediction', "gammaness"])
    dist_cog = dist_pred = np.sqrt((df['x'] - df['src_x'])**2 + (df['y'] - df['src_y'])**2)
    dist_pred = dist_pred = np.sqrt((df['source_x_prediction'] - df['src_x'])**2 + (df['source_y_prediction'] - df['src_y'])**2)
    df['sign_correct'] = (dist_pred < dist_cog)
    events = QTable.from_pandas(df)    

    for new, in_ in COLUMN_MAP.items():
        events.rename_column(in_[0], new)
        if len(in_) > 1:
            events.columns[new].unit = in_[1]
    
    events.columns['reco_alt'], events.columns['reco_az'] = transform_predictions(events)
    events['gh_score'].fill_value = -1
    events['reco_energy'].fill_value = -1

    run_info = QTable.from_pandas(read_data(infile, 'runs'))
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
        n_showers=(run_info.columns["shower_reuse"] * run_info.columns["num_showers"]).sum(),
        energy_min=u.Quantity(e_min[0], u.TeV),
        energy_max=u.Quantity(e_max[0], u.TeV),
        max_impact=u.Quantity(max_impact[0], u.m),  #??
        spectral_index=index[0],
        viewcone=u.Quantity(view_cone[0], u.deg), #??
    )

    return events.filled(), sim_info


def load_dl1_sim_info(path):
    siminfo = QTable.read(path, '/configuration/simulation/run')
    for c in siminfo.columns:
        if c.startswith('corsika') or c.startswith('min') or c.startswith('max') or c.startswith('energy') or c.startswith('simtel') or c.startswith('spectral'):
            assert len(np.unique(siminfo[c])) == 1
    return SimulatedEventsInfo(
        n_showers = np.sum(siminfo['num_showers'] * siminfo['shower_reuse']),
        energy_min = u.Quantity(siminfo['energy_range_min'][0], siminfo.meta['energy_range_max_UNIT'].decode()),
        energy_max = u.Quantity(siminfo['energy_range_max'][0], siminfo.meta['energy_range_min_UNIT'].decode()),
        max_impact = u.Quantity(siminfo['max_scatter_range'][0], siminfo.meta['max_scatter_range_UNIT'].decode()),
        spectral_index = siminfo['spectral_index'][0],
        viewcone = u.Quantity(siminfo['max_viewcone_radius'][0], siminfo.meta['max_viewcone_radius_UNIT'].decode()),
    )


def load_dl2_events(path):
    ## load aict generated tables
    events = None

    return events
