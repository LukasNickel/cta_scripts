from cta_tools.io import read_lst_dl1
import numpy as np
from cta_tools.reco.theta import calc_wobble_thetas
from cta_tools.coords.transform import get_icrs_pointings, get_icrs_prediction
from cta_tools.io import read_lst_dl2
from cta_tools.utils import *
from astropy.io import fits
from pathlib import Path
from astropy.table import QTable
import pandas as pd
from pyirf.cuts import evaluate_binned_cut
from pyirf.binning import calculate_bin_indices
import operator
from aict_tools.io import read_data, append_predictions_cta
import pandas as pd
import click
import astropy.units as u
import tables
from astropy.time import Time, TimeDelta
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator

erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))


@click.command()
@click.argument("path")
def main(path):
    f = tables.open_file(path)
    if "delta_t" in f.root.dl1.monitoring.telescope:
        return 0
    else:
        f.close()
        data = read_lst_dl1(path, drop_nans=False)
        delta_t = TimeDelta(
            np.insert(np.diff(data["time"].mjd), 0, 0), format="jd", scale="tai"
        )
        print(delta_t)
        dt = pd.DataFrame(
            {
                "delta_t": delta_t.to(u.s),
                "obs_id": data["obs_id"],
                "event_id": data["event_id"],
            }
        )
        append_predictions_cta(path, dt, "/dl1/monitoring/telescope/delta_t")


if __name__ == "__main__":
    main()
