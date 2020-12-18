import numpy as np
from cta_tools.reco.wobble import wobble_predictions_lst
from astropy.io import fits
from pathlib import Path
from astropy.table import QTable
import pandas as pd
from pyirf.cuts import evaluate_binned_cut
from pyirf.binning import calculate_bin_indices
import operator
from aict_tools.io import read_data
import click
import astropy.units as u
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))


DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["CREATOR"] = f"Lukas Nickel"
# fmt: off
DEFAULT_HEADER["HDUDOC"] = (
    "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
)
DEFAULT_HEADER["HDUVERS"] = "0.2"
DEFAULT_HEADER["HDUCLASS"] = "GADF"


@click.command()
@click.argument('pattern')
@click.argument('cut_file')
@click.argument('output')
@click.option('--theta/--no-theta', default=True)
def main(pattern, cut_file, output, theta):
    files = Path(pattern).parent.glob(pattern.split('/')[-1])
    gh_cuts = QTable.read(cut_file, hdu="GH_CUTS")
    theta_cuts = QTable.read(cut_file, hdu="THETA_CUTS_OPT")

    observations = []
    index = []
    livetimes = []
    ontimes = []
    starts = []
    stops = []

    # pretty random, might be too long
    gti_thresh = 10
    for i, f in enumerate(files):
        df = read_data(f, 'events')
        tstart = df['dragon_time'].min()
        tstop = df['dragon_time'].max()
        ontime = (tstop-tstart)
        # from IPython import embed; embed()
        # add a dummy time before the first event
        timedeltas = np.diff(
            np.insert(
                df['dragon_time'].sort_values().values,
                0,
                0,
            )
        )
        large_deltas = (np.abs(timedeltas) > gti_thresh)
        gti_indices = np.split(
            np.arange(len(df)),
            np.where(large_deltas)[0],
        )
        # first entry will always be empty so we remove it 
        del gti_indices[0]
        gti_starts = []
        gti_stops = []
        livetime = 0
        for i in gti_indices:
            start = df['dragon_time'].iloc[i[0]]
            stop = df['dragon_time'].iloc[i[-1]]
            gti_starts.append(start)
            gti_stops.append(stop)
            livetime += (stop-start)

        # for the index table
        starts.append(tstart)
        stops.append(tstop)
        ontimes.append(ontime)
        livetimes.append(livetime)

        df.dropna(
            subset=['gamma_energy_prediction', 'gammaness', 'disp_prediction'],
            inplace=True
        )
        nevents = len(df)
        print("\nEvents in run: ", nevents)
        df['selected'] = True
        #df['selected'] = (df['gammaness'] > 0.6)
        gh_mask = evaluate_binned_cut(
            df['gammaness'],
            u.Quantity(df['gamma_energy_prediction'].values, u.TeV, copy=False),
            gh_cuts,
            operator.ge
        )
        df['selected'] &= gh_mask
        df = df[df['selected']]
        ngh = len(df)
        print('Events nach g/h cut: ', ngh, ngh/nevents*100, "%")

        pred = wobble_predictions_lst(df)
        df['ra_pred'], df['dec_pred'] = pred[0], pred[1]
        df['theta_on'], theta_offs = pred[2], pred[3]
        thetas = np.array(theta_offs + [pred[2]])
        df['ra_pnt'], df['dec_pnt'] = pred[4], pred[5]

        # theta cuts always on min distance
        # assuming cuts are smaller than off size distance
        theta_mask = evaluate_binned_cut(
            u.Quantity(thetas.min(axis=0), u.deg, copy=False),
            u.Quantity(df['gamma_energy_prediction'].values, u.TeV, copy=False),
            theta_cuts,
            operator.le
        )
        ebins = np.append(theta_cuts["low"], theta_cuts["high"][-1])
        bin_index = calculate_bin_indices(
            u.Quantity(
                df['gamma_energy_prediction'].values,
                u.TeV,
                copy=False),
            ebins
        )
        df['theta_cut_value'] = theta_cuts["cut"][bin_index]
        event_columns = {
            'ra_pred': 'RA',
            'dec_pred': 'DEC',
            'event_id': 'EVENT_ID',
            'dragon_time': 'TIME',
            'gamma_energy_prediction': 'ENERGY',
            'theta_on': 'THETA_ON',
            'theta_cut_value': 'THETA_CUT_VALUE',
        }
        for i, thetas in enumerate(theta_offs):
            df[f'theta_off{i}'] = thetas
            event_columns[f'theta_off{i}'] = f'THETA_OFF{i}'

        if theta:
            df = df[theta_mask]
            ntheta = len(df)
            print("Events nach theta cut: ", ntheta, ntheta/nevents*100, "%")
        pointing_columns_old = ['dragon_time', 'ra_pnt', 'dec_pnt']
        pointing_columns_new = ['TIME', 'RA_PNT', 'DEC_PNT']

        df_events = df[event_columns.keys()]
        df_events.rename(columns=event_columns, inplace=True)

        df_pointings = df[pointing_columns_old]
        df_pointings.columns = pointing_columns_new

        df_gti = pd.DataFrame()

        # from IPython import embed; embed()
        df_gti['START'] = gti_starts
        df_gti['STOP'] = gti_stops

        events = QTable.from_pandas(df_events)
        events['RA'].unit = u.deg
        events['DEC'].unit = u.deg
        events['TIME'].unit = u.s
        events['ENERGY'].unit = u.TeV
        event_header = DEFAULT_HEADER.copy()
        event_header['HDUCLAS1'] = 'EVENTS'
        event_header['OBS_ID'] = df['obs_id'].iloc[0]
        event_header['TSTART'] = tstart
        event_header['TSTOP'] = tstop

        event_header['MJDREFI'] = 40587  # ref time is this correct? 01.01.1970?
        event_header['MJDREFF'] = 0.
        event_header['ONTIME'] = ontime
        event_header['LIVETIME'] = livetime
        event_header['DEADC'] = (livetime/ontime)
        # assuming constant pointing
        event_header['RA_PNT'] = df_pointings['RA_PNT'].iloc[0]
        event_header['DEC_PNT'] = df_pointings['DEC_PNT'].iloc[0]
        event_header['EQUINOX'] = '2000.0'
        event_header['RADECSYS'] = 'FK5'  # ???
        event_header['ORIGIN'] = 'CTA'
        event_header['TELESCOP'] = 'LST1'
        event_header['INSTRUME'] = 'LST1'

        gtis = QTable.from_pandas(df_gti)
        gtis['START'].unit = u.s
        gtis['STOP'].unit = u.s
        gti_header = DEFAULT_HEADER.copy()
        gti_header['MJDREFI'] = 40587  # ref time is this correct? 01.01.1970?
        gti_header['MJDREFF'] = 0.
        gti_header['TIMEUNIT'] = 's'
        gti_header['TIMESYS'] = 'UTC'  # ??
        gti_header['TIMEREF'] = 'TOPOCENTER'  # ??

        pointings = QTable.from_pandas(df_pointings)
        pointings['RA_PNT'].unit = u.deg
        pointings['DEC_PNT'].unit = u.deg
        pointings['TIME'].unit = u.s
        pointing_header = gti_header.copy()
        pointing_header['OBSGEO-L'] = -17.89139
        pointing_header['OBSGEO-B'] = 28.76139
        pointing_header['OBSGEO-H'] = 2184.

        hdus = [
            fits.PrimaryHDU(),
            fits.BinTableHDU(events, header=event_header, name="EVENTS"),
            fits.BinTableHDU(pointings, header=pointing_header, name="POINTING"),
            fits.BinTableHDU(gtis, header=gti_header, name="GTI")
        ]

        fits.HDUList(hdus).writeto(
            Path(output).parent/f'{df["obs_id"].iloc[0]}.fits.gz',
            overwrite=True
        )
        observations.append((
            df['obs_id'].iloc[0],
            df_pointings['RA_PNT'].iloc[0],
            df_pointings['DEC_PNT'].iloc[0],
            tstart,
            tstop,
            livetime/ontime
        ))

        index.append((
            df['obs_id'].iloc[0],
            'events',
            'events',
            '.',
            f'{df["obs_id"].iloc[0]}.fits.gz',
            'EVENTS'
        ))
        index.append((
            df['obs_id'].iloc[0],
            'gti',
            'gti',
            '.',
            f'{df["obs_id"].iloc[0]}.fits.gz',
            'GTI'
        ))
        index.append((
            df['obs_id'].iloc[0],
            'aeff',
            'aeff_2d',
            '.',
            'irfs.fits.gz',
            'EFFECTIVE_AREA'
        ))
        index.append((
            df['obs_id'].iloc[0],
            'psf',
            'psf_table',
            '.',
            'irfs.fits.gz',
            'PSF'
        ))
        index.append((
            df['obs_id'].iloc[0],
            'edisp',
            'edisp_2d',
            '.',
            'irfs.fits.gz',
            'ENERGY_DISPERSION'
        ))
        index.append((
            df['obs_id'].iloc[0],
            'bkg',
            'bkg_2d',
            '.',
            f'{df["obs_id"].iloc[0]}.fits.gz',
            'BACKGROUND'
        ))

    # build index file
    observation_table = QTable(
        rows=observations,
        names=['OBS_ID', 'RA_PNT', 'DEC_PNT', 'TSTART', 'TSTOP', 'DEADC'],
        units=['', 'deg', 'deg', 's', 's', '']
    )
    obs_header = DEFAULT_HEADER.copy()
    obs_header['HDUCLAS1'] = 'INDEX'
    obs_header['HDUCLAS2'] = 'OBS'
    obs_header['TELESCOP'] = 'LST1'
    obs_header['MJDREFI'] = 40587  # ref time is this correct? 01.01.1970?
    obs_header['MJDREFF'] = 0.
    obs_header['ONTIME'] = sum(ontimes)
    obs_header['LIVETIME'] = sum(livetimes)
    obs_header['DEADC'] = (sum(livetimes) / sum(ontimes))
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(observation_table, header=obs_header, name="OBS_INDEX"),
    ]
    fits.HDUList(hdus).writeto(Path(output).parent/'obs-index.fits.gz', overwrite=True)

    index_table = QTable(
        rows=index,
        names=[
            'OBS_ID',
            'HDU_TYPE',
            'HDU_CLASS',
            'FILE_DIR',
            'FILE_NAME',
            'HDU_NAME'],
    )
    index_header = DEFAULT_HEADER.copy()
    index_header['HDUCLAS1'] = 'INDEX'
    index_header['HDUCLAS2'] = 'HDU'
    index_header['TELESCOP'] = 'LST1'
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(index_table, header=index_header, name="HDU_INDEX"),
    ]
    fits.HDUList(hdus).writeto(Path(output).parent/'hdu-index.fits.gz', overwrite=True)


if __name__ == '__main__':
    main()
