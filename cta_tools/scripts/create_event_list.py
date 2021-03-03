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
from aict_tools.io import read_data
import click
import astropy.units as u
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator

erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))


DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["CREATOR"] = "Lukas Nickel"
# fmt: off
DEFAULT_HEADER["HDUDOC"] = (
    "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
)
DEFAULT_HEADER["HDUVERS"] = "0.2"
DEFAULT_HEADER["HDUCLASS"] = "GAdata"


@click.command()
@click.argument('pattern')
@click.argument('cut_file')
@click.argument('output')
def main(pattern, cut_file, output):
    files = Path(pattern).parent.glob(pattern.split('/')[-1])
    gh_cuts = QTable.read(cut_file, hdu="GH_CUTS")
    theta_cuts = QTable.read(cut_file, hdu="THETA_CUTS")

    observations = []
    index = []
    ontimes = []
    livetimes = []
    timerefs = []
    starts = []
    stops = []

    # pretty random, might be too long
    gti_thresh = 10
    for i, f in enumerate(files):
        print(f)
        data = read_lst_dl2(f, drop_nans=False)
        ## convert to plain numpy array

        times = data['time']
        tstart = times[0]
        tstop = times[-1]
        starts.append(tstart)
        stops.append(tstop)
        ontime = (tstop - tstart).to_value(u.s)

        # this is wrong due to cuts applied!
        # need to add these to the dl1 file!
        # directly from lstchain pr 593
        # using method from PR#566
        deltaT = np.diff(times.mjd) # this is much faster using only the values
        deltaT = deltaT[(deltaT > 0) & (deltaT < 0.002)]
        rate = 1 / np.mean(deltaT)
        dead_time = np.amin(deltaT)
        dead_corr = 1 / (1 + rate * dead_time)
        livetime = ontime * dead_corr
        ontimes.append(ontime)
        livetimes.append(livetime)

        nevents = len(data)
        passed_gh = evaluate_binned_cut(data['gh_score'], data['reco_energy'], gh_cuts, operator.ge)
        data = data[passed_gh]
        ngh = len(data)
        print("\nEvents in run: ", nevents)
        print('Events nach g/h cut: ', ngh, ngh/nevents*100, "%")
        theta_on, off_thetas = calc_wobble_thetas(data)
        passed_theta = evaluate_binned_cut(theta_on, data['reco_energy'], gh_cuts, operator.le)
        for theta in off_thetas:
            passed_theta |= evaluate_binned_cut(theta, data['reco_energy'], gh_cuts, operator.le)

        data = data[passed_theta]
        ntheta = len(data)
        print("Events nach theta cut: ", ntheta, ntheta/nevents*100, "%")

        # 
        data = remove_nans(data)
        print('Events after nan dropping:', len(data))
 

        # this is not super efficient 
        pointing_icrs = get_icrs_pointings(data)
        predictions_icrs = get_icrs_prediction(data)
        data = rename_columns(data, 'ogadf')
        data['RA'] = predictions_icrs.ra
        data['DEC'] = predictions_icrs.dec
        data['RA_PNT'] = pointing_icrs.ra
        data['DEC_PNT'] = pointing_icrs.dec
        
        

        selected_times = data['TIME']
        mjd_offset = selected_times[0]
        offset_times = (selected_times - mjd_offset).to_value(u.s)
        int_mjd_offset = np.floor(mjd_offset.mjd)
        float_mjd_offset = (mjd_offset.value - int_mjd_offset)
        
        timerefs.append(mjd_offset)


        events = data[['RA', 'DEC', 'ENERGY']]
        # can gammapy work with this or do we need to convert?
        events['TIME'] = offset_times
        event_header = DEFAULT_HEADER.copy()
        event_header['HDUCLAS1'] = 'EVENTS'
        event_header['OBS_ID'] = data['obs_id'][0]
        event_header['TSTART'] = offset_times[0]
        event_header['TSTOP'] = offset_times[-1]

        event_header['MJDREFI'] = int_mjd_offset
        event_header['MJDREFF'] = float_mjd_offset
        event_header['TIMEUNIT'] = 's'
        event_header['TIMESYS'] = 'tai'
        event_header['TIMEREF'] = 'TOPOCENTER'
        event_header['ONTIME'] = ontime
        event_header['DEADC'] = dead_corr #1/(1+2.6e-5*2800)# taken from the lstchain pr (livetime/ontime)
        event_header['LIVETIME'] = event_header["DEADC"]*event_header["ONTIME"]
        # assuming constant pointing
        event_header['RA_PNT'] = pointing_icrs[0].ra.deg
        event_header['DEC_PNT'] = pointing_icrs[0].dec.deg
        event_header['EQUINOX'] = '2000.0'
        event_header['RADECSYS'] = 'FK5'  # ???
        event_header['ORIGIN'] = 'CTA'
        event_header['TELESCOP'] = 'LST1'
        event_header['INSTRUME'] = 'LST1'

        #  everythin is a gti for now
        gtis = QTable()
        gtis['START'] = [offset_times[0]]
        gtis['STOP'] = [offset_times[-1]]
        gti_header = DEFAULT_HEADER.copy()
        gti_header['MJDREFI'] = int_mjd_offset
        gti_header['MJDREFF'] = float_mjd_offset
        gti_header['TIMEUNIT'] = 's'
        gti_header['TIMESYS'] = 'tai'
        gti_header['TIMEREF'] = 'TOPOCENTER'

        pointings = data[['RA_PNT', 'DEC_PNT']]
        pointings['TIME'] = offset_times
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
            Path(output).parent/f'{data["obs_id"][0]}.fits.gz',
            overwrite=True
        )
        observations.append((
            data['obs_id'][0],
            pointings['RA_PNT'][0].to_value(u.deg),
            pointings['DEC_PNT'][0].to_value(u.deg),
            tstart,
            tstop,
            livetime/ontime
        ))

        index.append((
            data['obs_id'][0],
            'events',
            'events',
            '.',
            f'{data["obs_id"][0]}.fits.gz',
            'EVENTS'
        ))
        index.append((
            data['obs_id'][0],
            'gti',
            'gti',
            '.',
            f'{data["obs_id"][0]}.fits.gz',
            'GTI'
        ))
        index.append((
            data['obs_id'][0],
            'aeff',
            'aeff_2d',
            '.',
            'irfs.fits.gz',
            'EFFECTIVE_AREA'
        ))
        index.append((
            data['obs_id'][0],
            'psf',
            'psf_table',
            '.',
            'irfs.fits.gz',
            'PSF'
        ))
        index.append((
            data['obs_id'][0],
            'edisp',
            'edisp_2d',
            '.',
            'irfs.fits.gz',
            'ENERGY_DISPERSION'
        ))
        index.append((
            data['obs_id'][0],
            'bkg',
            'bkg_2d',
            '.',
            f'{data["obs_id"][0]}.fits.gz',
            'BACKGROUND',
        ))

    # build index file
    observation_table = QTable(
        rows=observations,
        names=['OBS_ID', 'RA_PNT', 'DEC_PNT', 'TSTART', 'TSTOP', 'DEADC'],
        #units=['', 'deg', 'deg', 's', 's', '']
    )
    #from IPython import embed; embed()
    observation_table['TSTART'] -= timerefs[0]
    observation_table['TSTART'] = observation_table['TSTART'].to(u.s)
    observation_table['TSTOP'] -= timerefs[0]
    observation_table['TSTOP'] = observation_table['TSTOP'].to(u.s)
    obs_header = DEFAULT_HEADER.copy()
    obs_header['HDUCLAS1'] = 'INDEX'
    obs_header['HDUCLAS2'] = 'OBS'
    obs_header['TELESCOP'] = 'LST1'
    obs_header['MJDREFI'] = np.floor(timerefs[0].value)#[0] #int(data['TIME'][0].mjd)
    obs_header['MJDREFF'] = timerefs[0].value - obs_header['MJDREFI']
    obs_header['TIMEUNIT'] = 's'
    obs_header['TIMESYS'] = 'tai'
    obs_header['TIMEREF'] = 'TOPOCENTER'
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
