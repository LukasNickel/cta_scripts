import numpy as np
import h5py
from cta_tools.reco.wobble import wobble_predictions_lst
from pathlib import Path
from astropy.table import QTable
from pyirf.cuts import evaluate_binned_cut
import operator
from aict_tools.io import read_data
import click
import astropy.units as u
from aict_tools.io import append_column_to_hdf5, remove_column_from_file
from tqdm import tqdm
from aict_tools.apply import create_mask_h5py



# from aict tools. find a different place for these
from operator import le, lt, eq, ne, ge, gt
OPERATORS = {
    '<': lt, 'lt': lt,
    '<=': le, 'le': le,
    '==': eq, 'eq': eq,
    '=': eq,
    '!=': ne, 'ne': ne,
    '>': gt, 'gt': gt,
    '>=': ge, 'ge': ge,
}

text2symbol = {
    'lt': '<',
    'le': '<=',
    'eq': '==',
    'ne': '!=',
    'gt': '>',
    'ge': '>=',
}

@click.command()
@click.argument('pattern')
@click.argument('log_file')
@click.option('--recalculate/--no-recalculate', default=False)
@click.option('--config', '-c', type=str)
def main(pattern, cut_file, log_file, recalculate, config):
    files = Path(pattern).parent.glob(pattern.split('/')[-1])
    with open(configuration_path) as f:
        config = yaml.load(f)


    for f in tqdm(files):
        print('Evaluating cuts on file', f)
        
        with h5py.File(f, 'r+') as file_:
            if 'cuts' in file_.keys():
                if not recalculate:
                    print('Already applied cuts. Skipping file ', f)
                    continue
                else:
                    print('Already applied cuts. Removing old results ', f)
                    del file_['cuts']
        df = read_data(f, 'events')

        if ('theta_on' not in df.columns) or (recalculate is True):
            theta_cols = [c for c in df.columns if c.startswith('theta')]
            ra_cols = [c for c in df.columns if c.startswith('ra')]
            dec_cols = [c for c in df.columns if c.startswith('dec')]
            for c in theta_cols+ra_cols+dec_cols:
                remove_column_from_file(f, 'events', c)
            # calculate theta stuff
            # rapred, decpred, onthteas, offthetas, rapnt, decpnt
            pred = wobble_predictions_lst(df)
            ra_pred, dec_pred = pred[0], pred[1]
            append_column_to_hdf5(f, ra_pred, 'events', 'ra_pred')
            append_column_to_hdf5(f, dec_pred, 'events', 'dec_pred')

            ra_pnt, dec_pnt = pred[4], pred[5]
            append_column_to_hdf5(f, ra_pnt, 'events', 'ra_pnt')
            append_column_to_hdf5(f, dec_pnt, 'events', 'dec_pnt')

            theta_on, theta_offs = pred[2], pred[3]
            assert len(theta_on) == len(df)
            append_column_to_hdf5(f, theta_on, 'events', 'theta_on')

            for i, theta_off in enumerate(theta_offs):
                append_column_to_hdf5(f, theta_off, 'events', f'theta_off{i}')
        else:
            theta_on = df['theta_on'].values
            theta_offs = [df[c] for c in df.columns if c.startswith('theta_off')]

        # actually evaluate cuts
        thetas = np.array(theta_offs + [theta_on])
        theta_mask = evaluate_binned_cut(
            u.Quantity(thetas.min(axis=0), u.deg, copy=False),
            u.Quantity(df['gamma_energy_prediction'].values, u.TeV, copy=False),
            theta_cuts,
            operator.le
        )

        assert len(theta_mask) == len(df)
        append_column_to_hdf5(f, theta_mask, 'cuts', 'passed_theta')

        gh_mask = evaluate_binned_cut(
            df['gammaness'],
            u.Quantity(df['gamma_energy_prediction'].values, u.TeV, copy=False),
            gh_cuts,
            operator.ge
        )
        assert len(gh_mask) == len(df)
        append_column_to_hdf5(f, gh_mask, 'cuts', 'passed_gh')

        if cut_feature:
            for feature, val, op in zip(cut_feature, value, operatorid):
                mask = OPERATORS[op](df[feature], val)
                append_column_to_hdf5(f, mask, 'cuts', f'passed_{feature}')

    with open(log_file, 'a+') as f:
        f.write('done')
