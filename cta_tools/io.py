import tables
from aict_tools.cta_helpers import horizontal_to_camera_cta_simtel
import numpy as np

import pandas as pd


def load_energy(filename, tablename='gamma_energy_prediction'):
    energy = {
        'tel': {},
        'array': {'true': {}, 'pred': {}}
    }
    with tables.open_file(filename) as f:
        if 'simulation' in f.root:
            energy['array']['true'] = f.get_node('/simulation/event/subarray/shower').col(
                'true_energy')
            energy['true']['obs_id'] = f.get_node('/simulation/event/subarray/shower').col(
                'obs_id')
            energy['true']['event_id'] = f.get_node('/simulation/event/subarray/shower').col(
                'event_id')
        if 'dl2' in f.root:
            array_table = f.get_node(f'/dl2/event/subarray/{tablename}')
            energy['array']['pred']['mean'] = array_table.col('values_block_0')[:,0]
            energy['array']['pred']['std'] = array_table.col('values_bloc_0')[:,1]
            energy['array']['pred']['obs_id'] = array_table.col('values_block_1')[:,0]
            energy['array']['pred']['event_id'] = array_table.col('values_bloc_1')[:,1]
            
            #for tel in f.get_node('/dl2/event/telescope'):
            

            # add telescope predictions, including matching of ids
    return energy


def load_position_prediction(filename, tablename='disp_prediction'):
    pos = {
        'tel': {},
        'array': {'true': {}, 'pred': {}, 'ids': ()},
    }
    with tables.open_file(filename) as f:
        if 'simulation' in f.root:
            pos['array']['true'][value] = f.get_node('/simulation/event/subarray/shower').col(
                'true_energy')
        energy['array']['true'][value] = f.get_node('/simulation/event/subarray/shower').col(
            'true_energy'
        )
        for tel in f.get_node('/dl2/event/telescope'):
            tel_name = tel._v_name
            disp_table = f.get_node(f'{tel._v_pathname}/{tablename}')
            event_ids = disp_table.col('event_id')
            obs_ids = disp_table.col('obs_id')
            _ids = disp_table.col('event_id')
            event_ids = disp_table.col('event_id')



def read_pandas(path, tel_ids):
    file_table = tables.open_file(path)

    #  basically what I do in the aict cta dl1 PR
    layout = pd.read_hdf(path, '/configuration/instrument/subarray/layout')
    optics = pd.read_hdf(path, '/configuration/instrument/telescope/optics')
    layout = layout.merge(optics, how='outer', on="name")
    layout.set_index('tel_id', inplace=True)

    tels_to_load = [
        f"/dl1/event/telescope/parameters/{tel.name}"
        for tel in file_table.root.dl1.event.telescope.parameters
        if int(tel.name.split('_')[-1]) in tel_ids
    ]

    tel_dfs = []
    for tel in tels_to_load:
        print(tel)
        # as not all columns are located here, we cant just use
        # columns=columns
        tel_df = pd.read_hdf(path, tel)

        # Pointing information has to be loaded from the monitoring tables
        # Some magic has to be performed to merge the dfs,
        # because only the last pointing is stored with the stage-1 ctapipe tool
        # We also need the trigger tables as monitoring is based on time not events
        # ToDo: Verify this for real data! MC contains only one pointing

        tel_key = tel.split('/')[-1]
        tel_triggers = pd.read_hdf(
            path,
            "/dl1/event/telescope/trigger"
        )
        tel_df = tel_df.merge(
            tel_triggers,
            how='inner',
            on=['obs_id', 'event_id', 'tel_id']
        )
        tel_pointings = pd.read_hdf(
            path,
            f"/dl1/monitoring/telescope/pointing/{tel_key}"
        )
        tel_df = tel_df.merge(
            tel_pointings,
            how='left',
            on='telescopetrigger_time'
        )
        # for chunked reading there might not be a matching trigger time
        if tel_df['azimuth'].isnull().iloc[0]:
            # find the closest earlier pointing
            earliest_chunktime = tel_df['telescopetrigger_time'].min()
            time_diff = (
                tel_pointings['telescopetrigger_time']
                - earliest_chunktime
            )
            earlier = (time_diff <= 0)
            closest_pointing = tel_pointings.loc[time_diff[earlier].idxmax()]
            tel_df.at[0, 'azimuth'] = closest_pointing['azimuth']
            tel_df.at[0, 'altitude'] = closest_pointing['altitude']
        tel_df[['azimuth', 'altitude']] = tel_df[['azimuth', 'altitude']].fillna(
            method='ffill'
        )

        tel_number = int(tel.split('_')[-1])
        tel_df["equivalent_focal_length"] = layout.loc[tel_number][
            "equivalent_focal_length"
        ]

        #print()
        true_params = pd.read_hdf(path, tel.replace('/dl1', '/simulation'))
        true_params.columns = [f'true_{name}' for name in true_params.columns]
        true_params = true_params.rename(columns={'true_event_id': 'event_id'})
        #print(true_params.columns)
        tel_df = tel_df.merge(true_params, how='left', on='event_id')
        # combine the telescope dataframes
        tel_dfs.append(tel_df)

    df = pd.concat(tel_dfs)
    print(len(df))
    true_information = pd.read_hdf(
        path,
        "/simulation/event/subarray/shower"
    )

    df = df.merge(true_information, on=['obs_id', 'event_id'], how="left")



    file_table.close()

    df['source_x'], df['source_y'] = horizontal_to_camera_cta_simtel(
        df['true_alt'],
        df['true_az'],
        np.rad2deg(df['altitude']),
        np.rad2deg(df['azimuth']),
        df['equivalent_focal_length']
    )

    return df

