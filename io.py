import tables
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



