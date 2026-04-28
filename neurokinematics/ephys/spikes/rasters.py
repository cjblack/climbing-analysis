
from pathlib import Path

import pandas as pd
from neurokinematics.io import save_dataframe

def get_movement_aligned_rasters(alignment, sorter, storage_format: str = 'pickle'):
    movement_events = alignment['movement_event'].unique()
    nodes = alignment['node'].unique()
    unit_ids = sorter.unit_ids

    # make dir
    data_dir = Path(sorter.get_annotation('phy_folder')).parent.parent / 'rasters'
    data_dir.mkdir(exist_ok=True)
    
    pre_event = 0.5
    post_event = 0.5
    aligned_spike_times = []
    for uid in unit_ids:
        spike_times = sorter.get_unit_spike_train_in_seconds(unit_id=uid)

        for nd in nodes:
            for me in movement_events:

                aligned_movements = alignment.query("node==@nd & movement_event==@me")
                for idx, row in aligned_movements.iterrows():
                    trial = row['trial']
                    event_time = row['event_times_ts']
                    spikes_in_window = spike_times[(spike_times>(event_time-pre_event)) & (spike_times <= (event_time+post_event))]
                    spike_raster = spikes_in_window - event_time
                    aligned_spike_times.append({
                        "unit_id": uid,
                        "movement_event": me,
                        "movement_index": idx,
                        "node": nd,
                        "trial": trial,
                        "event_time_ts": event_time,
                        "event_time_sample": row['event_times_samples'],
                        "spike_raster": np.array(spike_raster)
                    })

    raster_df = pd.DataFrame.from_records(aligned_spike_times)

    if storage_format == "pickle":
        save_dataframe(raster_df, data_dir / 'movement_aligned_rasters.pkl', storage_format='pickle')
    elif storage_format == 'parquet':
        # not fully tested - spike rasters are ragged, np.arrays, parquet might not like this.
        save_dataframe(
            raster_df,
            data_dir / 'movement_aligned_rasters',
            partition_cols=['trial']
            )
    
    return raster_df