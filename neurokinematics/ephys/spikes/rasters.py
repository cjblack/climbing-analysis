"""Computing spike rasters from aligned events.

Interface to use alignment dataframes to extract spike rasters from spikeinterface SortingExtractor.
Currently implements extraction for movement aligned data with `get_movement_aligned_rasters`.
"""


from pathlib import Path

import numpy as np
import pandas as pd

from neurokinematics.io import save_dataframe
from neurokinematics.data.processed import SpikeRasterProcessed

def get_movement_aligned_rasters(alignment: pd.DataFrame, sorter, save_path: Path | str, storage_format: str = 'pickle'):
    """Computes and optionally save movement aligned spike rasters (e.g. spike times aligned to movement onset/ movement event windows).

    Args:
        alignment (pd.DataFrame): Dataframe created from events/movement_event_alignment.csv.
        sorter (SortingExtractor): Spikeinterface sorting extractor object containing spike times.
        save_path (Path | str): Directory to save aligned raster data to.
        storage_format (str, optional): Format to store raster dataframe as. Options are 'pickle', 'parquet'. Storing with 'parquet' may yield issues, if so, switch to 'pickle'. Defaults to 'pickle'.

    Returns:
        spike_raster_proc_obj (dataclass): Lightweight dataclass containing metadata for movement aligned rasters.

    Example:
        >>> spike_raster_proc_obj = get_movement_aligned_rasters(
        ...     alignment = alignment_df,
        ...     sorter = sorting,
        ...     save_path = "path/to/outputs"
        ... )
        >>> spike_raster_proc_obj.output_path
        PosixPath('path/to/outputs/movement_aligned_rasters.pkl')

    """
    # extract alignment information for creating data frame
    movement_events = alignment['movement_event'].unique() # get unique movement event types - this may differ across analyses
    nodes = alignment['node'].unique() # get unique nodes - this may differ across pose estimation models
    unit_ids = sorter.unit_ids # extract unit ids from spikeinterface SortingExtractor object
    
    # set alignment for spike rasters - hardcoded as +/- 0.5s
    pre_event = 0.5
    post_event = 0.5

    aligned_spike_times = [] # empty list to fill with aligned spike rasters

    # loop through units    
    for uid in unit_ids:

        spike_times = sorter.get_unit_spike_train_in_seconds(unit_id=uid) # extract time stamps in seconds

        # loop through nodes
        for nd in nodes:
            # loop through movement events
            for me in movement_events:

                aligned_movements = alignment.query("node==@nd & movement_event==@me") # query alignment for current node and movement event

                # iterate through queried data rows
                for idx, row in aligned_movements.iterrows():
                    trial = row['trial']
                    event_time = row['event_times_ts']

                    spikes_in_window = spike_times[(spike_times>(event_time-pre_event)) & (spike_times <= (event_time+post_event))] # collect spikes within event window
                    spike_raster = spikes_in_window - event_time # shift center to zero for convenience, but store event_time_ts to reconstruct original times
                    
                    # append data
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

    raster_df = pd.DataFrame.from_records(aligned_spike_times) # create dataframe from aligned spike times list


    # make dir
    data_dir = Path(save_path) / 'rasters' #Path(sorter.get_annotation('phy_folder')).parent.parent / 'rasters'
    data_dir.mkdir(parents=True, exist_ok=True)
    if storage_format == "pickle":
        output_path = data_dir / 'movement_aligned_rasters.pkl'
        save_dataframe(raster_df, output_path, storage_format='pickle')
        spike_raster_proc_obj = SpikeRasterProcessed(
            unit_ids = unit_ids,
            nodes = nodes,
            event_types = movement_events,
            output_path = output_path,
            storage_format=storage_format
        )

    elif storage_format == 'parquet':
        # not fully tested - spike rasters are ragged, np.arrays, parquet might not like this.
        output_path = data_dir / 'movement_aligned_rasters'
        save_dataframe(
            raster_df,
            output_path,
            partition_cols=['trial']
            )
        spike_raster_proc_obj = SpikeRasterProcessed(
            unit_ids = unit_ids,
            nodes = nodes,
            event_types = movement_events,
            output_path = output_path,
            storage_format=storage_format,
            partition_cols = ['trial'],
        )
    else:
        raise ValueError("storage_format must be 'pkl', 'parquet'.")

    
    return spike_raster_proc_obj