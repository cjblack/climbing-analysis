from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm

from neurokinematics.io import load_csv, load_zarr, save_dataframe, save_dataset

from neurokinematics.ephys.io import load_phy_sorting

from neurokinematics.pose.features import resample_padded_pose



def get_movement_aligned_features(alignment: str | pd.DataFrame, sorter: str, movement_dataset: str | xr.Dataset, save_path: Path | str | dict, bin_size: float = 0.02):
    """Bins spikes based on node specific movement periods

    Args:
        alignment (str | pd.DataFrame): Path or dataframe to pre-computed 'video_alignment.csv' file.
        sorter (str): Path to sorting folder, generally speaking this should be 'phy_output' generated during spike sorting.
        movement_dataset (str | xr.Dataset): Path or dataset containing movement specific data.
        save_path (Path | str | dict): Path to save data to. If dict is given, must be in format {'pose': 'path/to/save/pose', 'spikes': 'path/to/save/spikes'}
        bin_size (float, optional): Bin size in seconds to bin spikes into. Defaults to 0.02.

    Raises:
        ValueError: Checks frame rates are the same across 'events'. Should pass if testing on a single session

    Returns:
        spike_ds (xr.Dataset): xarray dataset containing binned spikes
        pose_ds (xr.Dataset): xarray dataset containing resampled movement data
        unbinned_spikes_df (df.DataFrame): Pandas dataframe containing unbinned spike times

    Example:
        >>> spike_ds, pose_ds, unbinned_spikes_df = get_movement_aligned_features(
        ...     alignment = 'path/to/video_alignment.csv',
        ...     sorter = 'path/to/ephys/sorter',
        ...     movement_dataset = 'path/to/movement_features.zarr',
        ...     save_path = {
        ...            'pose': 'path/to/pose/outputs',
        ...            'spikes': 'path/to/spikes/outputs',
        ...     },
        ...     bin_size = 0.05
        ... )
    """

    # load sorter if sorter is directory
    if isinstance(sorter, (str, Path)):
        sorter = load_phy_sorting(sorter)

    # load movement dataset if it's a file path
    if isinstance(movement_dataset, (str, Path)):
        movement_dataset = load_zarr(movement_dataset, method='xarray')
    
    # load alignment if it's a file path
    if isinstance(alignment, (str, Path)):
        alignment = load_csv(alignment, method='pandas')

    # set up save paths...
    if isinstance(save_path, (str, Path)):
        spike_save_path = Path(save_path)
        pose_save_path = Path(save_path)

    elif isinstance(save_path, dict):
        spike_save_path = Path(save_path['spikes'])
        pose_save_path = Path(save_path['pose'])

    
    fs = sorter.sampling_frequency
    if not np.allclose(movement_dataset.frame_rate.values, movement_dataset.frame_rate.values[0]):
        raise ValueError(f"Inconsistent frame rates detected: {np.unique(movement_dataset.frame_rate.values)}")

    fps = movement_dataset.frame_rate.values[0] # all values should be the same
    trial_ids = np.unique(movement_dataset.trial.values)
    mov_len = movement_dataset.time.shape[0]
    no_events = movement_dataset.event.shape[0]
    no_nodes = movement_dataset.node.shape[0]
    no_coords = movement_dataset.coord.shape[0]
    pose_features = movement_dataset.attrs['features']
    no_features = len(pose_features)
    unit_ids = sorter.unit_ids
    

    # attributes dict
    attrs = {
        "bin_size": bin_size,
        "pose_fps": fps,
        "ephys_fs": fs,
        "trial_ids": trial_ids,
        "no_events": no_events,
        "unit_ids": unit_ids,
        "pose_features": pose_features
    }


    # create bins based on movement data size
    duration_s = mov_len / fps
    n_bins = int(np.ceil(duration_s / bin_size))
    bin_edges = np.linspace(0, (n_bins * bin_size), n_bins + 1)
    bin_centers = bin_edges[:-1] + bin_size / 2

    # create arrays to fill
    spike_counts = np.zeros((no_events, len(bin_centers), len(unit_ids)))
    pose_resampled = dict()
    for feat in pose_features:
        # directional features carry a 'coord' dim (x/y); scalar-per-node features
        # (e.g. 'speed', 'confidence') do not.
        if 'coord' in movement_dataset[feat].dims:
            pose_resampled[feat] = np.full((no_events, len(bin_centers), no_nodes, no_coords), fill_value = np.nan)
        else:
            pose_resampled[feat] = np.full((no_events, len(bin_centers), no_nodes), fill_value = np.nan)

    #pose_resampled = np.full((no_events, len(bin_centers), no_nodes, no_coords, no_features), fill_value = np.nan)
    valid_bins = np.zeros((no_events, len(bin_centers)), dtype=bool)
    pre_movement_bins = np.zeros((no_events, len(bin_centers)), dtype=bool)
    unbinned_spikes = []

    for i, event_id in tqdm(enumerate(movement_dataset.event.values), total=no_events, desc="Extracting spikes", unit="events"):
        movement_sub = movement_dataset.isel(event=event_id)
        trial = movement_sub.trial.values
        start_id = movement_sub.start_idx.values
        end_id = movement_sub.end_idx.values
        valid_samples = movement_sub.valid.values
        node = movement_sub.reference_node.values

        for fi, feat in enumerate(pose_features):

            pose_resampled_i, valid_bins_i = resample_padded_pose(movement_sub[feat].values, movement_sub.valid.values, fps, bin_edges, method='mean')
            pose_resampled[feat][i] = pose_resampled_i
            #pose_resampled[i,:,:,:, fi] = pose_resampled_i
            if feat == pose_features[0]:
                valid_bins[i,:] = valid_bins_i

        # label bins falling before the detected movement onset as pre-movement
        n_pre = int(movement_sub.n_pre.values) if 'n_pre' in movement_dataset else 0
        onset_time_s = n_pre / fps
        pre_movement_bins[i, :] = (bin_centers < onset_time_s) & valid_bins[i, :]

        start = alignment.query('video_index==@trial & frame_id == @start_id')['sample_index'].item()
        end = alignment.query('video_index==@trial & frame_id == @end_id')['sample_index'].item()

        for unit_idx, uid in enumerate(unit_ids):
            spike_times = sorter.get_unit_spike_train(unit_id=uid)

            spikes_during_movement_sample = spike_times[(spike_times > start) & (spike_times < end)]
            spikes_during_movement = (spikes_during_movement_sample - start) / fs

            no_spikes = len(spikes_during_movement)
            no_bins = np.sum(valid_samples)
            firing_rate = no_spikes / (no_bins * bin_size)

            counts, _ = np.histogram(spikes_during_movement, bins = bin_edges)
            spike_counts[event_id, :, unit_idx] = counts

            for spk in spikes_during_movement_sample:
                unbinned_spikes.append({
                    "unit_id": int(uid),
                    "event_id": int(event_id),
                    "trial": int(trial),
                    "pose_frame_start_id": int(start_id),
                    "pose_frame_end_id": int(end_id),
                    "ephys_sample_start_id": int(start),
                    "ephys_sample_end_id": int(end),
                    "ephys_fs": float(fs),
                    "pose_fps": float(fps),
                    "reference_node": str(node),
                    "absolute_spike_time_sample": int(spk),
                    "relative_spike_time_sample": int(spk - start),
                    "spike_time_ts": float((spk - start) / fs)
                })

    # create dataframe for unbinned spieks
    unbinned_spikes_df = pd.DataFrame(unbinned_spikes)

    
    # saving data
    save_dataframe(unbinned_spikes_df, file_path = spike_save_path / 'unbinned_movement_spikes.parquet', storage_format='parquet')
    spike_ds = build_aligned_spike_binned_dataset(spike_counts, valid_bins, movement_dataset, bin_centers, unit_ids, attrs, spike_save_path, pre_movement=pre_movement_bins)
    pose_ds = build_resampled_movements_dataset(pose_resampled, valid_bins, movement_dataset, bin_centers, attrs, pose_save_path, pre_movement=pre_movement_bins)

    return spike_ds, pose_ds, unbinned_spikes_df



def build_aligned_spike_binned_dataset(spike_counts: np.ndarray, valid: np.ndarray, movement_dataset: xr.Dataset, time_bins: np.ndarray, unit_ids: np.ndarray, attrs: dict, save_path: Path | str | None = None, pre_movement: np.ndarray | None = None):

    if pre_movement is None:
        pre_movement = np.zeros_like(valid, dtype=bool)

    ds = xr.Dataset(
        data_vars = {
            # binned spike counts
            "spike_counts":(
                ['event', 'time_bin', 'unit'],
                spike_counts
            ),
            # boolean indicated non-padded indices
            "valid": (
                ['event', 'time_bin'],
                valid
            ),
            # boolean: True for bins before the detected movement onset (pre-movement)
            "pre_movement": (
                ['event', 'time_bin'],
                pre_movement
            ),
            # subject id
            "id":(
                ['event'],
                movement_dataset.id.values
            ),
            # experiment date
            "date":(
                ['event'],
                movement_dataset.date.values
            ),
            # node that initialised the moving
            "reference_node":(
                ['event'],
                movement_dataset.reference_node.values
            ),
            # trial - video that movement came from...might change later
            "trial": (
                ['event'],
                movement_dataset.trial.values
            )
        },
        coords = {
            "event": movement_dataset.event.values, 
            "time_bin": time_bins,
            "unit": np.arange(len(unit_ids))
        },
        attrs = attrs
    )

    if save_path:
        bin_info = int(np.ceil(attrs['bin_size']*1000.))
        save_path = Path(save_path) / f'movement_spike_counts_{bin_info}ms.zarr'
        
        save_dataset(
            ds,
            save_path,
            chunks = {
                "event": min(100, movement_dataset.event.values.shape[0]),
                "time_bin": -1,
                "unit": -1
            }
        )

    return ds


def build_resampled_movements_dataset(movement_dict: dict, valid: np.ndarray, movement_dataset: xr.Dataset, time_bins: np.ndarray, attrs: dict, save_path: Path | str | None = None, pre_movement: np.ndarray | None = None):

    pose_features = attrs['pose_features']

    if pre_movement is None:
        pre_movement = np.zeros_like(valid, dtype=bool)

    ds = xr.Dataset(
        data_vars = {
            # boolean indicated non-padded indices
            "valid": (
                ['event', 'time_bin'],
                valid
            ),
            # boolean: True for bins before the detected movement onset (pre-movement)
            "pre_movement": (
                ['event', 'time_bin'],
                pre_movement
            ),
            # subject id
            "id":(
                ['event'],
                movement_dataset.id.values
            ),
            # date of experiment
            "date":(
                ['event'],
                movement_dataset.date.values
            ),
            # node that initialised the movement
            "reference_node":(
                ['event'],
                movement_dataset.reference_node.values
            ),
            # trial - video that movement came from
            "trial": (
                ['event'],
                movement_dataset.trial.values
            )
        },
        coords = {
            "event": movement_dataset.event.values,
            "time_bin": time_bins,
            "node": movement_dataset.node.values,
            "coord": movement_dataset.coord.values
        },
        attrs = attrs
    )

    for feat, farray in movement_dict.items():
        farray = np.asarray(farray)
        if farray.ndim == 4:
            # binned directional features - e.g. 'position', 'velocity', 'acceleration'
            ds[feat] = (('event', 'time_bin', 'node', 'coord'), farray)
        else:
            # binned scalar-per-node features - e.g. 'speed', 'confidence'
            ds[feat] = (('event', 'time_bin', 'node'), farray)

    if save_path:
        bin_info = int(np.ceil(attrs['bin_size']*1000.))
        save_path = Path(save_path) / f'resampled_movements_{bin_info}ms.zarr'
        
        save_dataset(
            ds,
            save_path,
            chunks = {
                "event": min(100, movement_dataset.event.values.shape[0]),
                "time_bin": -1,
                "coord": -1
            }
        )

    return ds