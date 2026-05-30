import warnings

import numpy as np
import xarray as xr
from pathlib import Path

#from neurokinematics.pose.io import save_movement_dataset
from neurokinematics.io import save_dataset
from neurokinematics.pose.utils import pixels_to_cm


def pad_movements(movement_list: list, pad_value = np.nan):
    """Pad movement arrays to be the same size.

    Args:
        movement_list (list): List of dictionaries containing movement data. Dictionary should contain 'node_array', which is a IxNxM matrix, with I being samples, N being nodes, and M being coordinate (x,y)
        pad_value (np.nan, optional): Value to pad array by. Defaults to np.nan.

    Returns:
        padded (ndarray): Array containing padded data
        mov_list (list): List containing metadata for each 'event'
        valid (ndarray): Array containing boolean mask for samples, used to remove padded values during analysis
        lengths (ndarray): Array containing length in samples of each event 
    """

    lengths = np.array([mv['node_array'].shape[0] for mv in movement_list])
    max_len = lengths.max()

    n_events = len(movement_list)
    n_nodes = movement_list[0]['node_array'].shape[1]
    n_coords = movement_list[0]['node_array'].shape[2]

    padded = np.full(
        shape = (n_events, max_len, n_nodes, n_coords),
        fill_value = pad_value,
        dtype = float
    )

    valid = np.zeros((n_events, max_len), dtype=bool)
    mov_list = [None] * n_events

    for mov_id, movement in enumerate(movement_list):
        mov_len = movement['node_array'].shape[0]
        padded[mov_id, :mov_len, :, :] = movement['node_array']
        valid[mov_id, :mov_len] = True

        # ensuring order of metadata  - this is quite redundant
        mov_list[mov_id] = {
            'node_list': movement['node_list'],
            'start': movement['start'],
            'end': movement['end'],
            'reference_node': movement['reference_node'],
            'no_nodes': movement['no_nodes'],
            'trial': movement['trial'],
            'date': movement['date'],
            'id': movement['id'],
            'type': movement['type'],
            'frame_rate': movement['frame_rate']
        }
    
    return padded, mov_list, valid, lengths


def resample_padded_pose(movement_data: np.ndarray, valid: np.ndarray, fps: float, bin_edges: np.ndarray, method: str = 'mean'):
    """Resample a padded pose array. Used typically in conjunction with binning spikes

    Args:
        movement_data (np.ndarray): Array containing paddded pose data as Samples x Nodes x Coordinates. Originally designed for x,y position data
        valid (np.ndarray): Boolean array indicating indicies of valid pose values (True) and padded values (False)
        fps (float): Frame rate used for pose data in frames per second.
        bin_edges (np.ndarray): Bin edges used for binning associated data (e.g. spikes).
        method (str, optional): Method for resampling. Options are 'mean', 'median', 'first', or 'last'. Defaults to 'mean'.

    Raises:
        ValueError: Raises when non-existent method is used.

    Returns:
        pose_resampled (np.ndarray): Array containing resampled pose data, with padding.
        valid_bins (np.ndarray): Boolean array containing valid resampled points based on bins.
    """

    # get valid pose indicies and samples
    valid_pose = movement_data[valid]
    sample_times = np.arange(valid_pose.shape[0]) / fps

    n_bins = len(bin_edges) - 1
    n_nodes = movement_data.shape[1]    
    if len(movement_data.shape) == 2:
        n_coords = 1
    else:
        n_coords = movement_data.shape[2]

    attrs = {
        "fps": fps,
        "n_nodes": n_nodes,
        "n_coords": n_coords
    }

    # create resampling array
    pose_shape = movement_data.shape[1:]
    pose_resampled = np.full(
        (n_bins, *pose_shape),
        np.nan,
        dtype = float
    )
    
    # create validity array
    valid_bins = np.zeros(n_bins, dtype = bool)

    # loop through bins to resample
    for bin_idx in range(n_bins):
        # fine indicies of padded array inside current bin
        in_bin = (
            (sample_times >= bin_edges[bin_idx]) &
            (sample_times < bin_edges[bin_idx + 1])
        )

        # skip if there are no valid samples
        if not np.any(in_bin):
            continue

        # extract only valid pose samples in bin
        samples = valid_pose[in_bin]

        # choose method for resampling and store in resampled array
        if method == "mean":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category = RuntimeWarning)
                pose_resampled[bin_idx] = np.nanmean(samples, axis=0)
        elif method == "median":
            pose_resampled[bin_idx] = np.nanmedian(samples, axis=0)
        elif method == "first":
            pose_resampled[bin_idx] = samples[0]
        elif method == "last":
            pose_resampled[bin_idx] = samples[-1]
        else:
            raise ValueError(f"method must be one of: 'mean', 'median', 'first', 'last'. Method provided was: {method}")
        
        valid_bins[bin_idx] = True

    return pose_resampled, valid_bins

def build_movement_dataset(padded: np.ndarray, movement_list: list, valid: np.ndarray, lengths: np.ndarray, save_path: Path | str | None = None):
    """Creates xarray dataset and optionally saves to zarr store

    Args:
        padded (ndarray): Array containing padded pose data
        movement_list (list): List of movement metadata
        valid (ndarray): Array containing boolean mask for samples
        lengths (ndarray): _description_
        save_path (Path | str | None, optional): Path to save zarr store. Defaults to None.

    Returns:
        _type_: _description_
    """
    node_names = movement_list[0]['node_list']
    frame_rates = np.array([mv['frame_rate'] for mv in movement_list])

    if not np.allclose(frame_rates, frame_rates[0], rtol=0, atol=1e-9):
        raise ValueError(f"Inconsistent frame rates detected: {np.unique(frame_rates)}")

    frame_rate = frame_rates[0]
    time = np.arange(lengths.max()) / frame_rate
    
    ds = xr.Dataset(
       data_vars = {
           "position": (
               ['event', 'time', 'node', 'coord'],
               padded
           ),
           "valid": (
               ['event', 'time'],
               valid
           ),
            "start_idx": (
                ['event'],
                [mv['start'] for mv in movement_list]
            ),
            "end_idx": (
                ['event'],
                [mv['end'] for mv in movement_list]
            ),
            "reference_node": (
                ['event'],
                [mv['reference_node'] for mv in movement_list]
            ),
            "trial": (
                ['event'],
                [mv['trial'] for mv in movement_list]
            ),
            "type": (
                ['event'],
                [mv['type'] for mv in movement_list]
            ),
            "id": (
                ['event'],
                [mv['id'] for mv in movement_list]
            ),
            "date": (
                ['event'],
                [mv['date'] for mv in movement_list]
            ),
            "frame_rate": (
                ['event'],
                frame_rates
            )
       },
       coords = {
           "event": np.arange(len(movement_list)),
           "time": time,
           "node":  node_names,
           "coord": ["x", "y"]
       }
    )
    
    ds['velocity'] = compute_velocity(ds['position'], dim='time')
    ds['speed'] = compute_speed(ds['velocity'], dim='coord')
    ds['acceleration'] = compute_acceleration(ds['velocity'], dim='time')

    ds.attrs = {'features': ['position', 'velocity', 'speed', 'acceleration']}

    if save_path:
        save_path = Path(save_path) / 'movement_features.zarr'
        save_dataset(
            ds, 
            save_path, 
            chunks = {
                "event": min(100, len(movement_list)), 
                "time": -1, 
                "node": -1, 
                "coord": -1
            }
        )

    return ds


### * compute features * ###

def compute_velocity(position_ds, dim: str = "time"):
    velocity = position_ds.differentiate(dim)
    return velocity

def compute_speed(velocity, dim: str = "coord"):
    speed = np.sqrt((velocity ** 2).sum(dim))
    return speed

def compute_acceleration(velocity, dim: str = "time"):
    acceleration = velocity.differentiate(dim)
    return acceleration

def extract_metadata_from_trajectories(ds: xr.Dataset)
    id = ds.id.values[0]
    date = ds.date.values[0]
    date = str(np.datetime_as_string(date, unit='D'))
    return date, id

def extract_max_velocity_from_trajectories(ds: xr.Dataset, node: str):
    mask = (ds.reference_node == node).compute()
    ds_sub = ds.where(mask, drop=True)
    vy = ds_sub.velocity.sel(coord='y', node = node)
    vx = ds_sub.velocity.sel(coord='x' node = node)
    vy = np.nanmax(vy, axis=1) * pixels_to_cm()
    vx = np.nanmax(vx, axis=1) * pixels_to_cm()

    return vx, vy