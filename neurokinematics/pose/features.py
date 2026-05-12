import numpy as np
import xarray as xr
from pathlib import Path

from neurokinematics.pose.io import save_movement_dataset

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


def build_movement_dataset(padded: ndarray, movement_list: list, valid: ndarray, lengths: ndarray, save_path: Path | str | None = None):
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
    ds = xr.Dataset(
       data_vars = {
           "position": (
               ['event', 'sample', 'node', 'coord'],
               padded
           ),
           "valid": (
               ['event', 'sample'],
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
                [mv['frame_rate'] for mv in movement_list]
            )
       },
       coords = {
           "event": np.arange(len(movement_list)),
           "sample": np.arange(lengths.max()),
           "node":  node_list,
           "coord": ["x", "y"]
       }
    )

    if save_path:
        save_path = Path(save_path) / 'movement_positions.zarr'
        save_movement_dataset(
            ds, 
            save_path, 
            chunks = {
                "event": min(100, len(movement_list)), 
                "sample": -1, 
                "node": -1, 
                "coord": -1
            }
        )

    return ds