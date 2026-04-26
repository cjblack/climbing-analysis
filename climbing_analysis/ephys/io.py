import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import spikeinterface.extractors as se
import yaml
from spikeinterface.core import write_binary_recording
from open_ephys.analysis import Session

from climbing_analysis.io import load_config

CHANNEL_MAP_PATH = Path(__file__).resolve().parent / 'channel_maps'
SORTING_CFG_PATH = Path(__file__).resolve().parent.parent.parent / 'configs' / 'spike_cfg'
PROBE_INTERFACE_PATH = Path(__file__).resolve().parent / 'probe_interfaces'

def read_data(data_path: str, rec_type: str = 'openephys', stream_name: str = 'Record Node 109#Acquisition_Board-100.acquisition_board-B'):
    """
    Read data in from recording folder
    """
    if rec_type == 'openephys':
        recording = se.read_openephys(folder_path=data_path, stream_name=stream_name)
    return recording


def get_continuous(data_path: str, node_idx=0, rec_idx=0):
    """Get continuous data from Open Ephys recording

    Args:
        data_path (str): _description_
        node_idx (int, optional): _description_. Defaults to 0.
        rec_idx (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    session = Session(data_path)
    recording_ = session.recordnodes[node_idx].recordings[rec_idx]
    recording_dir = recording_.directory
    continuous = session.recordnodes[node_idx].recordings[rec_idx].continuous[0]
    
    return continuous, recording_dir

def get_settings_xml(record_node_path: str):
    tree = ET.parse(record_node_path + '/settings.xml')
    root = tree.getroot()
    return root

def write_binary(recording,file_loc: str):
    """
    Save binary file
    """
    write_binary_recording(recording, file_loc)

def get_sorting_params(cfg: str):
    """
    Get parameters for spike sorting
    """
    sorting_cfg = load_config(str(SORTING_CFG_PATH / cfg))

    return sorting_cfg

def initialize_zarr_store(
        zarr_path: Path,
        n_samples: int,
        n_channels: int,
        fs: float,
        chunk_len: int,
        dtype: str,
        attrs: dict
):
    """Initialize zarr store for lfp data

    Args:
        zarr_path (Path): file path of zarr store
        n_samples (int): number of samples in recording
        n_channels (int): number of channels in recording
        fs (float): sampling rate in hz
        chunk_len (int): length of chunks
        dtype (str): data type
        attrs (dict): metadata dict
    """
    
    zarr_path = Path(zarr_path)
    
    import dask.array as da
    
    data = da.zeros(
        (n_samples, n_channels),
        chunks = (chunk_len, n_channels),
        dtype = dtype
    )

    ds = xr.Dataset(
        data_vars = {
            "processed": (("time", "channel"), data)
        },
        coords={
            "time": np.arange(n_samples) / fs,
            "channel": np.arange(n_channels)
        },
        attrs=attrs
    )

    ds.to_zarr(zarr_path, mode="w", consolidated=False)