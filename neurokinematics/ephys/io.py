"""I/O for saving/loading specific ephys data.

read/load
- openephys binary
- phy
- spikeinterface sorting analyzer
- openephys continuous object
- openephys xml settings
- sorting configs

save/initialize
- binary files
- zarr store

"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import spikeinterface.extractors as se
from spikeinterface.core import load_sorting_analyzer
import yaml
from spikeinterface.core import write_binary_recording
from open_ephys.analysis import Session

from neurokinematics.io import load_config

CHANNEL_MAP_PATH = Path(__file__).resolve().parent / 'channel_maps'
SORTING_CFG_PATH = Path(__file__).resolve().parent.parent.parent / 'configs' / 'spike_cfg'
PROBE_INTERFACE_PATH = Path(__file__).resolve().parent / 'probe_interfaces'

RECORDING_READERS = {
    "openephys": se.read_openephys,
    "plexon": se.read_plexon,
    "spikegadgets": se.read_spikegadgets,
    "spikeglx": se.read_spikeglx,
    "neuralynx": se.read_neuralynx
}

###
### READ/LOAD
###

def read_data(data_path: str, rec_type: str = 'openephys', stream_name: str = 'Record Node 109#Acquisition_Board-100.acquisition_board-B'):
    """Read data from ephys recordings for use with spikeinterface - currently only tested with open ephys formats.

    Args:
        data_path (str): Path to data.
        rec_type (str, optional): Ephys recording type - currently only works with openephys. Defaults to 'openephys'.
        stream_name (str, optional): Stream name for recording, necessary for openephys. Defaults to 'Record Node 109#Acquisition_Board-100.acquisition_board-B'.

    Returns:
        recording: Spikeinterface recording extractor.
    """

    rec_type = rec_type.lower() # incase rec_type is upper case

    # check rec_type in RECORDING_READERS dict
    if rec_type not in RECORDING_READERS:
        valid = ", ".join(RECORDING_READERS)
        raise ValueError(f"Unsupported recording type '{rec_type}'. Valid options are: {valid}")
    
    # create reader
    reader = RECORDING_READERS[rec_type]
    
    # read recording
    recording = reader(folder_path=data_path, stream_name=stream_name)
    #if rec_type == 'openephys':
    #    recording = read_openephys(folder_path=data_path, stream_name=stream_name)
    return recording


def load_phy_sorting(directory: str):
    """Read phy as spikeinterface sorting extractor.

    Args:
        directory (str): Path to phy output location.

    Returns:
        sorting (SortingExtractor): Spikeinterface sorting extractor
    """

    sorting = se.read_phy(directory)
    return sorting


def load_analyzer(directory: str):
    """Load spikeinterface sorting analyzer.

    Args:
        directory (str): Path to sorting analyzer

    Returns:
        analyzer (SortingAnalyzer): Spikeinterface sorting analyzer
    """
    
    analyzer = load_sorting_analyzer(directory)
    return analyzer


def load_settings_xml(record_node_path: str):
    """Loads settings from open ephys stored xml settings file.

    Args:
        record_node_path (str): Path to directory containing 'settings.xml'

    Returns:
        root (root element): Root element from xml tree
    """
    
    tree = ET.parse(record_node_path + '/settings.xml')
    root = tree.getroot()
    return root


def get_continuous(data_path: str, node_idx=0, rec_idx=0):
    """Get continuous data from Open Ephys recording

    Args:
        data_path (str): Root folder for data recording
        node_idx (int, optional): Index of node to extract continuous data from. Defaults to 0.
        rec_idx (int, optional): Index of recording to extract continuous data from. Defaults to 0.

    Returns:
        continuous: continuous data for relevant ephys recording stream
        recording_dir (str): Directory of recording 
    """
    
    session = Session(data_path) # create session
    recording_ = session.recordnodes[node_idx].recordings[rec_idx] # get specific recording
    
    recording_dir = recording_.directory # extract recording directory
    continuous = recording_.continuous[0] # extract continuous data
    
    return continuous, recording_dir


def get_sorting_cfg(cfg: str):
    """Wrapper of neurokinematics.io.load_config to simplify loading spike sorting configs.

    Args:
        cfg (str): Name of config file for spike sorting ending in `.yaml`

    Returns:
        sorting_cfg (dict): Dictionary containing information to use during spike sorting
    """

    sorting_cfg = load_config(str(SORTING_CFG_PATH / cfg))
    return sorting_cfg

###
### SAVE
###

def write_binary(recording, file_loc: str):
    """Write spikeinterface recording object to a binary file.

    Args:
        recording (RecordingExtractor): Spikeinterface recording extractor created during spike sorting / loaded from read_data
        file_loc (str): File path for save
    """

    write_binary_recording(recording, file_loc)
    

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
    
    zarr_path = Path(zarr_path) # create path for zarr store
    
    import dask.array as da # import dask for data
    
    data = da.zeros(
        (n_samples, n_channels),
        chunks = (chunk_len, n_channels),
        dtype = dtype
    )

    #  setup dataset
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