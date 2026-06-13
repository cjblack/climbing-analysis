""" 

DEPRECATED -- replaced primarily by ephys.io

"""


import os
from pathlib import Path
import urllib.error
import xml.etree.ElementTree as ET
import numpy as np
import spikeinterface.extractors as se
import yaml
from spikeinterface.core import write_binary_recording
import spikeinterface.extractors as se
from probeinterface import get_probe, read_probeinterface
from open_ephys.analysis import Session



def get_stream_name(data_path: str | Path, rec_type: str, rec_node: str, stream_type: str = 'AP'):
    """_summary_

    Args:
        data_path (str | Path): _description_
        rec_type (str): _description_
        rec_node (str): _description_
        stream_type (str, optional): _description_. Options: 'AP' for spikes, 'LFP' for LFP. Defaults to 'AP'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if rec_type == 'openephys':
        streams = se.get_neo_streams('openephysbinary', data_path)
        for i, sn in enumerate(streams[0]):
            # check if acquisition board or PXI neuropixel
            if (rec_node in sn and 'acquisition_board' in sn and 'ADC' not in sn) or (rec_node in sn and stream_type in i):
                stream_name = sn


    else:
        if rec_type in se.recording_extractor_full_dict.keys():
            streams = se.get_neo_streams(rec_type, data_path)
        else:
            raise ValueError(f'{rec_type} is not a valid. Please select one of the following options: f{se.recording_extractor_full_dict.keys()}')

    return stream_name

'''
CHECK DEPRECATED
'''

CHANNEL_MAP_PATH = Path(__file__).resolve().parent / 'channel_maps'
SORTING_PARAMS_PATH = Path(__file__).resolve().parent / 'sorting_params'
PROBE_INTERFACE_PATH = Path(__file__).resolve().parent / 'probe_interfaces'

def read_data(data_path: str, rec_type: str = 'openephys', stream_name: str = 'Record Node 109#Acquisition_Board-100.acquisition_board-B'):
    """
    Read data in from recording folder
    """
    if rec_type == 'openephys':
        recording = se.read_openephys(folder_path=data_path, stream_name=stream_name)
    return recording

def get_lfp(data_path: str, node_idx=0, rec_idx=0):
    """
    Extract LFP from recording session
    *note: this is hard coded to recording node index
    """
    session = Session(data_path)
    continuous = session.recordnodes[node_idx].recordings[rec_idx].continuous[0]
    
    return continuous

def get_settings_xml(record_node_path: str):
    tree = ET.parse(record_node_path + '/settings.xml')
    root = tree.getroot()
    return root

def create_probe(probe_manufacturer: str, probe_id: str, channel_map: str):
    """
    Create probe object for mapping channels during spike sorting
    """
    
    channel_map_ = np.load(CHANNEL_MAP_PATH / channel_map) # load channel map
    # first we'll try to read the probe locally
    try:
        probe = read_probeinterface(str((PROBE_INTERFACE_PATH / f'{probe_id}.json').resolve()))
        probe = probe.probes[0]
        probe.set_device_channel_indices(channel_map_)
        print(f'Loaded probe {probe_id}.')
        return probe
    except Exception as e:
        print(f'Failed to load local probe file: {probe_id}')
    
    try:
        probe = get_probe(probe_manufacturer, probe_id) # create probe from manufacturer and id
        #channel_map_ = channel_map_ - 1 # account for python 0 index
        probe.set_device_channel_indices(channel_map_) # set channel indices to channel map
        print(f'Loaded probe {probe_id} from online library.')
        return probe
    except Exception as e:
        raise RuntimeError(f'Failed to load probe from both local and online: {e}')

def write_binary(recording,file_loc: str):
    """
    Save binary file
    """
    write_binary_recording(recording, file_loc)

def get_sorting_params(params: str):
    """
    Get parameters for spike sorting
    """
    with open(str(SORTING_PARAMS_PATH / params), 'r') as f:
        sorting_params = yaml.load(f, Loader=yaml.SafeLoader)
    return sorting_params
