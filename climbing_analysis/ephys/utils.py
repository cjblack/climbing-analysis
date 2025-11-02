import spikeinterface.extractors as se
from spikeinterface.core import write_binary_recording
from probeinterface import get_probe
import numpy as np
from pathlib import Path

CHANNEL_MAP_PATH = Path(__file__).resolve().parent / 'channel_maps'

def read_data(data_path: str, rec_type: str = 'openephys'):
    if rec_type == 'openephys':
        recording = se.read_openephys(folder_path=data_path, stream_name='Record Node 109#Acquisition_Board-100.acquisition_board-B') # this is hardcoded to the initial rec node
    return recording
def get_lfp(data_path: str):
    session = Session(directory)
    continuous = session.recordnodes[node_idx].recordings[rec_idx].continuous[0]
def create_probe(probe_manufacturer: str, probe_id: str, channel_map: str):
    probe = get_probe(probe_manufacturer, probe_id) # create probe from manufacturer and id
    channel_map_ = np.load(CHANNEL_MAP_PATH / channel_map) # load channel map
    #channel_map_ = channel_map_ - 1 # account for python 0 index
    probe.set_device_channel_indices(channel_map_) # set channel indices to channel map
    return probe

def write_binary(recording,file_loc: str):
    write_binary_recording(recording, file_loc)

def get_sorting_params(params: str):
    with open(params, 'r') as f:
        sorting_params = yaml.load(f, Loader=yaml.SafeLoader)
    return sorting_params
