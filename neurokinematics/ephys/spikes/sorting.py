"""Interface for simplifying spike sorting.


"""

from pathlib import Path
import matplotlib.pyplot as plt
from spikeinterface import create_sorting_analyzer
from spikeinterface.exporters import export_to_phy
from spikeinterface.extractors import read_phy
from spikeinterface.sorters import run_sorter
#from spikeinterface.core import load_sorting_analyzer
import spikeinterface.widgets as sw
from neurokinematics.ephys.io import *
from neurokinematics.ephys.utils import create_probe
#from climbing_analysis.pose.utils import pixels_to_cm # REMOVE
#from scipy.ndimage import gaussian_filter1d


def sort(data_path: str, cfg_file:str): 
    """Sort spikes from data file - default is running kilosort4 on open ephys data recorded with H5 probe.
    Consequently, this has only been tested with the default parameters. More tests are required for other recording setups.


    Args:
        data_path (str): Directory path containing '.oebin' file.
        cfg_file (str): Config file name ending in '.yaml'. This config file must be stored in the projects root directory under 'configs/spike_cfg'.

    Returns:
        sorting: Spikeinterface sorting object.
        recording: Recording object, point to binary data.
        probe: Probe information from recording. This is based on the spike_cfg used
        analyzer: Spikeinterface analyzer object.
    """
    # Load sorting params
    sorting_cfg = get_sorting_cfg(cfg_file)

    rec_type = sorting_cfg['rec_type']
    sorter = sorting_cfg['sorter']
    probe_id = sorting_cfg['probe_id']
    probe_manufacturer = sorting_cfg['probe_manufacturer']
    group_mode = sorting_cfg['group_mode']
    channel_map = sorting_cfg['channel_map']
    stream_name = sorting_cfg['stream_name']
    
    data_path = Path(data_path) # windows path
    output_folder = data_path / sorter # set output folder for kilosort
    recording_path = data_path / f'{sorter}/recording.dat'
    phy_folder = data_path / f'{sorter}/phy_output'

    recording = read_data(data_path=Path(data_path), rec_type=rec_type, stream_name=stream_name)
    probe = create_probe(probe_manufacturer, probe_id, channel_map) # creates probe from manufacturer, id, and channel map
    recording = recording.set_probe(probe, group_mode=group_mode) # sets probe

    # Run spikesorting
    sorting = run_sorter(sorter_name=sorter, recording=recording, folder=output_folder)

    # save recording as binary format to kilosort4 folder
    recording.save_to_folder(data_path=recording_path) # might not need to run this step...**
    analyzer = sorting_analyzer(sorting, recording, data_path) # create sorting analyzer
    export_to_phy(analyzer, output_folder=phy_folder) # export to phy for visualization
    return sorting, recording, probe, analyzer

def sorting_analyzer(sorting, recording, data_path):
    """
    Create sorting analyzer
    """
    folder = data_path / 'sorting_analyzer'
    analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format='binary_folder',return_in_uV=True,folder=folder)
    analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels', 'spike_locations'])
    _ = analyzer.compute('spike_amplitudes')
    _ = analyzer.compute('principal_components', n_components=5, mode="by_channel_local")
    return analyzer
