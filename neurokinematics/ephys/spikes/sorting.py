"""Interface for simplifying spike sorting.


"""

from pathlib import Path
from spikeinterface import create_sorting_analyzer
from spikeinterface.exporters import export_to_phy
from spikeinterface.sorters import run_sorter
from neurokinematics.ephys.io import *
from neurokinematics.ephys.utils import create_probe

def sort(data_path: str, cfg_file:str, save_path: Path | str | None = None): 
    """Sort spikes from data file - default is running kilosort4 on open ephys data recorded with H5 probe.
    Consequently, this has only been tested with the default parameters. More tests are required for other recording setups.


    Args:
        data_path (str): Directory path containing '.oebin' file.
        cfg_file (str): Config file name ending in '.yaml'. This config file must be stored in the projects root directory under 'configs/spike_cfg'.
        save_path (Path | str | None, optional): Specifies folder to store results. None will default to storing results in location of the recording folder in the data_path. Defaults to None.


    Returns:
        sorting: Spikeinterface sorting object.
        recording: Recording object, point to binary data.
        probe: Probe information from recording. This is based on the spike_cfg used
        analyzer: Spikeinterface analyzer object.

    Usage:
        sorting, recording, probe, analyzer = sort('path/to/datafolder', cfg_file='cfg_file.yaml')
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
    to_compute = sorting_cfg['to_compute']
    
    data_path = Path(data_path) # windows path
    if save_path:
        save_path = Path(save_path)
        output_folder = save_path / sorter # when spikeinterface runs kilosort4, this folder will be created
        recording_path = save_path / sorter / 'recording.dat'
        phy_folder = save_path / sorter / 'phy_output'
    else:
        save_path = Path(data_path)
        output_folder = save_path / sorter # set output folder for kilosort
        recording_path = save_path / sorter / 'recording.dat'
        phy_folder = save_path / sorter / 'phy_output'

    recording = read_data(data_path=Path(data_path), rec_type=rec_type, stream_name=stream_name)
    probe = create_probe(probe_manufacturer, probe_id, channel_map) # creates probe from manufacturer, id, and channel map
    recording = recording.set_probe(probe, group_mode=group_mode) # sets probe

    # Run spikesorting
    sorting = run_sorter(sorter_name=sorter, recording=recording, folder=output_folder)

    # save recording as binary format to kilosort4 folder
    recording.save_to_folder(data_path=recording_path) # might not need to run this step...**
    analyzer = sorting_analyzer(sorting, recording, data_path, compute_dict = to_compute, save_path = save_path) # create sorting analyzer
    export_to_phy(analyzer, output_folder=phy_folder) # export to phy for visualization
    return sorting, recording, probe, analyzer

def sorting_analyzer(sorting, recording, data_path, compute_dict: dict, save_path: Path | str | None = None):
    """
    Create sorting analyzer
    """
    if save_path is None:
        save_path = data_path / 'sorting_analyzer'
    else:
        save_path = save_path / 'sorting_analyzer'
    analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format='binary_folder',return_in_uV=True, folder = save_path)#folder=folder)
    analyzer.compute(compute_dict)#(['random_spikes', 'waveforms', 'templates', 'noise_levels', 'spike_locations'])
    _ = analyzer.compute('spike_amplitudes')
    _ = analyzer.compute('principal_components', n_components=5, mode="by_channel_local")
    return analyzer
