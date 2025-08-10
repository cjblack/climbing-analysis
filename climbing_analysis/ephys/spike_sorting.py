from spikeinterface.sorters import run_sorter
from spikeinterface import create_sorting_analyzer, load_sorting_analyzer, generate_ground_truth_recording, extract_waveforms
from spikeinterface.exporters import export_to_phy

from climbing_analysis.ephys.utils import *
from pathlib import Path


def sort_spikes(data_path: str, rec_type:str = 'openephys', sorter='kilosort4', probe_manufacturer: str = 'cambridgeneurotech', probe_id: str = 'ASSY-236-H5', channel_map = 'H5_openephys_channel_map.npy'):
    data_path = Path(data_path) # windows path
    output_folder = data_path / sorter # set output folder for kilosort
    recording_path = data_path / 'kilosort4/recording.dat'
    phy_folder = data_path / 'kilosort4/phy_output'

    recording = read_data(data_path=Path(data_path), rec_type=rec_type)
    probe = create_probe(probe_manufacturer, probe_id, channel_map) # creates probe from manufacturer, id, and channel map
    recording = recording.set_probe(probe, group_mode='by_shank') # sets probe

    # Run spikesorting
    sorting = run_sorter(sorter_name=sorter, recording=recording, folder=output_folder)

    # save recording as binary format to kilosort4 folder
    recording.save_to_folder(data_path=recording_path) # might not need to run this step...**
    analyzer = sorting_analyzer(sorting, recording, data_path) # create sorting analyzer
    export_to_phy(analyzer, output_folder=phy_folder) # export to phy for visualization
    return sorting, recording, probe

def sorting_analyzer(sorting, recording, data_path):
    folder = data_path / 'analyzer_folder'
    analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format='binary_folder',return_in_uV=True,folder=folder)
    analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
    _ = analyzer.compute('spike_amplitudes')
    _ = analyzer.compute('principal_components', n_components=5, mode="by_channel_local")
    return analyzer
