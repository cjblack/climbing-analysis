"""Interface for simplifying spike sorting.

Leverages SpikeInterfaces spike sorting functionality to simplify spike sorting and data storage with a pre-defined configuration file (configs/spk_sorting_cfg).

Currently tested with kilosort4 and Cambridge Neurotech silicon probes.

Upcoming tests for Neuropixels recordings.

"""

from pathlib import Path
from spikeinterface import create_sorting_analyzer
from spikeinterface.preprocessing import apply_preprocessing_pipeline, PreprocessingPipeline
from spikeinterface.exporters import export_to_phy
from spikeinterface.sorters import run_sorter
from spikeinterface.metrics.quality import compute_quality_metrics
from neurokinematics.io import save_dataframe
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

    Example:
        >>> sorter, recording, probe, analyzer = sort(
        ...     data_path = 'path/to/folder/containing/oebin',
        ...     cfg_file = 'spike_sorting_cfg.yaml'
        ...     )
    """
    # Load sorting params — accept either a config filename or an already-loaded dict
    sorting_cfg = cfg_file if isinstance(cfg_file, dict) else get_sorting_cfg(cfg_file)
    
    rec_type = sorting_cfg['rec_type']
    sorter = sorting_cfg['sorter']
    probe_id = sorting_cfg['probe_id']
    probe_manufacturer = sorting_cfg['probe_manufacturer']
    group_mode = sorting_cfg['group_mode']
    channel_map = sorting_cfg['channel_map']
    stream_name = sorting_cfg['stream_name']
    to_compute = sorting_cfg['to_compute']
    quality_metrics = sorting_cfg['quality_metrics']
    preprocessing_steps = sorting_cfg.get('preprocess', None)
    
    data_path = Path(data_path) # windows path
    save_path = Path(save_path) if save_path else Path(data_path)
    output_folder  = save_path / sorter            # spikeinterface creates this when running kilosort4
    phy_folder     = save_path / sorter / 'phy_output'
    # export_to_phy writes the recording binary here as 'recording.dat'
    recording_path = phy_folder / 'recording.dat'
    qual_metrics_path = save_path / 'spike_qc_metrics.csv'

    recording = read_data(data_path=Path(data_path), rec_type=rec_type, stream_name=stream_name)
    if probe_id.lower() != 'neuropixels':
        probe = create_probe(probe_manufacturer, probe_id, channel_map) # creates probe from manufacturer, id, and channel map
        recording = recording.set_probe(probe, group_mode=group_mode) # sets probe
    if preprocessing_steps:
        recording = preprocess_ephys(recording, preprocessing_steps)

    # Run spikesorting
    sorting = run_sorter(sorter_name=sorter, recording=recording, folder=output_folder)

    analyzer, metrics = sorting_analyzer(sorting, recording, data_path, compute_dict = to_compute, quality_metrics = quality_metrics, save_path = save_path) # create sorting analyzer
    save_dataframe(metrics, qual_metrics_path, storage_format='csv') # save quality metrics
    export_to_phy(analyzer, output_folder=phy_folder) # export to phy for visualization

    # label-only auto-curation from the quality metrics -> writes phy cluster_group.tsv
    # (so phy opens with suggestions) and a curated_units.csv. Non-destructive: the
    # human remains the final authority on unit identity in phy.
    from neurokinematics.ephys.spikes.curation import auto_curate
    curation_cfg = sorting_cfg.get('curation', {}) if isinstance(sorting_cfg, dict) else {}
    _, curated_units_path = auto_curate(
        metrics,
        rules=curation_cfg.get('rules'),
        phy_folder=phy_folder,
        save_path=save_path,
        fail_label=curation_cfg.get('fail_label', 'mua'),
    )

    file_outputs = {
        'spike_quality_metrics': {'path': str(qual_metrics_path), 'file_type': qual_metrics_path.suffix, 'attrs': {}},
        'curated_units': {'path': str(curated_units_path), 'file_type': '.csv', 'attrs': {}},
        'phy': {'path': str(phy_folder), 'file_type': 'phy output folder', 'attrs': {}},
        'recording': {'path': str(recording_path), 'file_type': recording_path.suffix, 'attrs': {}},
        'sorting_analyzer': {'path': str(save_path / 'sorting_analyzer'), 'file_type': 'sorting analyzer dir', 'attrs': {}},
        sorter: {'path': str(save_path / sorter), 'file_type': f'{sorter} dir', 'attrs': {}}
    }

    return sorting, recording, probe, analyzer, metrics, file_outputs

def sorting_analyzer(sorting, recording, data_path, compute_dict: dict, quality_metrics: list, save_path: Path | str | None = None):
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
    metrics = compute_quality_metrics(analyzer, metric_names = quality_metrics)
    return analyzer, metrics

def preprocess_ephys(recording, preprocessing_dict):

    preprocessed_recording = apply_preprocessing_pipeline(recording, preprocessing_dict)

    return preprocessed_recording