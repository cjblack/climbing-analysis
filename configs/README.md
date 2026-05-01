# Configuration files

`neurokinematics` uses YAML config files to define session-level and invidivual (pose, spike, lfp, multi-modal) metadata and processing settings.

The structure of these config files is in active development, but designed for modularity in mind.

## Config folder structure

```text
neurokinematics/
├── configs/            
│   ├── lfp_cfg/            # LFP config files
│   ├── multimodal_cfg/     # multimodal config files
│   ├── pose_cfg/           # pose config files
│   ├── session_cfg/        # session config files
│   └── spk_sorting_cfg/    # spike sorting config files            
│       └── ephys_data/ 
...
```

## Main session config

The main session config points to separate sub-config files for spike sorting, LFP preprocessing, pose processing, and alignment:

Example:
```yaml
session:
  output_root: "set_location_for_data"
  behaviour: "demo"
  ephys:
    acquisition: "openephys"
    lfp:
      node_idx: 0
      rec_idx: 0
    spikes:
      node_idx: 1
      rec_idx: 0

configs:
  spikes: "demo_spike_sorting_cfg.yaml"
  lfp: "demo_lfp_cfg.yaml"
  pose: "demo_pose_cfg.yaml"
  multi_modal: "demo_camera_alignment_cfg.yaml"

pipeline:
  run_pose: true
  run_spikes: true
  run_lfp: true
```

### Fields
- `session.output_root` &rarr; path for output folder to store session.
- `session.behaviour` &rarr; name of behaviour type of session
- `session.ephys.acquisition` &rarr; name of acquisition system used for ephys recording
- `session.ephys.lfp.node_idx` &rarr; node index of Open Ephys record node using to capture LFP
- `session.ephys.lfp.rec_idx` &rarr; recording index of Open Ephys used for LFP
- `session.ephys.spikes.node_idx` &rarr; node index of Open Ephys record node using to capture spikes
- `session.ephys.spikes.rec_idx` &rarr; recording index of Open Ephys used for spikes
- `configs.spikes` &rarr; name of config file to use in session for spike data
- `configs.lfp` &rarr; name of config file to use in session for LFP data
- `configs.pose` &rarr; name of config file to use in session for pose data
- `config.multi_modal` &rarr; name of config file to use in session for multi-modal data and alignment
- `pipeline.run_pose` &rarr; boolean to determine whether to automatically run pose on session creation (currently unused)
- `pipeline.run_spikes` &rarr; boolean to determine whether to automatically run spike sorting on session creation (currently unused)
- `pipeline.run_lfp` &rarr; boolean to determine whether to automatically run lfp processing on session creation (currently unused)

## Spike config

The spike config contains information relevant to running spike sorting and accessing spike data:

```yaml
name: "sample_spike_sorting_cfg"
rec_type: "openephys"
sorter: "kilosort4"
to_compute:
 random_spikes: {}
 waveforms:
  ms_before: 1.0
  ms_after: 2.0
 templates: {}
 noise_levels: {}
 spike_locations: {}
 quality_metrics: {}
probe_manufacturer: "cambridge_neurotech"
probe_id: "ASSY-236-H5"
group_mode: 'auto'
channel_map: "h5_open_ephys_acquisition_channel_map.npy"
stream_name: "Record Node 109#Acquisition_Board-100.acquisition_board-B"
sample_rate: 30000.

```
### Fields
- `name` &rarr; config name (superfluous)
- `rec_type` &rarr; acquisition system used for ephys (redundant in session config)
- `sorter` &rarr; spike sorter used
- `to_compute` &rarr; list of SpikeInterface arguments and parameters to compute on sorted units
- `probe_manufacturer` &rarr; probe manufacturer, used for identifying probe map
- `probe_id` &rarr; id of probe used in recording, used to set geometries
- `group_mode` &rarr; group mode for sorting across probes
- `channel_map` &rarr; channel map to use for spike sorting, user defined
- `stream_name` &rarr; stream name for spike data, used to run sorting
- `sample_rate` &rarr; sample rate of recording

## Pose config

The pose config has data relevant for extracting and preprocessing pose data:

```yaml
# Simple preprocessing config file, only fills missing vals
pose_format:
  pose_type: 'sleap'
  file_format: 'h5'
  frame_rate: 200.
pose_preprocessing:
  fill_missing: true
  confidence:
    enabled: false
    thresh: 0.7
  velocity:
    enabled: false
    thresh: 20
post_processing:
  storage_format: 'csv' # right now this isn't doing much as data is stored as csv by default
movement_detection:
  enabled: true
  sort_cols:
    - 'Trial'
    - 'Date'
    - 'frame_id'
  group_cols:
    - 'Trial'
    - 'Date'
  node_list:
    - 'r_forepaw'
    - 'l_forepaw'
    - 'r_hindpaw'
    - 'l_hindpaw'
```

### Fields
- `pose_format.pose_type` &rarr; software used for markerless pose estimation
- `pose_format.file_format` &rarr; file format of pose data
- `pose_format.frame_rate` &rarr; frame rate of camera used in original data collection
- `pose_preprocessing.fill_missing` &rarr; boolean to determine filling of NaN values in pose estimation
- `pose_preprocessing.confidence` &rarr; argument and parameters to filter data by confidence scores
- `pose_preprocessing.velocity` &rarr; argument and parameters to filter data by velocity
- `post_processing.storage_format` &rarr; desired format to store data at (currently only supporting csv)
- `movement_detection.enabled` &rarr; sets whether movement detection will be performed
- `movement_detection.sort_cols` &rarr; sorting columns for organising pose estimation data to sort by
- `movement_detection.group_cols` &rarr; grouping columns for organising pose estimation data by group
- `movement_detection.node_list` &rarr; list of tracked nodes for performing movement detection on (defined by user and pose estimation run)

## LFP config

The LFP config has details relevant to extracting Open Ephys data for preprocessing and storage steps:

```yaml
# set all the high-level informaiton for preprocessing lfp
name: "lfp_preprocessing"
dtype: float32
chunking:
  chunk_duration_s: 10.0
  pad_duration_s: 1.0
filters:
  notch: 50.0
  bandpass: (0.1, 100.0)
  quality: 30.0
downsample_rate: 1000.0
storage_format: "zarr"
```

### Fields
- `name` &rarr; name of config file (superfluous)
- `dtype` &rarr; data type for storing processed LFP
- `chunking.chunk_duration_s` &rarr; duration in seconds to chunk LFP data
- `chunking.pad_duration_s` &rarr; pad duration in seconds to add to LFP data during chunking
- `filters.notch` &rarr; frequency in Hz of notch filter setting
- `filters.badnpass` &rarr; tuple containing low and high pass settings for bandpass filter
- `filters.quality` &rarr; float value for quality setting on filter
- `downsample_rate` &rarr; frequency in Hz to downsample data to
- `storage_format` &rarr; format to use for storing data (unused and defaults to zarr regardless, this will be updated in the future)

## Multimodal config

The multimodal config has details regarding alignment between markerless pose and ephys data:

```yaml
# Sample config for camera alignment
name: "camera_alignment_cfg"
camera:
  manufacturer: "flir"
  model: "blackfly_s"
acquisition_settings:
  record_node: 1
  record_index: 0
  event_channel: 67
  sample_rate: 30000.
detection_settings:
  fps: 200.
  threshold_ratio: 0.3
  minimum_bout_duration: 0.1
  bandwidth: 40.
```

### Fields

- `name` &rarr; name of config (superfluous)
- `camera.manufacturer` &rarr; name of camera manufacturer for metadata
- `camera.model` &rarr; name of camera model used in recording video data
- `acquisition_settings.record_node` &rarr; record node index of Open Ephys channel where strobe was recorded
- `acquisition_settings.record_index` &rarr; index of recording in Open Ephys data where strobe was recorded
- `acquisition_settings.event_channel` &rarr; analog channel id where camera strobe output was recorded
- `detection_settings.fps` &rarr; camera frame rate, used in detecting frame captures in ephys (redundant)
- `detection_settings.threshold_ratio` &rarr; threshold ratio used in detecting frame capture in ephys
- `detection_settings.minimum_bout_duration` &rarr; minimum bout duration for identifying a purposeful video recording
- `detection_settings.bandwidth` &rarr; bandwidth in Hz for filter used to identify strobe on analog channel