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

## Pose config

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

## LFP config

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

## Multimodal config
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