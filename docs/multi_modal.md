# multi_modal
The `multi_modal` module provides tools for temporally aligning electrophysiological recordings with behavioural data derived from markerless pose estimation.

The current implementation focuses on synchronisation and event alignment across acquisition systems, providing the foundation for downstream cross-modal analyses. Future development will expand this module to include multimodal feature extraction and analysis utilities.

## Module structure

- `alignment.py` &rarr; core utilities for synchronising pose and electrophysiology data

## Align video frames to ephys data using strobe method
If the camera's strobe ouput is recorded on an analog input channel of the ephys acquisition system, frame capture times can be identified directly from the recorded signal.

This workflow has currently been tested using a FLIR blackfly S camera (200fps) with the Open Ephys acquisition board.

```python
from neurokinematics.multi_modal.alignment import get_camera_events

event_data, ts, bouts, frame_captures, continuous = get_camera_events(
    data_path = "path/to/ephys", 
    camera_cfg_file = "demo_camera_alignment_cfg.yaml",
    save_path = "path/to/outputs"
    )
```
### What this does
- Identify frame captures times in ephys recording using the analog channel containing strobe data
- Stores ids of ephys samples with indexed frame captures

### Inputs
- Path to Open Ephys recording
- Camera alignment configuration file

### Outputs
- `video_alignment.csv` &rarr; frame capture times aligned to electrophysiology timestamps and samples

## Align movement events to ephys
```python
from neurokinematics.multi_modal.alignment import align_movements_to_ephys
movement_alignment_df = align_movements_to_ephys(
    dirs = {
        "events": "path/to/events", # event path contains "movement_events.pkl" - required
        "pose": "path/to/pose", # pose path contains 'pose_data.csv' - required
        "alignment": "path/to/alignment" # alignment path contains 'video_alignment.csv' - required
    },
    save_path = "path/to/outputs"
)
```
### What this does
- Align previously extracted and stored movement event times to ephys indicies
- Stores aligned movement events

### Inputs
- Extracted movement events (`movement_events.pkl`)
- Processed pose data (`pose_data.csv`)
- Video alignment data (`video_alignment.csv`)

### Outputs
- `movement_event_alignment.csv` &rarr; behavioural events aligned to electrophysiology timestamps and samples for each specified node
- Pandas dataframe containing aligned events