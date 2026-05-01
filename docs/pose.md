# pose

The `pose` module provides tools for preprocessing markerless pose estimation outputs (currently SLEAP) and extracting movement-related features for downstream analysis.

It is designed to standardise pose data across sessions and optionally derive movement events that can later be aligned with electrophysiology data.

## Module structure

- `preprocessing/` &rarr; core preprocessing pipeline for SLEAP data
    - `base.py` &rarr; high-level utility (`process_sleap`)
    - `cleaning.py` &rarr; filtering and interpolation utilities
- `movement_events.py` &rarr; extraction of movement events from processed pose data. Utilises velocity thresholding
- `coordination.py` &rarr; analysis of inter-limb coordination using phase offset metric
- `plotting.py` &rarr; visualisation utilities for pose data and derived features
- `io.py` &rarr; input/output helpers for loading and saving pose-related data


> **Note:** currently, sleap files need to have a specific naming convention for parsing file content.
>
> the convention is `<subj_id>_<experiment_type>_<YYYYMMDD>_<TN>.h5`
> where: 
>> - subj_id = the subject id
>> - experiment_type = type of experiment run
>> - YYYMMDD = date format
>> - TN = T represents 'trial', N represents trial number
> 
>
>The format itself has no bearing on the actual data, but is used for organisational purposes in function calls, so other formats will raise issues. Future updates will optimising the naming convention to be more modular across experiments.

## Process SLEAP files
```python
from neurokinematics.pose.preprocessing.base import process_sleap
pose = process_sleap(
    data_path = "path/to/sleap/h5/files",
    pose_cfg = "demo_pose_cfg.yaml",
    save_path = "path/to/save/directory" # optional
)

print(pose.pose_output_path)
pose_df = pose.load_pose()
movement_df = pose.load_movement() # if enabled
```
### What this does
- Load and preprocess all .h5 SLEAP files in the `data_path` based on the config file.
- Applies filtering, interpolation, and feature extraction steps defined in the config file.

### Inputs
- Directory containing SLEAP `.h5` files for a recording session.
- YAML configuration file defining preprocessing steps and pose-related parameters
- Save path

### Outputs
- `pose_data.csv` &rarr; processed pose time series
- `movement_events.pkl` &rarr; (*optional*) extracted movement events
- Lightweight object for accessing results

## Additional utilities

The `pose` module also includes simple analysis and visualization tools for exploring pose data. These utilities are evolving and will be expanded in future updates.

### Phase offset analysis

The `coordination` module provides functions for computing inter-limb phase relationships from processed pose data, allowing quantification of coordination patterns during movement.

For example usage, see [phase offset demo](/notebooks/phase_offset_demo.ipynb).