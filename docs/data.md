# data
The `data` module provides session-level abstractions for organising, processing, and aligning multi-modal experimental datasets.

It is designed to coordinate workflows involving electrophysiology and markerless pose estimation, providing a unified interface for processing time-synchronised behavioural and neural recordings across experimental sessions.

## Module structure

- `session.py` &rarr; session class for orchestrating preprocessing, alignment, and data management
- `processed.py` &rarr; lightweight classes for accessing processed outputs and associated metadata

## Loading a session

The current implementation has been developed around the wall climbing acquisition and analysis workflow, and assumes defined file naming convention for pose data.

```python
from neurokinematics.data.session import ExperimentSession

session = ExperimentSession(
    session_id = 'exp_01',
    ephys_data_path = 'path/to/ephys/folder',
    pose_data_path = 'path/to/pose/folder',
    output_root_path = 'path/to/output',
    cfg = 'session.yaml' # user defined config
)
```

### What this does
- Instantiates a session class
- Creates folder structure for data storage
- Points to ephys and pose data

### Inputs
- User defined session identifier
- Open Ephys recording directory
- Pose data directory (containing `.h5` files for the session)
- Output directory for `neurokinematics` folders
- Session specific config file

### Outputs
- `ExperimentSession` object
- Access to preprocessing, alignment, and metadata management workflows

The instantiated object can execute individual processing stages:

```python
# run pose processing
session.run_pose_processing()
# run spike sorting
session.run_spike_sorting()
# run lfp processing
session.run_lfp_processing()

# alignment
session.align_video()
session.align_movements()
```

Alternatively, the full preprocessing and alignment workflow can be executed through a single high-level call:
```python
session.preprocess_and_align()
```

### What this does
- Coordinates pose, spike, and LFP preprocessing workflows
- Executes multi-modal alignment steps in the correct dependency order
- Tacks session-specific metadata and processing outputs
- Provides a unified interface for downstream analysis across experimental sessions

## Reload session
```python
from neurokinematics.data.session import ExperimentSession

reloaded_session = ExperimentSession.from_existing(session_path = "path/to/previously/created/session")

```

### What this does
- Provides access to the previously created session

### Inputs
- Path of previously created session

### Outputs
- `ExperimentSession` object containing metadata and access to a previously created session

## Additional utilities

The `data` module is under active development and is intended to expand into a higher-level orchestration layer for large-scale experimental datasets, including support for batch processing, database integration, and cross-session analyses.