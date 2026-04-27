[![Unit Test](https://github.com/cjblack/neurokinematics/actions/workflows/session_class_test.yml/badge.svg)](https://github.com/cjblack/neurokinematics/actions/workflows/session_class_test.yml)

 # Quick install
 Clone repository, and create environment:
 ```bash
git clone https://github.com/cjblack/neurokinematics.git
cd neurokinematics
conda env create -f environment_redux.yml
 ```

 For cuda capabilities (recommended if using this pacakge for spike sorting), use the following:
 ```bash
 conda env create -f environment_cuda.yml --solver=libmamba
 ```
 
 # Overview

 This repository provides a pipeline for processing and analysing electrophysiology (neuro) and markerless pose estimation (kinematics) data from behavioural neuroscience experiments.

 Tools included for
 * Spike sorting
 * LFP processing
 * Synchronising neural and behavioural recordings

 This code base is designed with the custom climbing behaviour in mind ([Naturalistic climbing reveals adaptive strategies for interlimb coordination in freely moving mice](https://doi.org/10.1016/j.isci.2026.115901)).

# Sorting spikes
Currently tested with data acquired from Cambridge Neurotech H5 probe using the Open Ephys acquisition system, and spikesorting with kilosort4.

This requires a config file to be stored in the `configs/spikes_cfg` in the main project directory. This will need to be updated and tested depending on your probe/acquisition system.

```python

from neurokinematics.ephys.spikes.sorting import sort

# Set data directory and param file
data_path = 'path/to/.oebin'
cfg_file = 'cfgfile.yaml' # located in 'configs/spike_cfg'

# Sort spikes
sorting, recording, probe, analyzer = sort(data_path=data_path, cfg_file=param_file)

```

Data can then be viewed with phy2.

# Processing LFP
Custom pre-processing of raw lfp traces

```python
from neurokinematics.ephys.lfp import process_lfp

data_path = 'path/to/datafolder'

# chunk, filter, and downsample
process_lfp(data_path)
```

# Video alignment
This currently will return, among other things a list of arrays called `frame_captures`. Each list index is the start of a new video, and each value in the array corresponds to the ephys sample that a video frame was captured. This is also saved as a `.csv` file withint the `events` folder in the data directory.

```python
from neurokinematics.ephys.events import get_camera_events

data_path = 'path/to/datafolder'
camera_channel = 67

event_data, ts, bouts, frame_captures, _ = get_camera_events(data_path, event_channel = camera_channel)
```

# Process a session
Process workflow ingests recording session into DataJoint pipeline. Does not perform signal processing yet, simply registers and structures data in the database.

Ensure you have created a .env file with the relevant fields that allows you to link to DB of choice.

```shell
python workflows.process_session "path/to/datafolder"
```

```python
from workflows.process_session import run
data_path = "path/to/datafolder"
run(data_path)

```


# Loading climbing session
This is for loading ephys and pose data together in one session object. You'll need to have relevant pose data stored in a `PoseData` folder within the folder containing your open ephys acquired data.

At the moment this is specific to the wall climbing video acquisition and analysis pipeline.
```python
from neurokinematics.data.session import ClimbingSession

data_path = 'path/to/ephys/data/folder'

csession = ClimbingSession(data_path)
```