[![Unit Test](https://github.com/cjblack/neurokinematics/actions/workflows/session_class_test.yml/badge.svg)](https://github.com/cjblack/neurokinematics/actions/workflows/session_class_test.yml)

 # neurokinematics

 ## Overview

 This repository provides a pipeline for processing and analysing electrophysiology (neuro) and markerless pose estimation (kinematics) data from behavioural neuroscience experiments.

 Tools included for
 * Spike sorting
 * LFP processing
 * Synchronising neural and behavioural recordings

 This code base is designed with the custom climbing behaviour in mind ([Naturalistic climbing reveals adaptive strategies for interlimb coordination in freely moving mice](https://doi.org/10.1016/j.isci.2026.115901)).

 ## Quick install
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


## Usage

### [`pose`](https://github.com/cjblack/neurokinematics/tree/main/neurokinematics/pose)

### Process SLEAP files
```python
from neurokinematics.pose.preprocessing.base import process_sleap
pose = process_sleap(
    data_path = "path/to/sleap/h5/files",
    pose_cfg = "simple_pose_cfg.yaml",
    save_path = "path/to/save/directory # optional
)

print(pose.pose_output_path)
pose_df = pose.load_pose()
movement_df = pose.load_movement() # if enabled
```
This will
- Load and preprocess all .h5 SLEAP files in the `data_path` based on the config file.
- Save processed pose data to `pose_data.csv`.
- Optionally extract and save movement events to `movement_events.pkl` (if enabled in the config file).
- Returns a lightweight object to examine metadata and load processed pose and/or movement event (if enabled) results as a dataframe.

If `save_path` is not provided, outputs are written to `pose/` folder inside `data_path`.

### [`spikes`](https://github.com/cjblack/neurokinematics/tree/main/neurokinematics/ephys/spikes)
Currently tested with data acquired from Cambridge Neurotech H5 probe using the Open Ephys acquisition system, and spikesorting with kilosort4.

### Run spike sorting
This subpackage uses `spikeinterface` to perform spike sorting and some plotting. 

Spike sorting with neurokinematics requires simply the directory to an ephys recording and a [spike sorting config file](https://github.com/cjblack/neurokinematics/tree/main/configs/spk_sorting_cfg). The config will need to be updated and tested depending on your probe/acquisition system.

```python
from neurokinematics.ephys.spikes.sorting import sort

# Set data directory and param file
data_path = 'path/to/.oebin'
cfg_file = 'cfgfile.yaml' # located in 'configs/spike_cfg'
save_path = 'path/to/savefolder' # set to desired save location, default will store to data_path directory

# Sort spikes
sorting, recording, probe, analyzer = sort(data_path=data_path, cfg_file=cfg_file, save_path=save_path)
```

Data can then be viewed with phy2.

### Plotting spike data
Neurokinematics can use the spikeinterface objects stored during spike sorting, along with some of spikeinterfaces widgets for plotting.

```python
from neurokinematics.ephys.io import load_analyzer
from neurokinematics.ephys.spikes.plotting import plot_waveforms

analyzer_path = 'path/to/analyzer/folder'
plots_path = 'path/to/save/plots'
analyzer = load_analyzer(analyzer_path)
unit_ids = [12, 16, 17]
plot_waveforms(analyzer, unit_ids, save_path = plots_path)
```
![Example waveforms](docs/unit_waveforms.png)

### Extracting and plotting rasters
```python
from neurokinematics.ephys.spikes.rasters import get_movement_aligned_rasters
spike_rasters = get_movement_aligned_rasters(
    alignment = alignment_df,
    sorter = sorting,
    save_path = "path/to/outputs"
)
print(spike_rasters_obj.output_path)
spike_rasters_df = spike_rasters.load() # returns spike rasters as dataframe
```
This will
- Align spikes for units in the `sorter` object to movement times defined in `alignment`.
- Save aligned rasters as `movement_aligned_rasters.pkl` to `save_path`.
- Return lightweight class to examine metadata and load aligned spikes as a dataframe.

This file can then be used for plotting the resulting spike rasters.

```python
from neurokinematics.io import load_pickle
from neurokinematics.ephys.spikes.plotting import plot_movement_psth

raster_df = load_pickle('path/to/movement_aligned_rasters.pkl')
unit_ids = [16, 17, 12]
movement_plot_params = 
    {  
    'pre_event': 0.5,
    'post_event': 0.5,
    'node': 'r_hindpaw',
    'movement_event': 'end',
    'cmap': 'winter'
    'bin_size': 0.05
    }

plot_movement_psth(raster_df, unit_ids, movement_plot_params) # plot with respect to end of movement
movement_plot_params['movement_event'] = 'max'
plot_movement_psth(raster_df, unit_ids, movement_plot_params) # plot with respect to maximum velocity of movement
```

![Example rasters](docs/r_hindpaw_end_3_units_psth.png) ![Example rasters](docs/r_hindpaw_max_3_units_psth.png)

### [`lfp`](https://github.com/cjblack/neurokinematics/tree/main/neurokinematics/ephys/lfp)
### Pre-process raw lfp data from OpenEphys

```python
from neurokinematics.ephys.lfp.preprocessing import preprocess_lfp

lfp_proc_obj = preprocess_lfp(
    data_path = "path/to/ephys", 
    node_idx = 0, # based on record node id
    rec_idx = 0, # based on recording folder id
    save_path = "path/to/outputs"
    ) 
lfp, metadata = lfp_proc_obj.load(return_metadata=True) # Load data and metadata
```
This will
- Load Open Ephys .continuous data (specified by `node_idx` and `rec_idx`).
- Chunk, downsample, and filter all channels.
- Store results in a zarr store (default) or memory map.
- Return lightweight class to examine metadata and load processed lfp data.

### [`multimodal`](https://github.com/cjblack/neurokinematics/tree/main/neurokinematics/multi_modal)
This currently will return, among other things a list of arrays called `frame_captures`. Each list index is the start of a new video, and each value in the array corresponds to the ephys sample that a video frame was captured. This is also saved as a `.csv` file withint the `events` folder in the data directory.

```python
from neurokinematics.ephys.events import get_camera_events

data_path = 'path/to/datafolder'
camera_channel = 67

event_data, ts, bouts, frame_captures, _ = get_camera_events(data_path, event_channel = camera_channel)
```

### [`data`](https://github.com/cjblack/neurokinematics/tree/main/neurokinematics/data)

#### Loading a session
This is for loading ephys and pose data together in one session object. You'll need to have relevant pose data stored in a `PoseData` folder within the folder containing your open ephys acquired data.

At the moment this is specific to the wall climbing video acquisition and analysis pipeline.
```python
from neurokinematics.data.session import ClimbingSession

data_path = 'path/to/ephys/data/folder'

csession = ClimbingSession(data_path)
```


## In development 
### [`workflows`](https://github.com/cjblack/neurokinematics/tree/main/workflows)
Process workflow ingests recording session into DataJoint pipeline. Does not perform signal processing yet, simply registers and structures data in the database.

Ensure you have created a .env file with the relevant fields that allows you to link to DB of choice.

#### Terminal example

```shell
python workflows.process_session "path/to/datafolder"
```

#### Python example
```python
from workflows.process_session import run
data_path = "path/to/datafolder"
run(data_path)
```