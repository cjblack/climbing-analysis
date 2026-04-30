# spikes

The `spikes` module provides tools for spike sorting, waveform inspection, and movement-aligned single-unit analysis. It integrates with SpikeInterface for spike sorting and analysis, while providing additional utilities for aligning sorted units with behavioural events derived from pose estimation.

Spike sorting with neurokinematics requires the directory to an ephys recording and a [spike sorting config file](https://github.com/cjblack/neurokinematics/tree/main/configs/spk_sorting_cfg). The config will need to be updated and tested depending on your probe/acquisition system.

## Module structure

- `sorting.py` &rarr; core module for spike sorting with Open Ephys recordings
- `rasters.py` &rarr; alignment of sorted units to movement events
- `plotting.py` &rarr; visualisation utilities for waveforms, rasters, and PSTHs

## Notes

- Currently tested with Open Ephys recordings from Cambridge Neurotech H5 probes
- Spike sorting workflows are configured around Kilosort4 and SpikeInterface
- Configuration files may require adaptation for different probes, acquisition systems, or sorter settings
- Movement-aligned analyses require a pre-computed `movement_event_alignment.csv` file

## Run spike sorting

```python
from neurokinematics.ephys.spikes.sorting import sort

# Set data directory and param file
data_path = 'path/to/.oebin'
cfg_file = 'cfgfile.yaml' # located in 'configs/spike_cfg'
save_path = 'path/to/savefolder' # set to desired save location, default will store to data_path directory

# Sort spikes
sorting, recording, probe, analyzer = sort(data_path=data_path, cfg_file=cfg_file, save_path=save_path)
```

### What this does
- Loads Open Ephys recordings and probe metadata
- Runs spike sorting through SpikeInterface using a configuration designed workflow
- Computes waveform and unit metrics through SpikeInterface
- Stores results for downstream visualisation and manual curation

### Inputs
- Path to Open Ephys `.oebin` file
- Spike sorting config file
- Optional save path

### Outputs
- Sorting object
- Analyzer object
- Probe metadata
- Recording object
- Files compatible with Phy2 for manual inspection

## Waveform plotting
Neurokinematics can use the `spikeinterface` results stored during spike sorting for plotting.

```python
from neurokinematics.ephys.io import load_analyzer
from neurokinematics.ephys.spikes.plotting import plot_waveforms

# Load spikeinterface sorting analyzer
analyzer_path = 'path/to/analyzer/folder'
analyzer = load_analyzer(analyzer_path)

# Plot
plot_waveforms(
    analyzer = analyzer, 
    unit_ids = [12, 16, 17], 
    save_path = "path/to/outputs"
    )
```
### What this does
- Loads a saved analyzer object
- Plot waveforms of selected units
- Optionally saves waveform figures

![Example waveforms](unit_waveforms.png)

## Movement-aligned rasters and PSTHs
This step requires a pre-computed movement alignment file (`movement_event_alignment.csv`).
```python
from neurokinematics.ephys.io import load_phy_sorting
from neurokinematics.ephys.spikes.rasters import get_movement_aligned_rasters

# Load sorter
phy_sorter_path = 'path/to/phy/output'
sorter = load_phy_sorting(phy_sorter_path)

# Plot spike aligned rasters
spike_rasters = get_movement_aligned_rasters(
    alignment = alignment_df, # pre computed movement alignment dataframe
    sorter = sorter,
    save_path = "path/to/outputs"
)
print(spike_rasters.output_path)
spike_rasters_df = spike_rasters.load() # returns spike rasters as dataframe
```
### What this does
- Aligns spike times to behavioural events defined in `movement_event_alignment.csv`
- Stores movement-aligned spike trains for downstream analysis
- Returns a lightweight object for loading aligned rasters

This file can then be used for plotting the resulting spike rasters.

```python
from neurokinematics.io import load_pickle
from neurokinematics.ephys.spikes.plotting import plot_movement_psth

raster_df = load_pickle('path/to/movement_aligned_rasters.pkl')
unit_ids = [16, 17, 12]
movement_plot_params = 
    {  
    'node': 'r_hindpaw',
    'movement_event': 'end',
    'cmap': 'winter',
    'bin_size': 0.05
    }

plot_movement_psth(raster_df, unit_ids, movement_plot_params) # plot with respect to end of movement
movement_plot_params['movement_event'] = 'max'
plot_movement_psth(raster_df, unit_ids, movement_plot_params) # plot with respect to maximum velocity of movement
```
### What this does
- Create a combined spike raster and peri-stimulus time histogram for units aligned to movement events.
- Values set in `movement_plot_params` determine what node and movement event is plotted.
- Optionally save the plot as a `.png`.

![Example rasters](r_hindpaw_end_3_units_psth.png) ![Example rasters](r_hindpaw_max_3_units_psth.png)

