# Ephys analysis tools
Bare bones simplification spike sorting with single command. Currently designed for analysing open ephys data with kilosort4, but will be more general in the future.

Param files are currently located in `climbing_analysis` > `ephys` > `sorting_params`. If using this, be sure to edit the fields as necessary. `climbing_sorting_params.yaml` references a cambridge neurotech H5 probe with custom channel map `h5_open_ephys_acquisition_channel_map.npy`. Standard H5 channel map is `h5_channel_map.npy` (located in `climbing_analysis` > `ephys` > `channel_maps` folder).

```python
from climbing_analysis.ephys.spike_sorting import sort_spikes
param_file = 'climbing_sorting_params.yaml'
sorting, recording, probe, analyzer = sort_spikes(data_path='path/to/structure.oebin', param_file=param_file)
```
