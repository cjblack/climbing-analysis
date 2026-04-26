# Ephys analysis tools
Bare bones simplification spike sorting with single command. Currently designed for analysing open ephys data with kilosort4, but will be more general in the future.

Param files are currently located in the project root folder in `configs/spike_cfg`. If using this, be sure to edit the fields as necessary. `climbing_sorting_params.yaml` references a cambridge neurotech H5 probe with custom channel map `h5_open_ephys_acquisition_channel_map.npy`. Standard H5 channel map is `h5_channel_map.npy` (located in `neurokinematics/ephys/channel_maps` folder).

```python
from neurokinematics.ephys.spike.sorting import sort
cfg_file = 'climbing_sorting_cfg.yaml'
sorting, recording, probe, analyzer = sort(data_path='path/to/structure.oebin', cfg_file=cfg_file)
```
