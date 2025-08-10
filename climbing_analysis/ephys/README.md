# Ephys analysis tools
Bare bones simplification spike sorting with single command. Currently designed for analysing open ephys data with kilosort4, but will be more general in the future.

```python
from climbing_analysis.ephys.spike_sorting import sort_spikes
sorting, recording, probe = sort_spikes(data_path='path/to/structure.oebin')
```