Basic analysis tools for data collected during climbing

# Sorting spikes
Currently only tested on open ephys data with cambridge neurotech H5 probe, running kilosort4.

```python

from climbing_analysis.ephys.spike_sorting import sorting spikes

# Set data directory and param file
data_path = 'path/to/datafolder'
param_file = 'path/to/paramfile.yaml'

# Sort spikes
sorting, recording, probe, analyzer = sort_spikes(data_path=data_path, param_file=param_file)

````

Data can then be viewed with phy2.

# Loading climbing session
This is for loading ephys and pose data together in one session object. You'll need to have relevant pose data stored in a `PoseData` folder within the folder containing your open ephys acquired data.

At the moment this is specific to the wall climbing video acquisition and analysis pipeline.
```python
from climbing_analysis.data.session_data import ClimbingSessionData

data_path = 'path/to/ephys/data/folder'

csession = ClimbingSessionData(data_path)
```