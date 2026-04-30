## In development 
### datajoint workflows
Experimental workflow ingests recording session into DataJoint pipeline. Currently handles registration and data structuring. Signal processing steps are not yet included.

Requires a `.env` file with database connection details.

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