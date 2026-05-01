## In development 

The modules described here are **experimental** and currently under active development.

They are tested locally and used during ongoing development of `neurokinematics`, but their APIs, configuration files, and output formats are subject to change.

### datajoint workflows

Experimental workflows for integrating recording sessions into a database using DataJoint.

Requires a `.env` file with database connection details.

#### Command line example

```shell
python datajoint_workflow.process_session "path/to/datafolder"
```

#### Python example
```python
from datajoint_workflow.process_session import run
data_path = "path/to/datafolder"
run(data_path)
```