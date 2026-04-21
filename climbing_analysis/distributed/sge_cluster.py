from dask_jobqueue import SGECluster
from dask.distributed import Client

cluster = SGECluster(cores=36, memory='100GB')
client = Client(cluster)