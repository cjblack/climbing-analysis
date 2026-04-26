from dask.distributed import LocalCluster, Client

cluster = LocalCluster()
client = Client(cluster)