""" 
High-level I/O for saving and loading.

Contains simplified versions of storage operations to reduce overhead across code. 
Keep this module to saveas/load of file various file formats.

saveas
- json
- pandas dataframe -> csv

load
- json
- zarr
- memmap
- pickle
- yaml
"""

from pathlib import Path
import json
import pickle
import zarr
import pandas as pd
import dask.dataframe as dd
import yaml

###
### CONFIG PATHS
###

CFG_ROOT_PATH = Path(__file__).resolve().parent.parent / 'configs'


CFG_PATHS = {
    'multimodal': CFG_ROOT_PATH / 'multimodal_cfg',
    'pose': CFG_ROOT_PATH / 'pose_cfg',
    'spksorting': CFG_ROOT_PATH / 'spk_sorting_cfg'
}

###
### SAVING
###

def saveas_json(file_path: str, data: dict):
    """Save dictionary to json file.

    Args:
        file_path (str): File path to save data, ending in '.json'
        data (dict): Dictionary containing data to be saved.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def saveas_dataframe_to_csv(file_path: str, data: list):
    """Convert a list of dicts to a pandas dataframe and save it as a csv. Simplifies saving meta/chunk data.

    Args:
        file_path (str): File path to save data, ending in '.csv'
        data (list): List of dictionaries to be converted to a dataframe and saved as csv.
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


###
### LOADING...
###

def load_csv(file_path: str, pkg_format: str = 'pandas'):
    """Load csv files.

    Args:
        file_path (str): File path, ending in '.csv'
        pkg_format (str, optional): String of the dataframe package to use for loading the .csv file. Options are 'dask' and 'pandas'. 'dask' will do a lazy load. Defaults to 'pandas'.

    Returns:
        dataframe: Pandas or dask dataframe, depending on 'pkg_format' argument. Defaults to a pandas dataframe.
    """
    pkg_format = pkg_format.lower()
    if pkg_format == 'pandas':
        df = pd.read_csv(file_path)
    elif pkg_format == 'dask':
        df = dd.read_csv(file_path)
    return df

def load_json(file_path: str):
    """Load json files.

    Args:
        file_path (str): File path, ending in '.json'

    Returns:
        dict: Dictionary containing data from loaded .json file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_zarr(file_path: str, dataset: str, mode="r"):
    """Open zarr store - load is a misnomer for utiltiy sake, using open allows lazy access to data.

    Args:
        file_path (str): Path pointing to zarr store
        dataset (str): Folder within store to access.
        mode (str, optional): Reading mode for loading zarr store. Defaults to "r".

    Returns:
        data (zar.core.Array): Chunked and compressed data in zarr store.
        dict: Dictionary containing zarr store attributes
    """
    root = zarr.open(file_path, mode = mode)
    data = root[dataset]
    return data, dict(root.attrs)

def load_memmap(file_path: str, shape: tuple, dtype: str ="float32", mode: str = "r"):
    """Load a numpy memmap file

    Args:
        file_path (str): File path, ending in '.dat'
        shape (tuple): Shape of memory mapped data
        dtype (str, optional): Data type of memory mapped data. Defaults to "float32".
        mode (str, optional): Memmap read mode. Defaults to "r".

    Returns:
        memmap (np.ndarray): Returns the memory mapped object of the data stored on disk.
    """
    return np.memmap(file_path, dtype=dtype, mode=mode, shape=tuple(shape))

def load_pickle(filename: str, pkg_format: str = 'default'):
    
    pkg_format = pkg_format.lower()
    if pkg_format == 'default':
        with open(filename, "rb") as f:  # "rb" = read binary
            data = pickle.load(f)
    elif pkg_format == 'pandas':
        data = pd.read_pickle(filename)
    #elif pkg_format == 'dask':
    #    data = dd.read_pickle(filename)
    return data

def load_config(filename: str, config_type: str | None = None):
    """Load yaml files as dictionaries.

    Args:
        filename (str): Config file name ending in '.yaml'
        config_type (str | None, optional): Type of config file, simplify accessing config folders. Options: 'pose', 'multimodal', 'spike', None. If None, it will only load the config file if the full path to the file is provided. Defaults to None.

    Returns:
        _type_: _description_
    """
    if type(config_type) == str:
        filename = CFG_PATHS[config_type] / filename
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config