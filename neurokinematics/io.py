""" 
High-level I/O for saving and loading.

Contains simplified versions of storage operations to reduce overhead across code. 
Keep this module to saveas/load of file various file formats, set config paths, session directories, and associated checks.

config paths

validate
- file existence
- path existence

saving
- json
- pandas dataframe -> csv

loading
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



### * configs * ###

CFG_ROOT_PATH = Path(__file__).resolve().parent.parent / 'configs'


CFG_PATHS = {
    'multimodal': CFG_ROOT_PATH / 'multimodal_cfg',
    'pose': CFG_ROOT_PATH / 'pose_cfg',
    'spksorting': CFG_ROOT_PATH / 'spk_sorting_cfg',
    'lfp': CFG_ROOT_PATH / 'lfp_cfg',
    'session': CFG_ROOT_PATH / 'session_cfg'
}


### * validation * ###

def _require_file(file_path: Path | str):
    """Helper function to check if file for loading exists.

    Args:
        file_path (Path | str): File path to check.

    Raises:
        FileNotFoundError: Checks if file_path exists.
        FileNotFoundError: Checks if file_path is a file.

    Returns:
        Path: Input file_path if it passes checks.
    """
    
    file_path = Path(file_path) # turn into Path

    # does this exist? check
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    # is this a file? check
    if not file_path.is_file():
        raise FileNotFoundError(f"Expected a file, got: {file_path}")
    return file_path

def _require_path(path: Path | str):
    """Helper function to check if path exists.

    Args:
        path (Path | str): Input path to check.

    Raises:
        FileNotFoundError: Checks if path exists.

    Returns:
        Path: Returns path if it exists.
    """
    path = Path(path)
    
    # does this exist? check
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


### * directory * ###

def create_session_dirs(session_dir: str | Path, output_dir_name: str | Path ='neurokinematics'):
    """Create session directory folders.

    Args:
        session_dir (str | Path): Path to store session data/plots
        output_dir_name (str | Path, optional): . Defaults to 'neurokinematics'.

    Returns:
        dirs (dict): Dictionary of created directories with keys: 'root', 'pose', 'ephys', 'alignment', 'events', 'spikes', 'lfp', 'plots'
    """
    session_dir = Path(session_dir)
    output_dir = session_dir / output_dir_name

    dirs = {
        "root": output_dir,
        "pose": output_dir / 'pose',
        "ephys": output_dir / 'ephys',
        "alignment": output_dir / 'alignment',
        'events': output_dir / 'events',
        "spikes": output_dir / 'ephys' / 'spikes',
        "lfp": output_dir / 'ephys' / 'lfp',
        "plots": output_dir / 'plots'
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return dirs



### * saving files * ###

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

def save_dataframe(df, file_path, storage_format:str = 'csv', **kwargs):

    if storage_format == 'csv':
        df.to_csv(file_path)
    elif storage_format == 'pickle':
        df.to_pickle(file_path)
    elif storage_format == 'parquet':
        df.to_parquet(file_path, **kwargs)



### * loading files * ###

def load_csv(file_path: str, method: str = 'pandas'):
    """Load csv files.

    Args:
        file_path (str): File path, ending in '.csv'
        method (str, optional): String of the dataframe package to use for loading the .csv file. Options are 'dask' and 'pandas'. 'dask' will do a lazy load. Defaults to 'pandas'.

    Returns:
        dataframe: Pandas or dask dataframe, depending on 'method' argument. Defaults to a pandas dataframe.
    """

    file_path = _require_file(file_path)

    method = method.lower()
    if method == 'pandas':
        df = pd.read_csv(file_path)
    elif method == 'dask':
        df = dd.read_csv(file_path)
    else:
        raise ValueError("method must be 'pandas' or 'dask'.")
    
    return df

def load_json(file_path: str):
    """Load json files.

    Args:
        file_path (str): File path, ending in '.json'

    Returns:
        dict: Dictionary containing data from loaded .json file.
    """
    file_path = _require_file(file_path)
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

    file_path = _require_file(file_path)

    return np.memmap(file_path, dtype=dtype, mode=mode, shape=tuple(shape))

def load_pickle(filename: str, method: str = 'default'):
    """Loads pickled data.

    Args:
        filename (str): Name of file to load, must end in `.pkl`
        method (str, optional): Method to load data with. Options are 'default' or 'pandas'. Defaults to 'default'.

    Returns:
        data (dict | dataframe): Returns data as a dictionary or a dataframe.
    """

    # check file exists
    file_path = _require_file(filename)
    method = method.lower()
    if method == 'default':
        with open(file_path, "rb") as f:  # "rb" = read binary
            data = pickle.load(f)
    elif method == 'pandas':
        data = pd.read_pickle(file_path)
    else:
        raise ValueError("method must be 'default' or 'pandas'.")
    #elif method == 'dask':
    #    data = dd.read_pickle(filename)
    return data

def load_config(filename: Path | str, config_type: str | None = None):
    """Load yaml files as dictionaries.

    Args:
        filename (Path | str): Config file name ending in '.yaml'
        config_type (str | None, optional): Type of config file, simplify accessing config folders. Options: 'pose', 'multimodal', 'spike', None. If None, it will only load the config file if the full path to the file is provided. Defaults to None.

    Returns:
        _type_: _description_
    """
    if config_type is not None:
        if config_type not in CFG_PATHS:
            valid_ops = ", ".join(CFG_PATHS)
            raise ValueError(f"Invalid config_type '{config_type}'. Valid options are: {valid_ops}")
        
        file_path = CFG_PATHS[config_type] / filename
    else:
        file_path = Path(filename) # for tests

    file_path = _require_file(file_path)

    #if type(config_type) == str:
    #    filename = CFG_PATHS[config_type] / filename
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    return config