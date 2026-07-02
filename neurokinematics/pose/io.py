from pathlib import Path
import numpy as np
import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import h5py
import os
import glob

from neurokinematics.pose.preprocessing.cleaning import fill_missing, remove_high_velocity, remove_low_confidence
from neurokinematics.pose.metadata import resolve_file_metadata
from neurokinematics import io

import pickle
import xarray as xr

dask.config.set({"dataframe.convert-string": False}) # reduces unnecessary conversions during dask compute calls



### save ###
def save_movement_dataset(ds: xr.Dataset, save_path: Path | str, chunks: dict | None = None, event_chunk: int = 100, overwrite: bool = True):
    """Save xarray dataset as zarr store

    Args:
        ds (xr.Dataset): Xarray dataset to be stored
        save_path (Path | str): Save path of zarr store, must end in new folder name
        chunks (dict | None, optional): Dictionary containing chunk relevant information. Defaults to None.
        event_chunk (int, optional): Sets chunk size for events...current not in use. Defaults to 100.
        overwrite (bool, optional): Determines whether zarr store is overwritten. Defaults to True.
    """
    save_path = Path(save_path)
    # ensure ending with .zarr
    if save_path.suffix != '.zarr':
        save_path = save_path.with_suffix('.zarr')

    if chunks:
        ds = ds.chunk(chunks = chunks)

    if overwrite:
        mode = "w"
        save_path.mkdir(parents = True, exist_ok = True)
    else:
        mode = "w-"
        save_path.mkdir(parents = True, exist_ok = False)

    ds.to_zarr(save_path, mode=mode)


### load ###
def load_df_list(df_list_filename):
    dflist = []
    dfs = {}
    attrs = {}
    with pd.HDFStore(df_list_filename, mode='r') as store:
        for key in store.keys():
            try:
                df = store.get(key)
                if isinstance(df, pd.DataFrame):
                    # Load the DataFrame
                    dfs[key] = df

                    # Load its attributes if present
                    metadata = getattr(store.get_storer(key).attrs, 'metadata', None)
                    attrs[key] = metadata
            except Exception as e:
                print(f"Skipping {key}: {e}")

    for key, df in dfs.items():
        raw_attr = attrs.get(key)
        if raw_attr is None:
            dflist.append(df)
            continue

        try:
            df.attrs = raw_attr
        except Exception:
            try:
                df.attrs = pickle.loads(raw_attr)
            except Exception as e:
                print(f"Could not load attrs for {key}: {e}")
                df.attrs={}
        if 'Path' in df.attrs:
            try:
                # Convert WindowsPath to string safely
                if "Path" in str(type(df.attrs['Path'])):
                    df.attrs['Path'] = str(df.attrs['Path'])
            except Exception as e:
                print(f"Warning: could not sanitize 'Path' in {key}: {e}")
                df.attrs['Path'] = str(df.attrs['Path']) if hasattr(df.attrs['Path'], '__str__') else 'INVALID'

        dflist.append(df)    

    return dflist

def save_df_list(df_list):
    """
    Save list of data frames from SLEAP
    """
    df_names = []
    dates_ = []
    id_ = df_list[0].attrs['Id']
    type_ = df_list[0].attrs['Type']
    date_ = df_list[0].attrs['Date']
    path_ = df_list[0].attrs['Path']
    trial_ = df_list[0].attrs['Trial']

    for df in df_list:
        trial_ = df.attrs['Trial']
        date_ = df.attrs['Date']
        dates_.append(date_)
        name_ = date_+'_'+trial_
        df_names.append(name_) # set df name as trial
    unique_dates = np.unique(dates_)
    sub_path_ = id_+'_'+type_+'_'#+date_+'_DFs.h5'
    #file_name = path_ / sub_path_
    if len(unique_dates) == 1:
        sub_path_ = sub_path_ + unique_dates[0]+ '_DFS.h5'
        file_name = path_ / sub_path_
    else:
        sub_path_ = sub_path_ + 'Batch_DFS.h5'
        file_name = path_ / sub_path_
    with pd.HDFStore(file_name, mode='w') as store:
        for name, df in zip(df_names, df_list):
            store.put(name,df)
            store.get_storer(name).attrs.metadata = df.attrs

def _node_point_score(point_scores, node_idx):
    """Per-frame point/confidence score for one node (instance 0)."""
    arr = np.asarray(point_scores)
    if arr.ndim >= 3:      # (frames, nodes, instances)
        return arr[:, node_idx, 0]
    if arr.ndim == 2:      # (frames, nodes)
        return arr[:, node_idx]
    return arr             # (frames,)


def _frame_score(scores):
    """Per-frame instance/tracking score (instance 0)."""
    arr = np.asarray(scores)
    if arr.ndim >= 2:      # (frames, instances)
        return arr[:, 0]
    return arr             # (frames,)


def create_df(locs, node_locs, fps=200.,
              coords = ["X", "Y"], point_scores=None, instance_scores=None, tracking_scores=None):
    '''
    Creates a data frame with predictions (x,y coordinates) for each joint, and appends timestamps based on frame rate.

    When confidence scores are supplied they are stored alongside the coordinates so
    they propagate to every downstream file:
      - ``<node>_score``   : per-node point score (confidence per body part)
      - ``instance_score`` : overall confidence for the animal/object in that frame
      - ``tracking_score`` : confidence of identity assignment over time

    :param locs:
    :param node_locs:
    :param fps:
    :param point_scores: per-node point scores, shape (frames, nodes[, instances]).
    :param instance_scores: per-frame instance scores.
    :param tracking_scores: per-frame tracking scores.
    :return:
    '''
    locDictionary = dict()
    for node, val in node_locs.items():
        for i, coord in enumerate(coords):
            locDictionary[node+f'_{coord}'] = locs[:,val,i,0]
            #locDictionary[node+'_Y'] = locs[:,val,1,0]*-1
        if point_scores is not None:
            locDictionary[node+'_score'] = _node_point_score(point_scores, val)
    locDictionary['frame_id'] = np.arange(0,locs.shape[0],1) #timestamps in seconds
    if instance_scores is not None:
        locDictionary['instance_score'] = _frame_score(instance_scores)
    if tracking_scores is not None:
        locDictionary['tracking_score'] = _frame_score(tracking_scores)
    poseDf = pd.DataFrame(data=locDictionary)
    return poseDf

def get_df_list(id: str, directory: str, exp_type: str, date_: str = '', preprocess: bool =True):
    '''
    Creates a list of data frames for dataset
    :param id:
    :param preprocess:
    :return:
    '''

    os.chdir(directory) # change directory to get access to data

    if id == 'Group': # if subject id is group, load all files in folder
        files = glob.glob('*_' + exp_type + '_'+date_+'*h5') # create list of file names
    else: # otherwise, load specific subject data
        files = glob.glob(id + '_' + exp_type + '_'+date_ + '*h5') # get all analysis filetypes

    df_list = batch_load_files(files, preprocess=preprocess) # create list of data frames for each file

    return df_list

def batch_load_files(file_list,sample_rate=200., preprocess=False):
    dfs = [None] * len(file_list)
    for i, file in enumerate(file_list):
        df = load_file(file,sample_rate, preprocess)
        #df.attrs['Path'] = Path.cwd()#os.getcwd()
        dfs[i] = df
    return dfs

def load_file(filename,sample_rate=200.,preprocess=False):

    with h5py.File(filename, "r") as f:
        locations = f["tracks"][:].T  # x,y coords of labeled joints
        point_scores = f["point_scores"][:].T
        instance_scores = f["instance_scores"][:].T
        tracking_scores = f["tracking_scores"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]  # get node names, somewhat redundant given the next line
        node_locs = dict([(name, i) for i, name in enumerate(node_names)])  # create dictionary of {joint: idx}
    locations =fill_missing(locations)
    poseDF = create_df(locations, node_locs,
                       point_scores=point_scores,
                       instance_scores=instance_scores,
                       tracking_scores=tracking_scores)
    if preprocess==True:
        poseDF = KNP.remove_coordinate_jumps(poseDF)
    dir_info = os.path.split(filename) # file info
    exp_info = str.split(dir_info[1],'_') # experiment info
    sub_id = exp_info[0]
    exp_type = exp_info[1]
    exp_date = exp_info[2]
    exp_trial = exp_info[3].split('.')[0]
    poseDF.attrs = {'Path':dir_info[0],'File':dir_info[1],'Id':sub_id,'Type':exp_type,'Date':exp_date,'Trial':exp_trial, 'SampleRate':sample_rate}
    return poseDF

def dask_batch_load_files(file_list: list, tracker: str, meta_cfg: dict, sample_rate: float = 200., preprocess: dict | None = None):
    """Create a dask dataframe of all data, useful for distributed processing. File metadata are columnar entries. This is handled differently from batch_load_files, as pandas attributes are not partition specific.

    Args:
        file_list (list): List of strings of h5 files to load.
        sample_rate (float, optional): Camera sample rate in frames per second. Defaults to 200.0.
        preprocess (bool, optional): Run preprocessing steps on pose data if True - this is simple at the moment but will expand. Defaults to False.

    Returns:
        dask.dataframe: Lazy load of pose estimation data.
    """


    ddfs = dd.from_map(dask_load_file, file_list, tracker = tracker, sample_rate=sample_rate, preprocess=preprocess, meta_cfg=meta_cfg)

    
    return ddfs

def load_sleap(filename: str):
    """SLEAP loader

    Args:
        filename (str): _description_

    Returns:
        _type_: _description_
    """
    filename = Path(filename)
    with h5py.File(filename, "r") as f:
        locations = f["tracks"][:].T  # x,y coords of labeled joints
        point_scores = f["point_scores"][:].T
        instance_scores = f["instance_scores"][:].T
        tracking_scores = f["tracking_scores"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]  # get node names, somewhat redundant given the next line
        node_locs = dict([(name, i) for i, name in enumerate(node_names)])  # create dictionary of {joint: idx}
    locations[:,:,1,:] *= -1
    return {
            'locations': locations, 
            'point_scores': point_scores, 
            'instance_scores': instance_scores, 
            'tracking_scores': tracking_scores, 
            'node_names': node_names,
            'node_locs': node_locs,
            'coords': ["X", "Y"],
            }

def load_dlc(filename: str, n_subjects: int =1):
    """Anipose loader

    Args:
        filename (str): _description_
        n_subjects (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    filename = Path(filename)
    if filename.suffix == '.h5':
        df = io.load_file(filename)
        n_frames = df.shape[0]
        node_names = list(dict.fromkeys(df.columns.get_level_values(1)))
        node_locs = dict([(name, i) for i, name in enumerate(node_names)])
        coords = list(dict.fromkeys(df.columns.get_level_values(2)))

        n_nodes = len(node_names)
        #n_coords = len(list(dict.fromkeys(df.columns.get_level_values(2)))) - 1 # likelihood is one coord
        if 'likelihood' in coords:
            coords.remove('likelihood')
            point_scores = np.full((n_frames, n_nodes, n_subjects), np.nan)
        else:
            point_scores = None
        n_coords = len(coords)
        locations = np.full((n_frames, n_nodes, n_coords, n_subjects), np.nan)
        #point_scores = np.full((n_frames, n_nodes, n_subjects), np.nan)
        for j, node in enumerate(node_names):
            for i, coord in enumerate(coords):
                locations[:, j, i, 0] = df.xs((node, coord), level=(1,2), axis=1).to_numpy().ravel()
            if point_scores is not None:
                point_scores[:, j, 0] = df.xs((node, "likelihood"), level=(1,2), axis=1).to_numpy().ravel()
        coords = [coord.upper() for coord in coords] # make same convention as sleap with upper case coordinates

    else:
        raise ValueError(f"Invalid file type '{filename.suffix}'.")
    return {
        'locations': locations,
        'point_scores': point_scores,
        'node_names': node_names,
        'node_locs': node_locs,
        'coords': coords
        }

POSE_LOADERS = {
    'sleap': load_sleap,
    'anipose': load_dlc,
    'dlc': load_dlc,
}

def dask_load_file(filename: str, tracker: str, meta_cfg: dict, sample_rate: float = 200., preprocess: dict | None = None):
    """Load H5 data into a pandas dataframe for converting to dask dataframe. Compared to load_files, this stores file metadata as columnar instead of as dataframe attributes.

    Args:
        filename (str): H5 file path to be loaded.
        sample_rate (float, optional): Camera sample rate in frames per second. Defaults to 200.0.
        preprocess (bool, optional): Run preprocessing steps on pose data if True - this is simple at the moment but will expand. Defaults to False.

    Returns:
        pandas.DataFrame: Dataframe of pose estimation time series for extracted X and Y coordinates.
    """
    try:
        load_func = POSE_LOADERS[tracker]
    except KeyError:
        raise ValueError(f"No loader for tracker '{tracker}'. Avaliable trackers: {list(POSE_LOADERS.keys())}")
    loaded_pose = load_func(filename)

    # with h5py.File(filename, "r") as f:
    #     locations = f["tracks"][:].T  # x,y coords of labeled joints
    #     point_scores = f["point_scores"][:].T
    #     instance_scores = f["instance_scores"][:].T
    #     tracking_scores = f["tracking_scores"][:].T
    #     node_names = [n.decode() for n in f["node_names"][:]]  # get node names, somewhat redundant given the next line
    #     node_locs = dict([(name, i) for i, name in enumerate(node_names)])  # create dictionary of {joint: idx}
    #locations =fill_missing(locations)
    
    locations = loaded_pose['locations']
    point_scores = loaded_pose.get('point_scores', None)
    instance_scores = loaded_pose.get('instance_scores', None)
    tracking_scores = loaded_pose.get('tracking_scores', None)
    node_names = loaded_pose.get('node_names', None)
    node_locs = loaded_pose.get('node_locs', None)
    coords = loaded_pose.get('coords', ["X", "Y"])

    if preprocess is None:
        preprocess = {}
    # max gap (in frames) to interpolate across; longer gaps stay NaN. None
    # keeps the original behaviour of filling every gap.
    max_gap = preprocess.get("max_gap", None)
    if not preprocess:
        locations = fill_missing(locations, max_gap=max_gap)
    if preprocess.get("fill_missing", True):
        locations = fill_missing(locations, max_gap=max_gap)
    if preprocess.get("confidence", {}).get("enabled", False):
        locations = remove_low_confidence(locations, point_scores, thresh = preprocess['confidence'].get('thresh', 0.7), max_gap=max_gap)
    if preprocess.get("velocity", {}).get("enabled", False):
        locations = remove_high_velocity(locations, thresh=preprocess['velocity'].get('thresh', 20.), max_gap=max_gap)

    df = create_df(
                    locations, 
                    node_locs, 
                    fps = sample_rate,
                    coords = coords,
                    point_scores = point_scores,
                    instance_scores = instance_scores,
                    tracking_scores = tracking_scores
                    )

    dir_info = os.path.split(filename) # file info
    #exp_info = str.split(dir_info[1],'_') # experiment info
    #sub_id = exp_info[0]
    #exp_type = exp_info[1]
    #exp_date = exp_info[2]
    #exp_trial = exp_info[3].split('.')[0].split('T')[1]

    meta = resolve_file_metadata(filename, meta_cfg)

    df['Path'] = dir_info[0]
    df['File'] = dir_info[1]

    df['Id'] = meta['Id']#sub_id
    df['Type'] = meta['Type']#exp_type
    df['Date'] = pd.to_datetime(meta['Date'])#pd.to_datetime(exp_date)
    df['Trial'] = int(meta['Trial'])#int(exp_trial)
    df['SampleRate'] = sample_rate

    return df


def load_pickle(fname):
    with open(fname, "rb") as f:  # "rb" = read binary
        data = pickle.load(f)
    return data

