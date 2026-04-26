import numpy as np
import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import h5py
import os
import glob
import h5py
from neurokinematics.pose.preprocessing.cleaning import fill_missing, remove_high_velocity, remove_low_confidence
from pathlib import Path
import pickle
import h5py

dask.config.set({"dataframe.convert-string": False}) # reduces unnecessary conversions during dask compute calls

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

def create_df(locs, node_locs,fps=200.):
    '''
    Creates a data frame with predictions (x,y coordinates) for each joint, and appends timestamps based on frame rate
    :param locs:
    :param node_locs:
    :param fps:
    :return:
    '''
    locDictionary = dict()
    for node, val in node_locs.items():
        locDictionary[node+'_X'] = locs[:,val,0,0]
        locDictionary[node+'_Y'] = locs[:,val,1,0]*-1
    locDictionary['frame_id'] = np.arange(0,locs.shape[0],1) #timestamps in seconds
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
        node_names = [n.decode() for n in f["node_names"][:]]  # get node names, somewhat redundant given the next line
        node_locs = dict([(name, i) for i, name in enumerate(node_names)])  # create dictionary of {joint: idx}
    locations =fill_missing(locations)
    poseDF = create_df(locations, node_locs)
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

def dask_batch_load_files(file_list: list, sample_rate: float = 200., preprocess: dict | None = None):
    """Create a dask dataframe of all data, useful for distributed processing. File metadata are columnar entries. This is handled differently from batch_load_files, as pandas attributes are not partition specific.

    Args:
        file_list (list): List of strings of h5 files to load.
        sample_rate (float, optional): Camera sample rate in frames per second. Defaults to 200.0.
        preprocess (bool, optional): Run preprocessing steps on pose data if True - this is simple at the moment but will expand. Defaults to False.

    Returns:
        dask.dataframe: Lazy load of pose estimation data.
    """


    ddfs = dd.from_map(dask_load_file, file_list, sample_rate=sample_rate, preprocess=preprocess)

    
    ## SAVING AS PARQUET
    # ddfs = dd.concat(ddfs)
    # ddfs.to_parquet(data_path / 'pose' / 'processed', engine='pyarrow', compression='zstd')
    return ddfs

def dask_load_file(filename: str,sample_rate: float = 200., preprocess: dict | None = None):
    """Load H5 data into a pandas dataframe for converting to dask dataframe. Compared to load_files, this stores file metadata as columnar instead of as dataframe attributes.

    Args:
        filename (str): H5 file path to be loaded.
        sample_rate (float, optional): Camera sample rate in frames per second. Defaults to 200.0.
        preprocess (bool, optional): Run preprocessing steps on pose data if True - this is simple at the moment but will expand. Defaults to False.

    Returns:
        pandas.DataFrame: Dataframe of pose estimation time series for extracted X and Y coordinates.
    """

    with h5py.File(filename, "r") as f:
        locations = f["tracks"][:].T  # x,y coords of labeled joints
        node_names = [n.decode() for n in f["node_names"][:]]  # get node names, somewhat redundant given the next line
        node_locs = dict([(name, i) for i, name in enumerate(node_names)])  # create dictionary of {joint: idx}
    #locations =fill_missing(locations)
    
    if preprocess is None:
        preprocess = {}
        locations = fill_missing(locations)
    if preprocess.get("fill_missing", True):
        locations = fill_missing(locations)
    if preprocess.get("confidence", {}).get("enabled", False):
        locations = remove_low_confidence(locations, scores, thresh = preprocess['confidence'].get('thresh', 0.7))
    if preprocess.get("velocity", {}).get("enabled", False):
        locations = remove_high_velocity(locations, thresh=preprocess['velocity'].get('thresh', 20.))
    
    df = create_df(locations, node_locs)

    dir_info = os.path.split(filename) # file info
    exp_info = str.split(dir_info[1],'_') # experiment info
    sub_id = exp_info[0]
    exp_type = exp_info[1]
    exp_date = exp_info[2]
    exp_trial = exp_info[3].split('.')[0].split('T')[1]

    df['Path'] = dir_info[0]
    df['File'] = dir_info[1]
    df['Id'] = sub_id
    df['Type'] = exp_type
    df['Date'] = pd.to_datetime(exp_date)
    df['Trial'] = int(exp_trial)
    df['SampleRate'] = sample_rate

    return df


# def pixels_to_cm(wall='2mm_basic'):
#     # wall distance 6mm average = 25.3px
#     # wall distance 6mm std = 1.6px
#     if wall == '2mm_basic':
#         px_to_cm = (6.0/25.3)*0.1 #ratio to multiply pixels by
#     return px_to_cm

def load_pickle(fname):
    with open(fname, "rb") as f:  # "rb" = read binary
        data = pickle.load(f)
    return data

# def get_trial_order(dflist):
#     trial_ids = []
#     for df in dflist:
#         tid = int(df.attrs['Trial'].split('T')[-1])
#         trial_ids.append(tid)
#     trial_ids_sort = np.argsort(trial_ids)
#     return trial_ids_sort