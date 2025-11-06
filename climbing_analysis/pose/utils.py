from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import numpy as np
import h5py
import os
import glob
import h5py
from pathlib import Path
import pickle



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
        if attrs[key] is not None:
            df.attrs = attrs[key]
            dflist.append(df)

    return dflist

def create_df(locs, node_locs,fps=200):
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
    #locDictionary['timestamps'] = np.arange(0,locs.shape[0],1)*(1/fps) #timestamps in seconds
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

def batch_load_files(file_list,preprocess=False):
    dfs = []
    for file in file_list:
        df = load_file(file,preprocess)
        df.attrs['Path'] = Path.cwd()#os.getcwd()
        dfs.append(df)
    return dfs

def load_file(filename,sample_rate=200,preprocess=False):
    # load h5 file - can offload from function into a separate data handle
    # load h5 file - can offload from function into a separate data handle
    with h5py.File(filename, "r") as f:
        locations = f["tracks"][:].T  # x,y coords of labeled joints
        node_names = [n.decode() for n in f["node_names"][:]]  # get node names, somewhat redundant given the next line
        node_locs = dict([(name, i) for i, name in enumerate(node_names)])  # create dictionary of {joint: idx}
    locations =fill_missing(locations)
    poseDF = create_df(locations, node_locs)
    if preprocess==True:
        poseDF = KNP.remove_coordinate_jumps(poseDF)
    dirInfo = os.path.split(filename) # file info
    exInfo = str.split(dirInfo[1],'_') # experiment info
    subId = exInfo[0]
    exType = exInfo[1]
    exDate = exInfo[2]
    exTrial = exInfo[3].split('.')[0]
    poseDF.attrs = {'Path':dirInfo[0],'File':dirInfo[1],'Id':subId,'Type':exType,'Date':exDate,'Trial':exTrial, 'SampleRate':sample_rate}
    return poseDF

def fill_missing(Y, kind="linear"):
    """*Taken from sleap's pose estimation tools** Fills missing values independently along each dimension after the first."""
    # Store initial shape.
    initial_shape = Y.shape
    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))
    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

def pixels_to_cm(wall='2mm_basic'):
    # wall distance 6mm average = 25.3px
    # wall distance 6mm std = 1.6px
    if wall == '2mm_basic':
        px_to_cm = (6.0/25.3)*0.1 #ratio to multiply pixels by
    return px_to_cm

def load_pickle(fname):
    with open(fname, "rb") as f:  # "rb" = read binary
        data = pickle.load(f)
    return data

def get_trial_order(dflist):
    trial_ids = []
    for df in dflist:
        tid = int(df.attrs['Trial'].split('T')[-1])
        trial_ids.append(tid)
    trial_ids_sort = np.argsort(trial_ids)
    return trial_ids_sort