from pathlib import Path

from scipy.signal import find_peaks
import pandas as pd
import dask
import numpy as np

#from neurokinematics.pose.utils import load_df_list
from neurokinematics.pose.io import load_df_list

def extract_movements(df: pd.DataFrame, node_list: list, height: float = 10., distance: int = 100, thresh: float = 0.1):
    """Extracts start and stop time indices of node movements, as well as maximum velocity during movement bouts.

    Args:
        df (pd.DataFrame): Pandas Dataframe containing markerless pose estimation from one trial.
        node_list (list): List of nodes to extract movement information from.
        height (float, optional): Height cutoff in pixels for identifying movements. Defaults to 10..
        distance (int, optional): Distance between movement bouts in samples, this will be based on camera frame rate and expected time between movements. Defaults to 100.
        thresh (float, optional): Threshold in pixels of what is considered a movement. Defaults to 0.1.

    Returns:
        pd.DataFrame: Pandas DataFrame containing start, stop, and maximum velocity indices for each node.
    """

    stances = dict()
    trial_ = df['Trial'].min()#int(df.attrs['Trial'].split('T')[-1])
    date_ = df['Date'].min()
    for i, node in enumerate(node_list):
        y=df[node+'_Y'].to_numpy()
        y_diff = np.abs(np.diff(y))#np.gradient(y))
        pos_peaks, _ = find_peaks(y_diff, height=height, distance=distance)

        start_end = get_start_and_end(y_diff,pos_peaks,threshold=thresh)
        start_ = []
        end_ = []
        max_ = []
        for idxs in range(len(start_end)):
            start_.append(start_end[idxs][0])
            end_.append(start_end[idxs][1])
            max_.append(start_end[idxs][0]+np.argmax(y_diff[start_end[idxs][0]:start_end[idxs][1]]))

        stances[node]={'start':start_,'end':end_, 'max':max_}
    stances['trial'] = trial_
    stances['date'] = date_

    return pd.DataFrame.from_dict(stances)

def get_start_and_end(data: np.array, peaks, threshold: float):
    """Identifies start and stop of movements from a pose estimation time series.

    Args:
        data (np.ndarray): Array containing velocity/diff (n-1 samples) of time series.
        peaks (np.ndarray): Array containing indices of peak velocities/diff of time series.
        threshold (float): Threshold in pixels for what is considered a movement.

    Returns:
        list: List containing tuple of start and end times for movements.
    """
    start_end = []
    for p in peaks:
        idxs = []
        i_s = 1
        i_e = 1
        idx_s = p - i_s
        idx_e = p + i_e
        val_s = data[idx_s]
        val_e = data[idx_e]
        if idx_e >= len(data):
            idx_e = len(data)
            val_e = threshold - 1
        if idx_s <= 0:
            idx_s = 0
            val_s = threshold = 1
        else:
            val_e = data[idx_e]
        # find end index first
        while val_e > threshold:
            i_e = i_e + 1
            idx_e = p + i_e

            if idx_e >= len(data):
                val_e = threshold-1
            else:
                val_e = data[idx_e]
        # find start index last
        while val_s > threshold:
            i_s = i_s + 1
            idx_s = p - i_s

            if idx_s <= 0:
                val_s = threshold-1
            else:
                val_s = data[idx_s]
        if idx_s > 0 and idx_e < len(data):
           idxs = [idx_s,idx_e]
           start_end.append(idxs)
    return start_end