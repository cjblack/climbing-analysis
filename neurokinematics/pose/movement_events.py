"""Module for extracting movement events from pose data.

Currently this module supports extraction of starting and stopping of discrete events (i.e. stance->swing transitions) using a velocity thresholding approach.

The main function, `extract_movements` was designed specifically for use with naturalistic, freely-moving climbing. Therefore, implementation with other motor behaviours may require titrating parametrs.

Future updates will include 
    - Config files to simplify call/reproduce workflows
    - Additional methods for extracting movement features from different behaviour types

"""

from pathlib import Path

from scipy.signal import find_peaks
import pandas as pd
import dask
import numpy as np
import xarray as xr

#from neurokinematics.pose.utils import load_df_list
from neurokinematics.pose.io import load_df_list

def extract_movements(df: pd.DataFrame, node_list: list, height: float = 10., distance: int = 100, thresh: float = 0.1, pre_window_s: float = 0.0):
    """Extracts start and stop time indices of node movements, as well as maximum velocity during movement bouts.

    Optionally extends each extracted event window *backwards* in time by
    ``pre_window_s`` seconds so that pre-movement (baseline / preparatory)
    samples are captured. The detected movement onset is preserved separately
    (``onset``) so downstream code can label which samples precede movement; the
    velocity-threshold semantics of the returned ``movements_df`` are unchanged.

    Args:
        df (pd.DataFrame): Pandas Dataframe containing markerless pose estimation from one trial.
        node_list (list): List of nodes to extract movement information from.
        height (float, optional): Height cutoff in pixels for identifying movements. Defaults to 10..
        distance (int, optional): Distance between movement bouts in samples, this will be based on camera frame rate and expected time between movements. Defaults to 100.
        thresh (float, optional): Threshold in pixels of what is considered a movement. Defaults to 0.1.
        pre_window_s (float, optional): Seconds of pre-movement data to prepend to
            each event window. Converted to frames using the trial's frame rate
            and clamped at the start of the trial. ``0.0`` (default) reproduces the
            original onset-to-end window.

    Returns:
        pd.DataFrame: Pandas DataFrame containing start, stop, and maximum velocity indices for each node.
        movement_list (list): List containing movement trajectories for each node with respect to a reference node.
            Each entry carries ``start`` (window start, possibly pre-onset),
            ``onset`` (detected movement onset), ``n_pre`` (pre-movement frames in
            the window), and ``end``.

    Example:
        >>> movements_df, movement_list = extract_movements(
        ...     df = pose_df,
        ...     node_list = ['node1', 'node2', 'node3', 'node4'],
        ...     pre_window_s = 0.2,
        ...     )
    """

    stances = dict()
    movement_array = dict()
    movement_list = []
    trial_ = df['Trial'].min()#int(df.attrs['Trial'].split('T')[-1])
    date_ = df['Date'].min()
    id_ = df['Id'].min()
    type_ = df['Type'].min()
    frame_rate = df['SampleRate'].min()
    pre_frames = int(round(pre_window_s * frame_rate)) if pre_window_s and pre_window_s > 0 else 0
    for i, node in enumerate(node_list):
        y=df[node+'_Y'].to_numpy()
        y_diff = np.abs(np.diff(y))#np.gradient(y))
        pos_peaks, _ = find_peaks(y_diff, height=height, distance=distance)

        start_end = get_start_and_end(y_diff,pos_peaks,threshold=thresh)
        start_ = []
        end_ = []
        max_ = []
        x_ = dict()
        y_ = dict()
        node_array_dict = dict()
        for i, idxs in enumerate(start_end): #for idxs in range(len(start_end)):
            onset_idx = idxs[0]   # detected movement onset (velocity-threshold crossing)
            end_idx = idxs[1]#start_end[idxs][1]
            start_.append(onset_idx)
            end_.append(end_idx)
            max_.append(onset_idx+np.argmax(y_diff[onset_idx:end_idx]))

            # extend window backwards to capture pre-movement samples (clamped at trial start)
            win_start = max(0, onset_idx - pre_frames)
            n_pre = onset_idx - win_start   # pre-movement frames actually available
            mov_len = end_idx-win_start
            node_array = np.zeros((mov_len, len(node_list), 2)) #2 coordinates
            # carry per-node confidence (point scores) through if present
            has_scores = all((nd + '_score') in df.columns for nd in node_list)
            score_array = np.full((mov_len, len(node_list)), np.nan) if has_scores else None
            node_array_list = []
            for ii, nd in enumerate(node_list):
                 node_array[:, ii, 0] = df[nd+'_X'].iloc[win_start:end_idx].to_numpy() # x-coord
                 node_array[:, ii, 1] = df[nd+'_Y'].iloc[win_start:end_idx].to_numpy() # y-coord
                 if has_scores:
                     score_array[:, ii] = df[nd+'_score'].iloc[win_start:end_idx].to_numpy()
                 node_array_list.append(nd) # append double to keep track

            node_array_dict = {
                'node_array': node_array,
                'score_array': score_array,
                'node_list': node_array_list,
                'start': win_start,       # first sample of the (possibly extended) window
                'onset': onset_idx,       # detected movement onset within the window
                'n_pre': n_pre,           # number of pre-movement frames before onset
                'end': end_idx,
                'movement_length': node_array.shape[1],
                'reference_node': node,
                'no_nodes': len(node_list),
                'trial': trial_,
                'date': date_,
                'id': id_,
                'type': type_,
                'frame_rate': frame_rate
            }
            movement_list.append(node_array_dict)
        stances[node]={'start':start_,'end':end_, 'max':max_}
    stances['trial'] = trial_
    stances['date'] = date_

    return pd.DataFrame.from_dict(stances), movement_list

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