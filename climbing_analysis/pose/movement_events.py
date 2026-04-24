from pathlib import Path

from scipy.signal import find_peaks
import pandas as pd
import numpy as np

from climbing_analysis.pose.utils import load_df_list

def save_stances(directory):
    pose_directory = Path(directory) / 'pose'
    post_data_path = pose_directory / 'pose.h5'

    if not pose_data_path.exists():
        raise NameError('pose.h5 file not found.')
    
    df_list = load_df_list(pose_data_path)
    df_list_len = df_list.__len__()
    stance_list = [None]*df_list_len
    for i, df in enumerate(df_list):
        stance_list[i] = extract_stances(df)
    
    stance_dfs = pd.concat(stance_list)
    pd.DataFrame.to_csv(stance_dfs, pose_directory / 'stances.csv')


def extract_movements(df,height=10, distance=100, thresh=0.1):
    
    NODES = ['r_forepaw','l_forepaw','r_hindpaw','l_hindpaw']

    stances = dict()
    trial = int(df.attrs['Trial'].split('T')[-1])
    for i, node in enumerate(NODES):
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
    stances['trial'] = trial

    return pd.DataFrame.from_dict(stances)

def get_start_and_end(data, peaks, threshold):
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