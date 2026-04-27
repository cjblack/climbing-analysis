""" 
Batch preprocessing markerless pose estimation data.


"""

from pathlib import Path
import glob

import dask.dataframe as dd
import pandas as pd

from neurokinematics.io import load_config
from neurokinematics.pose.io import dask_batch_load_files
from neurokinematics.data.processed import PoseProcessed
from neurokinematics.pose.preprocessing.cleaning import *
from neurokinematics.pose.movement_events import extract_movements



def process_sleap(data_path: str, pose_cfg: str):
    
    # now requires config file
    cfg = load_config(pose_cfg, config_type='pose') #POSE_PREPROCESSING_CONFIG_PATH / preprocess_cfg)
    preprocessing = cfg['pose_preprocessing']
    file_format = cfg['pose_format']['file_format']
    sample_rate = cfg['pose_format']['frame_rate']
    movement_detection = cfg['movement_detection']

    # get / creat paths
    data_path = Path(data_path)
    pose_path = data_path / 'pose'
    pose_path.mkdir(exist_ok=True)

    file_path = (data_path / f'*{file_format}').as_posix()
    pose_output_path = pose_path / 'pose_data.csv'
    me_output_path = pose_path / 'movement_events.pkl'

    file_list = glob.glob(file_path)

    # call dask batch load - this is lazy
    ddf = dask_batch_load_files(
        file_list,
        sample_rate=sample_rate,
        preprocess=preprocessing
    )

    # save dataframe - right now we're computing before saving
    ddf = ddf.compute()
    ddf.to_csv(pose_output_path)

    # extract movements
    if movement_detection['enabled']:
        group_cols = movement_detection['group_cols']
        sort_cols = movement_detection['sort_cols']
        node_list = movement_detection['node_list']
        nunique_cols = ddf.groupby(group_cols)[group_cols].nunique().__len__()
        movement_events = [None] * nunique_cols
        ddf_group = ddf.sort_values(sort_cols).groupby(group_cols)
        for (date_, trial_), df in ddf_group:
            movement_events.append(extract_movements(df, node_list))
    
        movement_events_df = pd.concat(movement_events)
        pd.DataFrame.to_pickle(movement_events_df, me_output_path)

    # create lazy pose object
    pose_processed_obj = PoseProcessed(
        pose_output_path = pose_output_path,
        storage_format = 'csv',
        fps = sample_rate,
        preprocess = cfg,
        movement_output_path = me_output_path
    )

    return pose_processed_obj