""" 
Batch preprocessing markerless pose estimation data.

This module provides utilties for loading, preprocessing, and saving pose data from markerless pose-estimation pipelines.
It is currently only tested with SLEAP data converted to the '.h5' format. DLC and other pose estimation models, and file formats are will be developed in the future.

Main function, `process_sleap`, loads pose files from a single directory, applies pre-defined preprocessing of pose data, and optionally extracts movement events. 

Details of pre-processing steps and movement event extraction are set by the selected config file from `neurokinematics/configs/pose_cfg`. These files can be used as templates for custom configs.
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
    """Preprocess markerless pose data from sleap stored in a single directory.

    Args:
        data_path (str): Path to folder containing sleap `.h5` files
        pose_cfg (str): Config file stored in `configs/pose_cfg` directory

    Returns:
        PoseProcessed: Lightweight class storing metadata for preprocessing steps
    """

    # now requires config file
    cfg = load_config(pose_cfg, config_type='pose') #POSE_PREPROCESSING_CONFIG_PATH / preprocess_cfg)
    preprocessing = cfg['pose_preprocessing']
    file_format = cfg['pose_format']['file_format']
    sample_rate = cfg['pose_format']['frame_rate']
    movement_detection = cfg['movement_detection']

    # get / create paths
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

    # save dataframe, this is eager as we need to compute before saving
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