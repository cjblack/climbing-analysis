""" 
Batch preprocessing markerless pose estimation data.

This module provides utilties for loading, preprocessing, and saving pose data from markerless pose-estimation pipelines.
Current testing has only been performed with SLEAP data converted to the '.h5' format. DLC and other pose estimation models, and file formats will be tested in the future.

Main function, `process_sleap`, loads pose files from a single directory, applies pre-defined preprocessing of pose data, and optionally extracts movement events. 

Details of pre-processing steps and movement event extraction are set by the selected config file from `neurokinematics/configs/pose_cfg`. These files can be used as templates for custom configs.
"""

from pathlib import Path
import glob

import dask.dataframe as dd
import pandas as pd
import numpy as np


from neurokinematics.io import load_config, save_dataframe
from neurokinematics.pose.io import dask_batch_load_files
from neurokinematics.data.processed import PoseProcessed
from neurokinematics.pose.preprocessing.cleaning import *
from neurokinematics.pose.movement_events import extract_movements
from neurokinematics.pose.features import pad_movements, build_movement_dataset



def process_sleap(data_path: str, pose_cfg: str, save_path: Path | str | None = None):
    """Preprocess markerless pose data from sleap stored in a single directory.

    Args:
        data_path (str): Path to folder containing sleap `.h5` files.
        pose_cfg (str): Config file stored in `configs/pose_cfg` directory.
        save_path (Path | str | None, optional): Directory to save results to. Defaults to None.

    Returns:
        PoseProcessed: Lightweight class storing metadata for preprocessing steps.
        file_outputs (dict): Dictionary containing information for files created during call

    Example:
        >>> pose_proc_obj, file_outputs = process_sleap(
        ...     data_path = "path/to/converted/sleap/files"
        ...     pose_cfg = "demo_pose_cfg.yaml",
        ...     save_path = "path/to/outputs"
        ... )
        >>> pose_proc_obj.pose_output_path
        PosixPath('path/to/outputs/pose_data.csv')
    """

    # now requires config file — accept either a filename or an already-loaded dict
    cfg = pose_cfg if isinstance(pose_cfg, dict) else load_config(pose_cfg, config_type='pose')
    preprocessing = cfg['pose_preprocessing']
    file_format = cfg['pose_format']['file_format']
    sample_rate = cfg['pose_format']['frame_rate']
    movement_detection = cfg['movement_detection']

    # create file outputs dictionary - currently testing
    file_outputs = dict()

    # get / create paths
    data_path = Path(data_path)
    if save_path:
        # create save_path folder if provided
        pose_path = Path(save_path) #/ 'pose'
        pose_path.mkdir(parents=True, exist_ok=True)
    else:
        # if no save_path provided create folder in data_path
        pose_path = data_path / 'pose'
        pose_path.mkdir(exist_ok=True)

    # set pose and movement event output paths
    pose_output_path = pose_path / 'pose_data.csv'
    me_output_path = pose_path / 'movement_events.pkl'
    
    # get file list to load
    file_path = (data_path / f'*{file_format}').as_posix()
    file_list = glob.glob(file_path)

    # call dask batch load - this is lazy
    ddf = dask_batch_load_files(
        file_list,
        sample_rate=sample_rate,
        preprocess=preprocessing
    )

    # save dataframe, this is eager as we need to compute before saving
    ddf = ddf.compute()
    save_dataframe(ddf, pose_output_path, 'csv') # less modular -> ddf.to_csv(pose_output_path)

    # extract movements
    if movement_detection['enabled']:
        _, me_outputs = extract_movement_features(ddf, movement_detection, pose_path)
        file_outputs.update(me_outputs)

    file_outputs['pose_data'] = {'path': str(pose_output_path), 'file_type': str(pose_output_path.suffix), 'attrs': {}}
    
    # create lazy pose object
    pose_processed_obj = PoseProcessed(
        pose_output_path = pose_output_path,
        storage_format = 'csv',
        fps = sample_rate,
        preprocess = cfg,
        movement_output_path = me_output_path
    )

    return pose_processed_obj, file_outputs


def extract_movement_features(ddf: pd.DataFrame, movement_detection: dict, pose_path: Path | str):
    """Detect movement events from a cleaned pose dataframe and build the movement
    feature dataset (``movement_events.pkl`` + ``movement_features.zarr``).

    Factored out of :func:`process_sleap` so movement features can be regenerated
    independently — e.g. re-run with a different ``pre_window_s`` (pre-movement
    lead-in) without repeating pose cleaning.

    Args:
        ddf (pd.DataFrame): Cleaned pose dataframe (all trials), with the node
            ``*_X`` / ``*_Y`` columns plus ``Trial`` / ``Date`` / ``Id`` / ``Type`` /
            ``SampleRate``.
        movement_detection (dict): The ``movement_detection`` config block
            (``group_cols``, ``sort_cols``, ``node_list``, optional ``pre_window_s``).
        pose_path (Path | str): Directory to write the movement outputs into.

    Returns:
        tuple: ``(movement_ds, file_outputs)`` where ``file_outputs`` carries the
        ``movement_events`` and ``movement_features`` entries.
    """
    pose_path = Path(pose_path)
    me_output_path = pose_path / 'movement_events.pkl'
    file_outputs = {}

    group_cols = movement_detection['group_cols']
    sort_cols = movement_detection['sort_cols']
    node_list = movement_detection['node_list']
    pre_window_s = movement_detection.get('pre_window_s', 0.0)  # pre-movement lead-in (s)

    movement_events = []
    movement_lists = []
    ddf_group = ddf.sort_values(sort_cols).groupby(group_cols)
    for (date_, trial_), df in ddf_group:
        movements, movement_list = extract_movements(df, node_list, pre_window_s=pre_window_s)
        movement_events.append(movements)
        movement_lists.append(movement_list)

    movement_events_df = pd.concat(movement_events)
    save_dataframe(movement_events_df, me_output_path, 'pickle')

    movement_lists = np.concat(movement_lists)  # concatenate per-group lists
    padded, mov_list, valid, lengths, padded_scores = pad_movements(movement_lists)
    movement_ds = build_movement_dataset(
        padded, mov_list, valid, lengths,
        padded_scores=padded_scores, save_path=pose_path
    )

    file_outputs['movement_events'] = {'path': str(me_output_path), 'file_type': str(me_output_path.suffix), 'attrs': {}}
    file_outputs['movement_features'] = {'path': str(pose_path / 'movement_features.zarr'), 'file_type': '.zarr', 'attrs': {}}
    return movement_ds, file_outputs