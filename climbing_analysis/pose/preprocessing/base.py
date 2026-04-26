from pathlib import Path
import glob

import dask.dataframe as dd

from climbing_analysis.io import load_config
from climbing_analysis.pose.io import dask_batch_load_files
from climbing_analysis.data.processed import PoseProcessed
from climbing_analysis.pose.preprocessing.cleaning import *


POSE_PREPROCESSING_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / 'configs' / 'pose_cfg'


def process_sleap(data_path: str, sample_rate = 200.0, file_format: str = 'h5', storage_format: str = 'csv', preprocess_cfg: str | None = None):
    
    # load config file if provided
    if type(preprocess_cfg) == str:
        preprocess_cfg = load_config(POSE_PREPROCESSING_CONFIG_PATH / preprocess_cfg)

    # get / creat paths
    data_path = Path(data_path)
    file_path = (data_path / f'*{file_format}').as_posix()
    output_path = data_path / 'pose_data.csv'
    file_list = glob.glob(file_path)

    # call dask batch load - this is lazy
    ddf = dask_batch_load_files(
        file_list,
        sample_rate=sample_rate,
        preprocess=preprocess_cfg
    )

    # save dataframe - right now we're computing before saving
    ddf.compute().to_csv(output_path)

    # create lazy pose object
    pose_processed_obj = PoseProcessed(
        output_path = output_path,
        storage_format = storage_format,
        fps = sample_rate,
        preprocess = preprocess_cfg
    )

    return pose_processed_obj