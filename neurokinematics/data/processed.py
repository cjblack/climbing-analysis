""" Processed data containers for analysis pipelines.

This module defines lightweight classes that act as handles to preprocessed data stored on disk. These objects store paths and metadata, and provide methods for loading data on demand.

"""

from dataclasses import dataclass
import pathlib
from pathlib import Path
from typing import Optional

from neurokinematics.io import load_zarr, load_memmap, load_json, load_csv, load_pickle

@dataclass
class LFPProcessed:
    """Lightweight class for preprocessed lfp data and associated metadata.
    """
    output_path: Path
    shape: tuple
    dtype: str
    fs: float
    metadata_path: Path
    chunkmap_path: Path
    storage_format: str
    
    def load(self, return_metadata=False):
        """Load data

        Args:
            return_metadata (bool, optional): _description_. Defaults to False.

        Returns:
            data (array) or data, metadata (dict): Lazy load data with zarr or np.memmap depending on storage_format. Optionally return dictionary of metadata.
        """
        if self.storage_format == "memmap":
            data = load_memmap(self.output_path, shape=self.shape, dtype=self.dtype, mode='r')
            metadata = load_json(self.metadata_path)
            return data, metadata
        elif self.storage_format == "zarr":
            data, metadata = load_zarr(self.output_path, dataset="processed", mode="r")
            if return_metadata:
                return data, metadata
            return data
        
@dataclass
class PoseProcessed:
    """Lightweight class for preprocessed pose data.
    """
    pose_output_path: Path
    storage_format: str
    preprocess: dict
    fps: float
    movement_output_path: Optional[Path] = None

    def load_pose(self, pkg_format: str = 'dask'):
        """Load pose estimation data into dataframe.

        Args:
            pkg_format (str, optional): Method to use for loading data. Options are 'dask', 'pandas'. Using 'dask' provides a lazy load, while 'pandas' will load into memory. Defaults to 'dask'.

        Returns:
            data (dataframe): Dataframe containing preprocessed pose estimation data.
        """
        if self.storage_format == "csv":
            data = load_csv(self.pose_output_path, pkg_format=pkg_format)
            return data
        
    def load_movement(self, pkg_format: str = 'pandas'):
        """Load movement event data into dataframe using neurokinematics.io.load_pickle

        Args:
            pkg_format (str, optional): Method to use when reading in data. Options are 'default', which uses pickle, or 'pandas'. Defaults to 'pandas'.

        Returns:
            data (dataframe): Dataframe containing extracted movement events.
        """
        if type(self.movement_output_path) == pathlib.WindowsPath:
            data = load_pickle(self.movement_output_path, pkg_format)
            return data
