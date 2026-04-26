""" 
Processed data containers for analysis pipelines.

This module defines lightweight, "lazy" classes that represent outputs from preprocessing steps. These objects serve as handles for data stored on disk, rather than in memory.

"""

from dataclasses import dataclass
from pathlib import Path

from climbing_analysis.io import load_zarr, load_memmap, load_json, load_csv

@dataclass
class LFPProcessed:
    """Lightweight class for preprocessed lfp data.
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
            _type_: _description_
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
    output_path: Path
    storage_format: str
    preprocess: dict
    fps: float

    def load(self, pkg_format: str = 'dask'):
        if self.storage_format == "csv":
            data = load_csv(self.output_path, pkg_format=pkg_format)
            return data
