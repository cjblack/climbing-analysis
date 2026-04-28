"""Simplify preprocess of lfp data.

Performs data chunking, filtering, and downsampling before saving data in a zarr store or numpy memory map. 
This module is mainly intended to perform simple, automated, preprocessing of lfp data. 

"""

import math
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import zarr


from neurokinematics.io import saveas_json, saveas_dataframe_to_csv
from neurokinematics.data.processed import LFPProcessed

from neurokinematics.ephys.io import initialize_zarr_store, get_continuous
from neurokinematics.ephys.lfp.chunking import iter_chunks
from neurokinematics.ephys.lfp.filters import filter_lfp

from scipy.signal import resample_poly


def preprocess_lfp(data_path: Path, node_idx: int = 0, rec_idx: int = 0, fs_new=1000.0, chunk_duration_s=10.0, pad_duration_s=1.0, filter_info={"n_": 50.0, "bp_":(0.1,100.0), "quality": 30.0}, dtype="float32", save_path: Path | str | None = None, storage_format="zarr"):
    """Performs basic pre-processing of LFP data from .continuous recordings. Chunks, filters, and downsamples data to be stored as a memmap for later access.

    Args:
        data_path (Path): Folder path of recorded data
        node_idx (int): Index of record node used in open ephys recording
        rec_idx (int): Index of recording idx used in open ephys recording
        fs_new (float, optional): Sampling rate to downsample to. Defaults to 1000.0.
        chunk_duration_s (float, optional): Duration in seconds of data chunks. Defaults to 10.0.
        pad_duration_s (float, optional): Duration in seconds for data padding - for filtering. Defaults to 1.0.
        n_aux_chans (int, optional): Number of auxiliary data channels (accelerometer + digital/analog inputs). Defaults to 11.
        filter_info (dict, optional): Dictionary containing filter settings for notch "n_", bandpass "bp_" (low, high), and quality "quality". Defaults to {"n_": 50.0, "bp_":(0.1,100.0), "quality": 30.0}.
        dtype (str, optional): Datatype to store chunked data as. Defaults to "float32".
        storage_format (str, optional): String indicating file format to store processed data as. Options: "zarr", "memmap". Defaults to "zarr".

    Returns:
        dict: High level information of processed LFP data.

    Usage:
        _ = process_lfp('path/to/ephys/data', storage_format="zarr")
    """

    # create path vars
    data_path = Path(data_path)
    lfp_data, rec_dir = get_continuous(data_path = data_path, node_idx=node_idx, rec_idx=rec_idx) # node_idx and rec_node are hard coded
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
    else:
        save_path = Path(rec_dir) # convert to Path obj
    metadata_path = save_path / 'lfp_metadata.json'
    chunkmap_path = save_path / 'lfp_chunk_map.csv'
    
    # create data vars
    fs_og = lfp_data.metadata.sample_rate 
    n_samples, n_channels = lfp_data.samples.shape
    n_channels = n_channels #- n_aux_chans
    
    # calculate end data shape
    n_samples_out = int(np.ceil(n_samples * fs_new / fs_og))
    
    # calculate chunk length
    chunk_len_out = int(chunk_duration_s * fs_new)

    # setup data storage
    ## memmap
    if storage_format == 'memmap':
        output_path = save_path / 'lfp_preprocessed.dat'

        lfp_metadata = dict({
        "data_loc_original": str(data_path.as_posix()),
        "data_loc_processed": str(output_path.as_posix()),
        "data_loc_chunk_map": str(chunkmap_path.as_posix()),
        "n_channels": n_channels,
        "fs_original": fs_og,
        "fs_processed": fs_new,
        "n_samples_original": n_samples,
        "n_samples_processed": n_samples_out,
        "dtype_processed": dtype,
        "shape_processed": [n_samples_out, n_channels],
        "preprocessed_notch_filter": filter_info["n_"],
        "preprocessed_bandpass_filter": filter_info["bp_"],
        "preprocessed_filter_quality": filter_info["quality"]
        })

        lfp_out = np.memmap(
            output_path,
            dtype=dtype,
            mode="w+",
            shape=(n_samples_out, n_channels)
        )
    
    ## zarr
    elif storage_format == 'zarr':
        output_path = save_path / 'lfp_preprocessed'

        lfp_metadata = dict({
        "data_loc_original": str(data_path.as_posix()),
        "data_loc_processed": str(output_path.as_posix()),
        "data_loc_chunk_map": str(chunkmap_path.as_posix()),
        "n_channels": n_channels,
        "fs_original": fs_og,
        "fs_processed": fs_new,
        "n_samples_original": n_samples,
        "n_samples_processed": n_samples_out,
        "dtype_processed": dtype,
        "shape_processed": [n_samples_out, n_channels],
        "preprocessed_notch_filter": filter_info["n_"],
        "preprocessed_bandpass_filter": filter_info["bp_"],
        "preprocessed_filter_quality": filter_info["quality"]
        })

        initialize_zarr_store(
            zarr_path = output_path,
            n_samples = n_samples_out,
            n_channels = n_channels,
            fs = fs_new,
            chunk_len = chunk_len_out,
            dtype = dtype,
            attrs = lfp_metadata
        )

        root = zarr.open_group(str(output_path), mode="a")
        lfp_out = root['processed']

        # create attributes
        for k, v in lfp_metadata.items():
            root.attrs[k] = v
    
    # lfp_loader function
    lfp_loader = lambda start, end: lfp_data.get_samples(start,end)

    write_pos = 0 # to deal with rounding errors
    chunk_map = []

    # vars for progress bar
    chunk_size = int(chunk_duration_s * fs_og)
    n_chunks = math.ceil(n_samples / chunk_size)

    # create chunking iterator
    chunk_iter = process_lfp_chunks(
        lfp_loader,
        n_samples,
        n_channels,
        fs_og,
        fs_new,
        chunk_duration_s,
        pad_duration_s
    )

    # start chunking
    for core_start, core_end, chunk in tqdm(
        chunk_iter,
        total=n_chunks,
        desc="Processing LFP chunks",
        unit="chunk"
    ):
        out_start = write_pos
        out_end = write_pos + chunk.shape[0]
        lfp_out[out_start:out_end, :] = chunk.astype(dtype)
        
        chunk_map.append({
            "og_start": core_start,
            "og_end": core_end,
            "processed_start": out_start,
            "processed_end": out_end
        })

        write_pos = out_end
    
    if storage_format == "memmap":
        lfp_out.flush() # make sure to flush memory changes to disk

    # save metadata and chunk information
    saveas_json(metadata_path, lfp_metadata)
    saveas_dataframe_to_csv(chunkmap_path, chunk_map)

    lfp_proc_obj = LFPProcessed(
        output_path = output_path,
        shape = (n_samples_out, n_channels),
        dtype = dtype,
        fs = fs_new,
        metadata_path = metadata_path,
        chunkmap_path = chunkmap_path,
        storage_format = storage_format
    )

    return lfp_proc_obj

    
def process_lfp_chunks(lfp_loader, n_samples: int, n_channels: int, fs_og: int, fs_new: int, chunk_duration_s: float, pad_duration_s: float):
    """Iterator to processes chunks of LFP data by running filters and downsampling.

    Args:
        lfp_loader (func): Function to grab chunks of LFP data.
        n_samples (int): Number of samples in recording.
        n_channels (int): Number of channels in recording (includes auxiliary channels as well as ephys channels).
        fs_og (int): Original sampling rate from recording.
        fs_new (int): New sampling rate to downsample data to.
        chunk_duration_s (float): Duration in seconds of LFP chunks.
        pad_duration_s (float): Duration in seconds of padding for LFP chunks.

    Yields:
        core_start (int): Index for start of LFP data in padded chunk.
        core_end (int): Index for end of LFP data in padded chunk.
        ds_filtered_core(np.array): Array containing the filtered, and downsampled LFP core chunk.
    """

    chunk_size = int(chunk_duration_s*fs_og) # chunk duration in samples
    pad_size = int(pad_duration_s*fs_og) # pad duration in samples
    for read_start, read_end, core_start, core_end in iter_chunks(n_samples, chunk_size, pad_size):
        chunk = lfp_loader(read_start, read_end)[:,:n_channels] # only use ephys channels
        
        filtered = filter_lfp(chunk, fs_og) # filter chunk

        trim_start = core_start - read_start
        trim_end = trim_start + (core_end - core_start)
        filtered_core = filtered[trim_start:trim_end]
        ds_filtered_core = resample_poly(filtered_core, fs_new, fs_og, axis=0)

        yield core_start, core_end, ds_filtered_core

