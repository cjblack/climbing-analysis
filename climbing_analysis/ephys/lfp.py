from pathlib import Path
import numpy as np
import zarr
import pandas as pd
import json
import matplotlib.pyplot as plt
from spikeinterface.sorters import run_sorter
from spikeinterface import create_sorting_analyzer
from spikeinterface.exporters import export_to_phy
from spikeinterface.extractors import read_phy
from climbing_analysis.ephys.utils import *
from climbing_analysis.utils import saveas_json, saveas_dataframe, load_json
from climbing_analysis.ephys.preprocessing.chunking import iter_chunks
from climbing_analysis.ephys.preprocessing.filters import filter_lfp
from climbing_analysis.pose.utils import pixels_to_cm, get_trial_order
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly
import mne



def process_lfp(data_path: Path, fs_new=1000.0, chunk_duration_s=10.0, pad_duration_s=1.0, n_aux_chans = 11, filter_info={"n_": 50.0, "bp_":(0.1,100.0), "quality": 30.0}, dtype="float32", storage_format="memmap"):
    """Performs basic pre-processing of LFP data from .continuous recordings. Chunks, filters, and downsamples data to be stored as a memmap for later access.


    Args:
        data_path (Path): Folder path of recorded data
        fs_new (float, optional): Sampling rate to downsample to. Defaults to 1000.0.
        chunk_duration_s (float, optional): Duration in seconds of data chunks. Defaults to 10.0.
        pad_duration_s (float, optional): Duration in seconds for data padding - for filtering. Defaults to 1.0.
        n_aux_chans (int, optional): Number of auxiliary data channels (accelerometer + digital/analog inputs). Defaults to 11.
        filter_info (dict, optional): Dictionary containing filter settings for notch "n_", bandpass "bp_" (low, high), and quality "quality". Defaults to {"n_": 50.0, "bp_":(0.1,100.0), "quality": 30.0}.
        dtype (str, optional): Datatype to store chunked data as. Defaults to "float32".

    Returns:
        dict: High level information of processed LFP data.
    """
    lfp_data = get_lfp(data_path = data_path, node_idx=0, rec_idx=0)
    output_path = data_path / 'lfp_downsampled.dat'
    zarr_path = data_path / 'zarr_out'
    metadata_path = data_path / 'lfp_metadata.json'
    chunkmap_path = data_path / 'lfp_chunk_map.csv'
    fs_og = lfp_data.metadata['sample_rate']
    n_samples, n_channels = lfp_data.samples.shape
    n_channels = n_channels #- n_aux_chans
    # calculate end data shape
    n_samples_out = int(np.ceil(n_samples * fs_new / fs_og))
    # calculate chunk length
    chunk_len_out = int(chunk_duration_s * fs_new)

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

    if storage_format == 'memmap':
        lfp_out = np.memmap(
            output_path,
            dtype=dtype,
            mode="w+",
            shape=(n_samples_out, n_channels)
        )
    
    elif storage_format == 'zarr':
        root = zarr.open_group(str(zarr_path), mode="w")
        lfp_out = root.create_dataset(
            'lfp',
            shape = (n_samples_out, n_channels),
            chunks = (chunk_len_out, n_channels),
            dtype = dtype
        )

        # create attributes
        for k, v in lfp_metadata.items():
            root.attrs[k] = v
    
    lfp_loader = lambda start, end: lfp_data.get_samples(start,end)

    write_pos = 0 # to deal with rounding errors
    chunk_map = []
    for core_start, core_end, chunk in process_lfp_chunks(
        lfp_loader,
        n_samples,
        n_channels,
        fs_og,
        fs_new,
        chunk_duration_s,
        pad_duration_s
    ):
        out_start = write_pos
        out_end = write_pos + chunk.shape[0]
        lfp_out[out_start:out_end, :] = chunk.astype("float32")
        
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
    saveas_dataframe(chunkmap_path, chunk_map)

    return {"output_path": output_path, "shape": (n_samples_out, n_channels), "dtype": dtype, "fs": fs_new}


    
    
def process_lfp_chunks(lfp_loader, n_samples: int, n_channels: int, fs_og: int, fs_new: int, chunk_duration_s: float, pad_duration_s: float):
    """Processes chunks of LFP data by running filters and downsampling.

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

def load_processed_lfp(metadata_path, load_chunk_map=True):

    metadata_path = Path(metadata_path)
    metadata = load_json(metadata_path)

    data_path = Path(metadata["data_loc_processed"])
    if not data_path.is_absolute():
        data_path = metadata_path.parent / data_path
    lfp = np.memmap(
        data_path,
        dtype = metadata["dtype_processed"],
        mode = "r",
        shape =  metadata["shape_processed"]
    )

    chunk_map = None
    if load_chunk_map and "data_loc_chunk_map" in metadata:
        chunk_map_path = Path(metadata["data_lock_chunk_map"])
        if not chunk_map_path.is_absolute():
            chunk_map_path = metadata_path.parent / chunk_map_path
        
        if chunk_map_path.exists():
            chunk_map = pd.read_csv(chunk_map_path)
    
    return lfp, metadata, chunk_map


def get_lfp_samples(self, start_sample_index, end_sample_index):
        """
        Gets LFP data (currently stored in a separate recording node that is hardcoded)
        stores:
            self.lfp: open ephys object, containing lfp data
        """
        self.lfp = get_lfp(self.session_path)
        
        #self.lfp_samples = self.lfp.get_samples(start_sample_index=0, end_sample_index=-1)
        #self.lfp_shape = self.lfp.samples.shape
        #self.lfp_recording_loaded = True
        return self.lfp.get_samples(start_sample_index=start_sample_index, end_sample_index=end_sample_index)


def epoch_lfp(data, dflist, frame_captures, stances, node='r_forepaw',epoch_loc='start',xlim_=[-0.5,0.5], bin_size=0.02, smooth_sigma=1.0, fs=30000, prune_trials=True,save_fig=None):
    #fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3,1]})
    
    tstamp_pose_len = int(np.sum(np.abs(xlim_))*200)
    tstamp_ephys_len = int(np.sum(np.abs(xlim_))*30000)
    tstamps_pose = np.linspace(xlim_[0],xlim_[1],tstamp_pose_len)
    tstamps_ephys = np.linspace(xlim_[0],xlim_[1],tstamp_ephys_len)
    samp_len = xlim_[1]

    # Get proper trial order list
    trial_ids = get_trial_order(dflist)
    erps = []
    kinematics = []
    data, fs_ds = filter_lfp(data, fs=30000)
    for ii in range(len(frame_captures)):
        bt = frame_captures[ii]
        bout_start_id = len(bt) - (dflist[trial_ids[ii]].__len__())
        event_times = np.array(stances[trial_ids[ii]][node][epoch_loc])
        movement = dflist[trial_ids[ii]][node+'_Y'].to_numpy()

        for i, tstart in enumerate(event_times):
            tstart_ = bt[bout_start_id+tstart]
            tstart_ds = tstart_/ fs
            tstart_ds = int(tstart_ds*fs_ds)
            erp_ = data[tstart_ds-int(fs_ds*samp_len):tstart_ds+int(fs_ds*samp_len)] #- np.mean(data[tstart_ds-int(fs_ds/2):tstart_ds])
            erps.append(erp_)
            kinematics.append((movement[tstart-100:tstart+100]-movement[tstart])*pixels_to_cm()) # THIS IS INCORRECT
    
    if prune_trials:
        kinematic_prune = []
        erp_prune = []

        for i, v in enumerate(kinematics):
            if len(v) == 200:
                if epoch_loc == 'start':
                    #if (np.mean(v[50:100]) < 0.5) & (np.mean(v[100:150])>1.5):
                    if (np.max(v[:100]) < 3.0) & (np.max(v[100:])<4.0) & (np.mean(v[100:])>1.5):
                        kinematic_prune.append(v)
                        erp_prune.append(erps[i])
                elif epoch_loc == 'end':
                    #if (np.mean(v[50:100]) < -0.5) & (np.mean(v[100:150]) > -0.5):
                    if (np.max(v[:100]) < 1.0) & (np.min(v[:100])>-8.0) & (np.max(v[100:]) < 5.0) & (np.mean(v[100:])>-0.5):
                        kinematic_prune.append(v)
                        erp_prune.append(erps[i])
                elif epoch_loc == 'max':
                    if (np.max(v[:100]) < 3.0) & (np.max(v[100:]) < 5.0) & (np.mean(v[100:]) > 1.5):
                        kinematic_prune.append(v)
                        erp_prune.append(erps[i])
        erps = erp_prune
        kinematics = kinematic_prune
    return np.array(erps), np.array(kinematics), fs_ds

def morlet_lfp(data,dflist,frame_captures,stances,node='r_hindpaw',epoch_loc='start',freqs=np.arange(2,40,1), n_cycles=None, xlim_=[-0.5,0.5], save_fig=None):
    erps,kinematics,fs = epoch_lfp(data,dflist,frame_captures,stances,node=node,epoch_loc=epoch_loc, xlim_=xlim_)
    lfp_epochs = erps.reshape([erps.shape[0],1,erps.shape[1]])
    
    if n_cycles == None:
        n_cycles = freqs/4
    power = mne.time_frequency.tfr_array_morlet(
        lfp_epochs,
        sfreq=fs,
        freqs=freqs,
        n_cycles=n_cycles,
        output='power'
    )
    # single channel, so remove index
    power = power.squeeze()
    # Average prestimulus baseline
    mu = power[:,:,int((fs*(np.abs(xlim_[0])-0.5))/2):int(fs*(np.abs(xlim_[0])-0.1))].mean(axis=2,keepdims=True)
    power_z = ((power-mu)/mu)*100
    #power_db = 10*np.log10(power_z)
    power_avg = power_z.mean(axis=0)
    times = np.linspace(xlim_[0],xlim_[1],int(fs*(np.sum(np.abs(xlim_)))))
    # Plot
    plt.figure(figsize=(8,4))
    plt.pcolormesh(times, freqs,power_avg, shading='gouraud', cmap='turbo')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Average Spectrogram: {node} {epoch_loc}  movement (n={str(power.shape[0])})')
    cbar = plt.colorbar()
    cbar.set_label('Relative Power (%)')
    plt.xlim(xlim_)
    if save_fig:
        plt.savefig(save_fig+'/morlet_example.pdf')
        plt.savefig(save_fig+'/morelet_example.png')
    plt.show()
    return power_z
