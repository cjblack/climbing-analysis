from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import spikeinterface.full as si
import spikeinterface.widgets as sw
from climbing_analysis.ephys.utils import *
from climbing_analysis.pose.utils import pixels_to_cm
from scipy.ndimage import gaussian_filter1d
from climbing_analysis.ephys.preprocessing.filters import downsample_lfp_fast

def bin_spikes(spike_times, T, fs=200.0):

    edges = np.arange(T+1) / fs # this avoids increased len from small floating point errors
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts

def extract_data_from_session(csession, fs=200.0):
    unit_ids = csession.good_unit_idxs
    sorting = csession.sorter
    dflist = csession.pose_df_list
    frame_captures = csession.frame_captures
    csession.get_lfp_data()
    lfp = csession.lfp_samples


    kinematic_data = []
    spike_data = []  
    meta_data = []  
    lfp_data = []

    node_xy = np.unique(dflist[0].keys())
    nodes = np.unique([n[:-2] if n.endswith(('_X','_Y')) else n for n in node_xy]).tolist()
    nodes = nodes[:4] # this will need to be more generalizable

    # format trial data    
    trial_ids = []
    for df in dflist:
        tid = int(df.attrs['Trial'].split('T')[-1])
        trial_ids.append(tid)
    trial_ids_sort = np.argsort(trial_ids)

    # format spike data
    spike_trains = []
    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_trains.append(spike_train)

    # format lfp data
    lfp_ds, lfp_ds_fs = downsample_lfp_fast(lfp, fs_in=30000.0)
    lfp_ts = np.linspace(0,lfp_ds.shape[1],lfp_ds.shape[1])/fs

    # Get spike rasters for each movement across climbing trials for single session
    for ii in range(len(frame_captures)):
        bt = frame_captures[ii]/30000
        trial_len = dflist[trial_ids_sort[ii]].shape[0]
        bout_start_id = len(bt) - (trial_len)
        aligned_spikes = spike_train - bt[bout_start_id]
        spks_per_trial = []
        meta_data.append(dflist[trial_ids_sort[ii]].attrs)
        lfp_start_id = np.argmin(np.abs(lfp_ts-bout_start_id))
        # Get kinematics of remaining nodes
        node_array = np.zeros((len(nodes),trial_len))
        binned_spikes = []
        
        for i, n in enumerate(nodes):
            node_array[i,:] = dflist[trial_ids_sort[ii]][n+'_Y'].to_numpy()
        for strain in spike_trains:
            binned_spikes.append(bin_spikes(strain, trial_len,fs))
        lfp_data.append(lfp_ds[:,lfp_start_id:lfp_start_id+trial_len])
        binned_spikes = np.array(binned_spikes)
        spike_data.append(binned_spikes)
        kinematic_data.append(node_array)
    
    return spike_data, kinematic_data, lfp_data, meta_data

def save_trials(spike_data, kinematic_data, meta_data, output_dir):

    assert len(spike_data) == len(kinematic_data)
    output_dir = output_dir / 'binned_data'
    os.makedirs(output_dir, exist_ok=True)

    for i, (spk, kin) in enumerate(zip(spike_data, kinematic_data)):
        spk = np.asarray(spk)
        kin = np.asarray(kin)

        if spk.shape[1] != kin.shape[1]:
            raise ValueError(f'Time mismatch in trial {i}')
        trial_no = meta_data[i]['Trial'].strip('T')
        np.savez(os.path.join(output_dir, f"trial_{trial_no}.npz"), spikes=spk.astype(np.float32), kin=kin.astype(np.float32))



def extract_raster_movements(unit_ids, sorting, dflist, frame_captures, fs_e = 30000.0, fs_k = 200.0, bin_size=0.02, smooth_sigma=1.0, prune_trials=True,save_fig=None):
    """
    Extract spike rasters and node movements with respect to target node
    mv,mv_rn,spks = extract_raster_movements([7],csession.sorter,csession.pose_df_list,csession.frame_captures, csession.stances)
    """

    node_xy = np.unique(dflist[0].keys())
    nodes = np.unique([n[:-2] if n.endswith(('_X','_Y')) else n for n in node_xy]).tolist()

    dt = 1 / fs_k # set time step to kinematic sampling rate (fs_k), which is < ephys sampling rate (fs_e)
    
    spike_train_total = []
    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train_total.append(spike_train)
    spike_train_total = np.concatenate(spike_train_total)
    spike_train = spike_train_total/30000
    
    # Set lists
    spikes_to_store = []
    kin_to_store = []
    kin_to_store_rn = [] # rn - remaining nodes
    spks_per_trial_total = []
    all_counts = []
    bins = np.arange(xlim_[0], xlim_[1] + bin_size, bin_size)
    iii = 0


    # Sort sessions in numerical order
    trial_ids = []
    for df in dflist:
        tid = int(df.attrs['Trial'].split('T')[-1])
        trial_ids.append(tid)
    trial_ids_sort = np.argsort(trial_ids)

    # Get spike rasters for each movement across climbing trials for single session
    for ii in range(len(frame_captures)):
        bt = frame_captures[ii]/30000
        bout_start_id = len(bt) - (dflist[trial_ids_sort[ii]].__len__())
        #times_ = np.array(stances[trial_ids_sort[ii]][node][epoch_loc])
        aligned_spikes = spike_train - bt[bout_start_id]
        spks_per_trial = []

        # Get kinematics of node for corresponding trial
        movement = dflist[trial_ids_sort[ii]][node+'_Y'].to_numpy()
        # Get kinematics of remaining nodes
        node_array = np.zeros((len(nodes),len(movement)))
        for i, n in enumerate(nodes):
            node_array[i,:] = dflist[trial_ids_sort[ii]][n+'_Y'].to_numpy()
        for i, tstart in enumerate(times_):
            tstart_samp = tstart
            tstart = tstart/200.0
            spikes_in_window = aligned_spikes[(aligned_spikes>(tstart-0.5)) & (aligned_spikes <=(tstart+0.5))]
            spikes_to_store.append(spikes_in_window-tstart)
            spks_per_trial.append([i,spikes_in_window-tstart])
            # Store kinematics of node
            kin_to_store.append((movement[tstart_samp-100:tstart_samp+100]-movement[tstart_samp])*pixels_to_cm())
            # Store kinematics of mirrored node
            kin_to_store_rn.append((node_array[:,tstart_samp-100:tstart_samp+100]-node_array[:,[tstart_samp]])*pixels_to_cm())
            counts, _ = np.histogram(spikes_in_window-tstart, bins=np.arange(xlim_[0],xlim_[1]+bin_size,bin_size))
            all_counts.append(counts)
        spks_per_trial_total.append(spks_per_trial)


    return kin_to_store, kin_to_store_rn, spikes_to_store, nodes