"""Plotting utilities for spike data.

Utilities for visualising spike-sorting outputs and features.
Provides a lightweight abstraction layer over spikeinterface plotting tools along with project-specific plotting fucntions to simplify plotting and saving figures.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spikeinterface.widgets as sw
from neurokinematics.ephys.io import *
from neurokinematics.pose.utils import pixels_to_cm # REMOVE
from scipy.ndimage import gaussian_filter1d


# Simple plots for spike data

def plot_waveforms(analyzer, unit_ids: list, max_spikes: int = 100, save_path: Path | None = None):
    """Plots individual and average waveforms of specified units across identified channels.

    Args:
        analyzer (SortingAnalyzer): Spike sorting analyzer from spikeinterface, can either be used from running the sort function or loading directly from a save.
        unit_ids (list): List of unit ids to plot.
        max_spikes (int, optional): Maximum number of single spike waveforms to plot, best to set this number low, especially when plotting multiple units. Defaults to 100.
        save_path (Path, optional): Determines whether plot is saved as '.png'. Figure will be saved in the recording directory folder 'unit_plots' Defaults to False.
    """
    
    # lazy correction if plotting one unit
    if not isinstance(unit_ids, list):
        unit_ids = [unit_ids]

    # plot unit waveforms using spikewidget function
    sw.plot_unit_waveforms(analyzer, unit_ids=unit_ids, max_spikes_per_unit=max_spikes)
    plt.suptitle('Unit Waveforms')
    plt.tight_layout()

    if save_path:
        plots_dir = Path(save_path) / 'unit_plots'
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / 'unit_waveforms.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()


def plot_autocorrelogram(sorter, unit_ids: list, save_path: Path | None = None):
    """Plots autocorrelogram for specified units.

    Args:
        sorter (SortingExtractor): Spikeinterface Sorting Extractor object. Get from either running sort, or loading from previous sorting.
        unit_ids (list): List of unit ids to plot.
        save_fig (bool, optional): Determines whether plot is saved as '.png'. Figure will be saved in the recording directory folder 'unit_plots' Defaults to False.
    """
    
    # lazy correction if plotting one unit
    if not isinstance(unit_ids,list):
        unit_ids = [unit_ids]

    # plot autocorrelograms using spikewidget function
    w = sw.plot_autocorrelograms(sorter, unit_ids=unit_ids)
    plt.suptitle('Unit Autocorrelograms')
    plt.tight_layout()

    if save_path:
        plots_dir = Path(save_path) / 'unit_plots'#Path(sorter.get_annotation('phy_folder')).parent.parent / 'unit_plots'
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / 'unit_autocorrelograms.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()

def plot_psth(rasters_df, unit_ids: list, movement_plot_params: dict | None = None, save_path: Path | None = None): #, save_fig = False):
    # lazy correction if plotting one unit
    if not isinstance(unit_ids,list):
        unit_ids = [unit_ids]
    
    # get movement plot params
    if movement_plot_params:
        # dictionary extraction if provided
        node = movement_plot_params['node']
        movement_event = movement_plot_params['movement_event']
        mpl_cmap = movement_plot_params['cmap']
    else:
        # if no movement_plot_params given then defaults to first node and movement event
        node = rasters_df['node'].unique()[0]
        movement_event = rasters_df['movement_event'].unique()[0]
        mpl_cmap = 'default'

    n = len(unit_ids)
    ncols = min(5, n)
    nrows = math.ceil(n / 5)
    if mpl_cmap == 'default':
        cmap = lambda i: 'black'
    else:
        cmap = plt.get_cmap(mpl_cmap, n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = np.array(axes).reshape(nrows,ncols)
    
    for i, uid in enumerate(unit_ids):
        rasters = raster_df.query("unit_id==@uid & node==@node & movement_event==@movement_event")
        raster_index = rasters.index[0] # correct for starting position in row of raster
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        for ii, row in rasters.iterrows():
            pos_ = ii-raster_index
            spks = row['spike_raster']
            ax.vlines(spks, pos_+0, pos_+1, color=cmap(i), lw=1)
        ax.axvline(0.0, linestyle='--', color='red', linewidth=0.75, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Movement Index')
        ax.set_title(f"Unit id: {uid}")
    for j in range(n, nrows*ncols):
        row = j // ncols
        col = j % ncols
        axes[row, col].axis('off')

    plt.suptitle(f'Spike rasters: {node} {movement_event} movement')
    plt.tight_layout()
    if save_path:
        plots_dir = Path(save_path) / 'unit_plots'
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / f'{node}_{movement_event}_{n}_units_psth.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()


def plot_movement_psth(alignment, sorter, unit_ids: list, movement_plot_params: dict | None = None, save_fig = False):
    # lazy correction if plotting one unit
    if not isinstance(unit_ids,list):
        unit_ids = [unit_ids]
    
    # get movement plot params
    if movement_plot_params:
        # dictionary extraction if provided
        pre_event = movement_plot_params['pre_event']
        post_event = movement_plot_params['post_event']
        node = movement_plot_params['node']
        movement_event = movement_plot_params['movement_event']
        mpl_cmap = movement_plot_params['cmap']
    else:
        # if no movement_plot_params given then defaults to first node and movement event
        pre_event = 0.5
        post_event = 0.5
        node = alignment['node'].unique()[0]
        movement_event = alignment['movement_event'].unique()[0]
        mpl_cmap = 'default'

    aligned_movements = alignment.query("node==@node & movement_event==@movement_event")['event_times_ts'].values
    n = len(unit_ids)
    ncols = min(5, n)
    nrows = math.ceil(n / 5)
    if mpl_cmap == 'default':
        cmap = lambda i: 'black'
    else:
        cmap = plt.get_cmap(mpl_cmap, n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = np.array(axes).reshape(nrows,ncols)
    aligned_spike_times = []
    for uid in unit_ids:
        spike_times = sorter.get_unit_spike_train_in_seconds(unit_id=uid)
        spike_rasters = []
        for am in aligned_movements:
            spikes_in_window = spike_times[(spike_times>(am-pre_event)) & (spike_times <=(am+post_event))]
            spike_rasters.append(spikes_in_window - am)
        aligned_spike_times.append({
            "unit_id": uid,
            "movement_event": movement_event,
            "node": node,
            "spike_rasters": [np.asarray(sr) for sr in spike_rasters]
        })
    for i, raster_data in enumerate(aligned_spike_times):
        raster = raster_data['spike_rasters']
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        for ii, spks in enumerate(raster):
            ax.vlines(spks, ii+0, ii+1, color=cmap(i), lw=1)
        ax.axvline(0.0, linestyle='--', color='red', linewidth=0.75, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trial')
        ax.set_title(f"Unit id: {raster_data['unit_id']}")
    for j in range(n, nrows*ncols):
        row = j // ncols
        col = j % ncols
        axes[row, col].axis('off')

    plt.suptitle(f'Spike rasters: {node} {movement_event} movement')
    plt.tight_layout()
    if save_fig:
        plots_dir = Path(sorter.get_annotation('phy_folder')).parent.parent / 'unit_plots'
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / f'{node}_{movement_event}_rasters.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()
    return aligned_spike_times



def plot_session_psth(unit_ids, sorting, dflist, frame_captures, stances, node='r_forepaw', epoch_loc='start', xlim_=[-0.5,0.5], ylim_=[0,100],bin_size=0.02, smooth_sigma=1.0, prune_trials=True,save_fig=None):
    """
    Plot peristimulus time histogram from session
    """
    fig, ax =plt.subplots(3,1, sharex=True, gridspec_kw={'height_ratios': [4, 1, 1]})
    spike_train_total = []
    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train_total.append(spike_train)
    spike_train_total = np.concatenate(spike_train_total)
    spike_train = spike_train_total/30000
    
    # Set lists
    spikes_to_store = []
    kin_to_store = []
    mirror_kin_to_store = []
    spks_per_trial_total = []
    all_counts = []
    bins = np.arange(xlim_[0], xlim_[1] + bin_size, bin_size)
    iii = 0

    # Get the mirrored node - will expand later to get all nodes
    if node == 'r_forepaw':
        mirror_node = 'l_forepaw'
    elif node == 'l_forepaw':
        mirror_node = 'r_forepaw'
    elif node == 'r_hindpaw':
        mirror_node = 'l_hindpaw'
    elif node == 'l_hindpaw':
        mirror_node = 'r_hindpaw'

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
        times_ = np.array(stances[trial_ids_sort[ii]][node][epoch_loc])
        aligned_spikes = spike_train - bt[bout_start_id]
        spks_per_trial = []

        # Get kinematics of node for corresponding trial
        movement = dflist[trial_ids_sort[ii]][node+'_Y'].to_numpy()
        # Get kinematics of mirror node for corresponding trial
        mirror_movement = dflist[trial_ids_sort[ii]][mirror_node+'_Y'].to_numpy()
        for i, tstart in enumerate(times_):
            tstart_samp = tstart
            tstart = tstart/200.0
            spikes_in_window = aligned_spikes[(aligned_spikes>(tstart-0.5)) & (aligned_spikes <=(tstart+0.5))]
            spikes_to_store.append(spikes_in_window-tstart)
            spks_per_trial.append([i,spikes_in_window-tstart])

            # Store kinematics of node
            kin_to_store.append((movement[tstart_samp-100:tstart_samp+100]-movement[tstart_samp])*pixels_to_cm())
            # Store kinematics of mirrored node
            mirror_kin_to_store.append((mirror_movement[tstart_samp-100:tstart_samp+100]-movement[tstart_samp])*pixels_to_cm())
            counts, _ = np.histogram(spikes_in_window-tstart, bins=np.arange(xlim_[0],xlim_[1]+bin_size,bin_size))
            all_counts.append(counts)
        spks_per_trial_total.append(spks_per_trial)

    if prune_trials:
        kin_prune = []
        mirror_kin_prune = []
        spikes_prune = []

        for i, v in enumerate(kin_to_store):
            if len(v) == 200:
                if epoch_loc == 'start':
                    #if (np.mean(v[50:100]) < 0.5) & (np.mean(v[100:150])>1.5):
                    if (np.max(v[:100]) < 3.0) & (np.max(v[100:])<4.0) & (np.mean(v[100:])>1.5):
                        kin_prune.append(v)
                        mirror_kin_prune.append(mirror_kin_to_store[i])
                        spikes_prune.append(spikes_to_store[i])
                elif epoch_loc == 'end':
                    #if (np.mean(v[50:100]) < -0.5) & (np.mean(v[100:150]) > -0.5):
                    if (np.max(v[:100]) < 1.0) & (np.min(v[:100])>-8.0) & (np.max(v[100:]) < 5.0) & (np.mean(v[100:])>-0.5):
                        kin_prune.append(v)
                        mirror_kin_prune.append(mirror_kin_to_store[i])
                        spikes_prune.append(spikes_to_store[i])
                elif epoch_loc == 'max':
                    if (np.max(v[:100]) < 3.0) & (np.max(v[100:]) < 5.0) & (np.mean(v[100:]) > 1.5):
                        kin_prune.append(v)
                        mirror_kin_prune.append(mirror_kin_to_store[i])
                        spikes_prune.append(spikes_to_store[i])
        spikes_to_store = spikes_prune
        kin_to_store = kin_prune
        mirror_kin_to_store = mirror_kin_prune
    
    sorted_spikes = sorted(spikes_to_store,key=len, reverse=True)
    plt.title(f'{node} movement {epoch_loc}')

    for iii, x in enumerate(sorted_spikes):
        ax[0].vlines(x, iii + 0, iii + 1, color='black', lw=1)
    ax[0].axvline(0.0, linestyle='--', color='red', linewidth=0.75,alpha=0.5)
    ax[0].set_xlim(xlim_)
    ax[0].set_ylabel('Trial')
    if len(all_counts) > 0:
        all_counts = np.array(all_counts)
        firing_rate = (all_counts / bin_size)
        mean_rate = np.mean(firing_rate,axis=0)
        std_rate = np.std(firing_rate,axis=0)
        smoothed_rate = gaussian_filter1d(mean_rate, sigma=smooth_sigma)
        tbins = bins[:-1] + bin_size/2
        ax[1].plot(tbins,smoothed_rate,color='black')
        ax[1].axvline(0, linestyle='--',color='red', linewidth=0.75, alpha=0.5)
        ax[1].set_ylabel('Firing rate (spikes/s)')
    kin_ts = np.linspace(-0.5,0.5,200)
    ax[1].set_ylim(ylim_)
    for ki in kin_to_store:
        if len(ki) == 200:
            ax[2].plot(kin_ts,ki,color='black',alpha=0.5)
    ax[2].set_ylabel('Distance (cm)')
    ax[2].set_xlabel('Time (s)')
    ax[2].axvline(0, linestyle='--',color='red', linewidth=0.75, alpha=0.5)
    ax[2].set_ylim([-5.,5.])

    if save_fig:
        plt.savefig(save_fig+f'/{node}_movement-{epoch_loc}_unit_id{unit_ids[0]}_spikeraster.png')
        plt.savefig(save_fig+f'/{node}_movement-{epoch_loc}_unit_id{unit_ids[0]}_spikeraster.pdf')

    plt.show()
    return spikes_to_store, kin_to_store, mirror_kin_to_store