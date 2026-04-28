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



# Simple plots for spike data

def plot_waveforms(analyzer, unit_ids: list, max_spikes: int = 100, save_path: Path | None = None):
    """Plots individual and average waveforms of specified units across identified channels.

    Args:
        analyzer (SortingAnalyzer): Spike sorting analyzer from spikeinterface, can either be used from running the sort function or loading directly from a save.
        unit_ids (list): List of unit ids to plot.
        max_spikes (int, optional): Maximum number of single spike waveforms to plot, best to set this number low, especially when plotting multiple units. Defaults to 100.
        save_path (Path, optional): Determines whether plot is saved and to where. Figure will be saved in the save_path directory as a '.png'. Defaults to None.
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
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'unit_waveforms.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()


def plot_autocorrelogram(sorter, unit_ids: list, save_path: Path | None = None):
    """Plots autocorrelogram for specified units.

    Args:
        sorter (SortingExtractor): Spikeinterface Sorting Extractor object. Get from either running sort, or loading from previous sorting.
        unit_ids (list): List of unit ids to plot.
        save_path (Path, optional): Determines whether plot is saved and to where. Figure will be saved in the save_path directory as a '.png'. Defaults to None.
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
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'unit_autocorrelograms.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()

# def plot_movement_psth(rasters_df: pd.DataFrame, unit_ids: list, movement_plot_params: dict | None = None, save_path: Path | None = None):
#     """Plot psth with respect to movement events.

#     Args:
#         rasters_df (pd.DataFrame): Dataframe containing spike rasters aligned to movement.
#         unit_ids (list): List of unit ids to plot
#         movement_plot_params (dict | None, optional): Dictionary containing parameters for plotting requires:
#             {
#             'node': str, body part (node). This will be based on the nodes using during markerless pose estimation.
#             'movement_event': str, type of movement (e.g. 'start', 'end', 'max'). This will be based on the movement events you extract.
#             'mpl_cmap': str, matplotlib colormap to use.
#             }
        
#             Defaults to None, which defaults to plotting rasters with respect to the first rows node and event type, in black.

#         save_path (Path, optional): Determines whether plot is saved and to where. Figure will be saved in the save_path directory as a '.png'. Defaults to None.
#     """
#     # lazy correction if plotting one unit
#     if not isinstance(unit_ids,list):
#         unit_ids = [unit_ids]
    
#     # get movement plot params
#     if movement_plot_params:
#         # dictionary extraction if provided
#         node = movement_plot_params['node']
#         movement_event = movement_plot_params['movement_event']
#         mpl_cmap = movement_plot_params['cmap']
#     else:
#         # if no movement_plot_params given then defaults to first node and movement event
#         node = rasters_df['node'].unique()[0]
#         movement_event = rasters_df['movement_event'].unique()[0]
#         mpl_cmap = 'default'

#     n = len(unit_ids)
#     ncols = min(5, n)
#     nrows = math.ceil(n / 5)
#     if mpl_cmap == 'default':
#         cmap = lambda i: 'black'
#     else:
#         cmap = plt.get_cmap(mpl_cmap, n)
#     fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
#     axes = np.array(axes).reshape(nrows,ncols)
    
#     for i, uid in enumerate(unit_ids):
#         rasters = rasters_df.query("unit_id==@uid & node==@node & movement_event==@movement_event")
#         raster_index = rasters.index[0] # correct for starting position in row of raster
#         row = i // ncols
#         col = i % ncols
#         ax = axes[row, col]
#         for ii, row in rasters.iterrows():
#             pos_ = ii-raster_index
#             spks = row['spike_raster']
#             ax.vlines(spks, pos_+0, pos_+1, color=cmap(i), lw=1)
#         ax.axvline(0.0, linestyle='--', color='red', linewidth=0.75, alpha=0.5)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Movement Index')
#         ax.set_title(f"Unit id: {uid}")
#     for j in range(n, nrows*ncols):
#         row = j // ncols
#         col = j % ncols
#         axes[row, col].axis('off')

#     plt.suptitle(f'Spike rasters: {node} {movement_event} movement')
#     plt.tight_layout()
#     if save_path:
#         plots_dir = Path(save_path) / 'unit_plots'
#         plots_dir.mkdir(parents=True, exist_ok=True)
#         plot_path = plots_dir / f'{node}_{movement_event}_{n}_units_psth.png'
#         plt.savefig(plot_path.as_posix()) # save figure to analyzer path

#     plt.show()

def plot_movement_psth(rasters_df: pd.DataFrame, unit_ids: list, movement_plot_params: dict | None = None, save_path: Path | None = None):
    """Plot psth with respect to movement events.

    Args:
        rasters_df (pd.DataFrame): Dataframe containing spike rasters aligned to movement.
        unit_ids (list): List of unit ids to plot
        movement_plot_params (dict | None, optional): Dictionary containing parameters for plotting requires:
            {
            'node': str, body part (node). This will be based on the nodes using during markerless pose estimation.
            'movement_event': str, type of movement (e.g. 'start', 'end', 'max'). This will be based on the movement events you extract.
            'mpl_cmap': str, matplotlib colormap to use.
            'bin_size': Bin size in seconds for psth (default is 0.05)
            }
        
            Defaults to None, which defaults to plotting rasters with respect to the first rows node and event type, in black.

        save_path (Path, optional): Determines whether plot is saved and to where. Figure will be saved in the save_path directory as a '.png'. Defaults to None.
    """
    # lazy correction if plotting one unit
    if not isinstance(unit_ids,list):
        unit_ids = [unit_ids]
    
    # get movement plot params
    if movement_plot_params:
        # dictionary extraction if provided
        node = movement_plot_params['node']
        movement_event = movement_plot_params['movement_event']
        bin_size = movement_plot_params['bin_size']
        mpl_cmap = movement_plot_params['cmap']
    else:
        # if no movement_plot_params given then defaults to first node and movement event
        node = rasters_df['node'].unique()[0]
        movement_event = rasters_df['movement_event'].unique()[0]
        bin_size = 0.05
        mpl_cmap = 'default'

    n = len(unit_ids)
    ncols = min(5, n)
    n_unit_rows = math.ceil(n / ncols)
    if mpl_cmap == 'default':
        cmap = lambda i: 'black'
    else:
        cmap = plt.get_cmap(mpl_cmap, n)

    fig, axes = plt.subplots(
        n_unit_rows * 2,
        ncols,
        figsize=(3 * ncols, 4 * n_unit_rows),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1] * n_unit_rows},
    )

    axes = np.array(axes).reshape(n_unit_rows * 2, ncols)
    
    for i, uid in enumerate(unit_ids):
        rasters = rasters_df.query("unit_id==@uid & node==@node & movement_event==@movement_event")
        #raster_index = rasters.index[0] # correct for starting position in row of raster
        unit_row = i // ncols
        col = i % ncols
        raster_ax = axes[unit_row * 2, col]
        psth_ax = axes[unit_row * 2 + 1, col]
        #ax = axes[row, col]
        if rasters.empty:
            raster_ax.set_title(f"Unit id: {uid} (no data)")
            psth_ax.axis("off")
            continue
        all_spikes = []
        for trial_idx, (_, trial_row) in enumerate(rasters.iterrows()):
            #pos_ = ii-raster_index
            spks = np.asarray(trial_row['spike_raster'])
            all_spikes.extend(spks)
            raster_ax.vlines(
                spks,
                trial_idx,
                trial_idx+1,
                color=cmap(i),
                lw=1
            )
            #ax.vlines(spks, pos_+0, pos_+1, color=cmap(i), lw=1)
        all_spikes = np.asarray(all_spikes)
        if len(all_spikes) > 0:
            #t_min = all_spikes.min()
            #t_max = all_spikes.max()
            t_min = -0.5
            t_max = 0.5
            bins = np.arange(t_min, t_max + bin_size, bin_size)
            counts, edges = np.histogram(all_spikes, bins = bins)

            # firing rate = spikes / trial / second
            firing_rate = counts / len(rasters) / bin_size
            bin_centers = edges[:-1] + bin_size / 2
            psth_ax.plot(bin_centers, firing_rate, color = cmap(i), lw=1)
            psth_ax.fill_between(bin_centers, firing_rate, alpha=0.3, color=cmap(i))
        
        raster_ax.axvline(0.0, linestyle="--", color="red", linewidth=0.75, alpha=0.5)
        psth_ax.axvline(0.0, linestyle="--", color="red", linewidth=0.75, alpha=0.5)

        raster_ax.set_ylabel("Movement Index")
        psth_ax.set_ylabel("Hz")
        psth_ax.set_xlabel("Time (s)")

        raster_ax.set_title(f"Unit id: {uid}")

    # Turn off unused subplot pairs
    for j in range(n, n_unit_rows * ncols):
        unit_row = j // ncols
        col = j % ncols
        axes[unit_row * 2, col].axis("off")
        axes[unit_row * 2 + 1, col].axis("off")

    plt.suptitle(f"{node} {movement_event} movement: raster and psth")
    plt.tight_layout()
    if save_path:
        plots_dir = Path(save_path) / 'unit_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / f'{node}_{movement_event}_{n}_units_psth.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()