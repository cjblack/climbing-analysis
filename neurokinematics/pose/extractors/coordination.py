"""Module for computing coordination metrics.

This module is designed for analyses that examine coordination between node pairs from markerless pose estimation.

Currently, this module support the phase offset metric, but will be expanded in the future to incorporate temporal overlap metrics and stability measures.
"""
from datetime import datetime
from pathlib import Path
import dask
import numpy as np
import pandas as pd

from neurokinematics.io import load_file
from neurokinematics.pose.calibration import pixels_to_cm
from neurokinematics.registry import register_extractor


@register_extractor('pose', 'phaseoffset')
def compute_phase_offset_pairs(data: dict, params: dict, save_path: str | Path | None = None):#node_pairs: list):
    """Compute phase offset between node pairs using swing/reach initiation.

    Args:
        data (dict): Dictionary containing either directory of stored data, or pd.DataFrame for both pose_df (Dataframe containing pose data from trial/session - pose_data.csv) and stance_df (Dataframe containing movement event data from corresponding trial/session markerless pose data - `movement_events.pkl`) data.
        params (dict): Dictionary of params, this function takes only 'node_pairs' key, containing a list of tuples with the nodes to compute phase offset values with.

    Returns:
        phase_offset (dict): Dictionary containing the phase offset information for each coordinated movement between node pairs.

    Example:
        >>> poff = compute_phase_offset_pairs(
        ...     data = data,
        ...     params = {'node_pairs': [('node1', 'node2'), ('node3', 'node4')] },
        ...     )
    """
    if 'dirs' in data.keys():
        pose_df = load_file(data['dirs']['pose'] / 'pose_data.csv')
        stance_df = load_file(data['dirs']['pose'] / 'movement_events.pkl')
    else:
        pose_df = data['pose_df']
        stance_df = data['stance_df']

    node_pairs = params['node_pairs']

    # datetime
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # organise pose
    px_cm = pixels_to_cm()
    pose_group = pose_df.sort_values(['Date', 'Trial', 'frame_id']).groupby(['Date', 'Trial'])
    #phase_offset = dict()
    rows = []
    for npair in node_pairs:
        # create dictionary to store data * need to make this more efficient
        # phase_offset[npair] = dict()
        # phase_offset[npair]['id'] = []
        # phase_offset[npair]['date'] = []
        # phase_offset[npair]['trial'] = []
        # phase_offset[npair]['values'] = []
        # phase_offset[npair]['locs'] = []
        # phase_offset[npair]['movement_ratio'] = []
        # phase_offset[npair]['max_speed'] = []

        for df_id, ((date_, trial_), df) in enumerate(pose_group):
            phase_offsets = [] # create an empty list to fill with all phase offset values
            movement_ratio = []
            max_speed = []
            locs = []
            loc_starts = []
            loc_ends = []
            start_id = []
            ref_node = []
            comp_node = []
            date_ = pd.to_datetime(date_) # convert to proper dtype to avoid future issues with pandas query
            stances = stance_df.query(f'date==@date_ & trial==@trial_')

            no_stances = min([len(stances[npair[0]]['start']), len(stances[npair[1]]['start'])]) # get number of stances from each node in pair

            # Calculate phase offsets between pairs
            for i in range(len(stances[npair[1]]['start'])-1): # Compare first node in pair to second node stride
                for x in stances[npair[0]]['start']: # evaluate each movement start for first node in pair
                    if (x < stances[npair[1]]['start'][i+1]) & (x > stances[npair[1]]['start'][i]): # check that node position is moving within window
                        theta = (x - stances[npair[1]]['start'][i])/(stances[npair[1]]['start'][i+1]-stances[npair[1]]['start'][i]) # calculate phase offset as in Nirody et al., 2021
                        end_ratio = (x - stances[npair[1]]['start'][i]) / ((x - stances[npair[1]]['start'][i])+(stances[npair[1]]['end'][i]-stances[npair[1]]['start'][i]))
                        loc_start = df[npair[1]+'_Y'].values[stances[npair[1]]['start'][i]] # this gives the location of the comparison paw in Y
                        loc_end = df[npair[1]+'_Y'].values[stances[npair[1]]['start'][i+1]] # get the end location
                        theta_2 = x
                        max_speed.append(np.max(np.diff(df[npair[1]+'_Y'].values[stances[npair[1]]['start'][i]:stances[npair[1]]['start'][i+1]]*px_cm)))
                        phase_offsets.append(theta) # append phase offset
                        movement_ratio.append(end_ratio)
                        loc_starts.append(loc_start)
                        loc_ends.append(loc_end)
                        start_id.append(x)
                        ref_node.append(npair[1])
                        comp_node.append(npair[0])
                        locs.append([loc_start, loc_end, theta_2, df_id]) # append locations
            # Run the same loop but comparing second node to first in pair...
            for i in range(len(stances[npair[0]]['start'])-1):
                for x in stances[npair[1]]['start']:
                    if (x < stances[npair[0]]['start'][i+1]) & (x > stances[npair[0]]['start'][i]):
                        theta = (x - stances[npair[0]]['start'][i])/(stances[npair[0]]['start'][i+1]-stances[npair[0]]['start'][i])
                        end_ratio = (x - stances[npair[0]]['start'][i]) / ((x - stances[npair[0]]['start'][i])+(stances[npair[0]]['end'][i]-stances[npair[0]]['start'][i]))
                        loc_start = df[npair[0] + '_Y'].values[stances[npair[0]]['start'][i]]
                        loc_end = df[npair[0] + '_Y'].values[stances[npair[0]]['start'][i + 1]]
                        theta_2 = x
                        max_speed.append(np.max(np.diff(df[npair[0]+'_Y'].values[stances[npair[0]]['start'][i]:stances[npair[0]]['start'][i+1]]*px_cm)))
                        phase_offsets.append(theta)
                        movement_ratio.append(end_ratio)
                        loc_starts.append(loc_start)
                        loc_ends.append(loc_end)
                        start_id.append(x)
                        ref_node.append(npair[0])
                        comp_node.append(npair[1])
                        locs.append([loc_start,loc_end, theta_2, df_id])
            rows.append(
                {
                    'node_pair': npair[0]+'__coord__'+npair[1],
                    'phase_cycle': np.array(phase_offsets),
                    'phase_rad': np.pi*2*np.array(phase_offsets),
                    'phase_deg': np.rad2deg(np.pi*2*np.array(phase_offsets)),
                    'start_loc': np.array(loc_starts),
                    'end_loc': np.array(loc_ends),
                    'frame_id': np.array(start_id),
                    'reference_node': np.array(ref_node),
                    'comparison_node': np.array(comp_node),
                    'id': df['Id'].min(),
                    'date': df['Date'].min(),
                    'trial': df['Trial'].min()
                 }
            )
    poff_df = pd.DataFrame(rows)
    phase_offset_df = poff_df.explode(['phase_cycle', 'phase_rad', 'phase_deg', 'start_loc', 'end_loc', 'frame_id', 'reference_node', 'comparison_node'])
            # if phase_offsets:
            #     phase_offset[npair]['values'].append(phase_offsets) # append new list of phase offset values
            #     phase_offset[npair]['movement_ratio'].append(movement_ratio)
            #     phase_offset[npair]['max_speed'].append(max_speed)
            #     phase_offset[npair]['locs'].append(locs) # append new list of locations for phase offsets
            #     phase_offset[npair]['id'].append(df['Id'].min()) # append corresponding subject id
            #     phase_offset[npair]['date'].append(df['Date'].min()) # append corresponding file date
            #     phase_offset[npair]['trial'].append(df['Trial'].min())
    if save_path is not None:
        phase_offset_df.to_parquet(save_path / f'phase_offset_{date_str}.parquet')
    elif 'dirs' in data.keys():
        save_path = data['dirs']['results'] / 'coordination'
        save_path.mkdir(parents=True, exist_ok=True)
        phase_offset_df.to_parquet(save_path / f'phase_offset_{date_str}.parquet')

    file_outputs = {'phase_offset': {'path': save_path, 'file_type': 'parquet', 'attrs': {'node_pairs': node_pairs}}}

    return phase_offset_df, file_outputs
