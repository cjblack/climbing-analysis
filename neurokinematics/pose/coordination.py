"""Module for computing coordination metrics.

This module is designed for analyses that examine coordination between node pairs from markerless pose estimation.

Currently, this module support the phase offset metric, but will be expanded in the future to incorporate temporal overlap metrics and stability measures.
"""


import dask
import numpy as np
import pandas as pd
from neurokinematics.pose.calibration import pixels_to_cm

def compute_phase_offset_pairs(pose_df: pd.DataFrame, stance_df: pd.DataFrame, node_pairs: list):
    """Compute phase offset between node pairs using swing/reach initiation.

    Args:
        pose_df (pd.DataFrame): Dataframe containing pose data from trial/session
        stance_df (pd.DataFrame): Dataframe containing movement event data from corresponding trial/session markerless pose data (saved as `movement_events.pkl`)
        node_pairs (list): List of tuples containing node pairs to compute phase offset between.

    Returns:
        phase_offset (dict): Dictionary containing the phase offset information for each coordinated movement between node pairs.

    Example:
        >>> poff = compute_phase_offset_pairs(
        ...     pose_df = pose_df,
        ...     stance_df = stance_df,
        ...     node_pairs = [('node1', 'node2'), ('node3', 'node4')]
        ...     )
    """

    # organise pose
    px_cm = pixels_to_cm()
    pose_group = pose_df.sort_values(['Date', 'Trial', 'frame_id']).groupby(['Date', 'Trial'])
    phase_offset = dict()
    for npair in node_pairs:
        # create dictionary to store data * need to make this more efficient
        phase_offset[npair] = dict()
        phase_offset[npair]['id'] = []
        phase_offset[npair]['date'] = []
        phase_offset[npair]['trial'] = []
        phase_offset[npair]['values'] = []
        phase_offset[npair]['locs'] = []
        phase_offset[npair]['movement_ratio'] = []
        phase_offset[npair]['max_speed'] = []

        for df_id, ((date_, trial_), df) in enumerate(pose_group):
            phase_offsets = [] # create an empty list to fill with all phase offset values
            movement_ratio = []
            max_speed = []
            locs = []

            stances = stance_df.query(f'date=="{date_}" & trial=={trial_}')

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
                        locs.append([loc_start,loc_end, theta_2, df_id])
            if phase_offsets:
                phase_offset[npair]['values'].append(phase_offsets) # append new list of phase offset values
                phase_offset[npair]['movement_ratio'].append(movement_ratio)
                phase_offset[npair]['max_speed'].append(max_speed)
                phase_offset[npair]['locs'].append(locs) # append new list of locations for phase offsets
                phase_offset[npair]['id'].append(df['Id'].min()) # append corresponding subject id
                phase_offset[npair]['date'].append(df['Date'].min()) # append corresponding file date
                phase_offset[npair]['trial'].append(df['Trial'].min())
    return phase_offset
