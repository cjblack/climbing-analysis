import dask
import numpy as np
import pandas as pd
from climbing_analysis.pose.utils import pixels_to_cm

def compute_phase_offset_pairs(pose_df: pd.DataFrame, stance_df: pd.DataFrame, node_pairs: list):
    #GAP_END_PX = -598
    
    #pose_df = dd.read_csv(pose_file)
    #stance_df = dd.read_csv(stance_file)
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
            #counts_ = np.sum(df['tail_Y'].to_numpy() > GAP_END_PX)
            #if counts_ > 200:
            phase_offsets = [] # create an empty list to fill with all phase offset values
            movement_ratio = []
            max_speed = []
            locs = []
            # plot_phase_offsetV2
            stances = stance_df.query(f'date=="{date_}" & trial=={trial_}')
            #stances = climb_cycle_peaks(df) # get ids for starting and stopping movement
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
