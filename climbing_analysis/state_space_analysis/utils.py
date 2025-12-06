from pathlib import Path
import matplotlib.pyplot as plt
from spikeinterface.sorters import run_sorter
from spikeinterface import create_sorting_analyzer
from spikeinterface.exporters import export_to_phy
from spikeinterface.extractors import read_phy
from climbing_analysis.ephys.utils import *
from climbing_analysis.pose.utils import pixels_to_cm
from scipy.ndimage import gaussian_filter1d


def get_population_data(unit_ids, sorting, dflist, frame_captures, stances, node='r_forepaw', epoch_loc='start', xlim_=[-0.5,0.5], bin_size=0.01, smooth_sigma=5.0, prune_trials=True,save_fig=None):
    """
    Plot peristimulus time histogram from session
    """
    n_neurons = len(unit_ids)

    population_spike_trains = []
    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
        population_spike_trains.append(spike_train/30000)

    
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