from pathlib import Path
import matplotlib.pyplot as plt
import spikeinterface.full as si
import spikeinterface.widgets as sw
from climbing_analysis.ephys.utils import *
from climbing_analysis.pose.utils import pixels_to_cm
from scipy.ndimage import gaussian_filter1d



def plot_waveform(waveforms, unit_id):
    """
    Plot spike waveforms
    """
    wfs = waveforms.get_waveforms_one_unit(unit_id)
    for x in range(300):
        plt.plot(wfs[x, :, channel], color='purple', alpha=0.5)
    plt.plot(np.mean(wfs[:,:,channel],axis=0),color='black', linewidth=2)
    plt.show()

def plot_autocorrelogram(sorter,unit_ids):
    if not isinstance(unit_ids,list):
        unit_ids = [unit_ids]
    w = sw.plot_autocorrelograms(sorter, unit_ids=unit_ids)
    plt.show()
    return w

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