from pathlib import Path
import matplotlib.pyplot as plt
from spikeinterface.sorters import run_sorter
from spikeinterface import create_sorting_analyzer
from spikeinterface.exporters import export_to_phy
from spikeinterface.extractors import read_phy
from climbing_analysis.ephys.utils import *
from climbing_analysis.pose.utils import pixels_to_cm, get_trial_order
from climbing_analysis.ephys.preprocessing.filters import filter_lfp
from scipy.ndimage import gaussian_filter1d
import mne




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
