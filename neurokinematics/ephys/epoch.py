import numpy as np


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