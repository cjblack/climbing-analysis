import h5py
import numpy as np
from scipy.signal import decimate, savgol_filter
from climbing_analysis.models.glm import get_trial_data, make_kinematic_vector


def package_data(dflist, frame_captures, sorting, spike_ids):
    data_package = dict()
    data_length = len(dflist)
    for x in range(data_length):
        df = dflist[x]
        camera_frames = frame_captures[x]
        subj_id = df.attrs['Id']
        expt_date = df.attrs['Date']
        expt_trial = df.attrs['Trial']
        data_package[subj_id] = {expt_date:dict()}
        data_package[subj_id][expt_date] = {expt_trial:dict()}
        #data_package[subj_id][expt_date][expt_trial] = dict()
        start_idx = len(camera_frames) - len(df)
        start_ts = camera_frames[start_idx]/30000
        end_ts = camera_frames[-1]/30000
        

        kinematics = make_kinematic_vector(df, type='normal')
        spikes = []
        for sid in spike_ids:
            spike_train = sorting.get_unit_spike_train(unit_id)/30000
            spike_ts = spike_train[(spike_train >= start_ts) & (spike_train <= end_ts)]
            spike_ts = spike_ts - start_ts
            spikes.append(np.array(spike_ts))

        data_package[subj_id][expt_date][expt_trial] = {'data_types':
                                   {'behaviour': 
                                    {'data':kinematics, 'type': 'kinematics'}, 
                                    'ephys': {'data':spikes, 'type': 'spike_times'}
                                    }
                                    }
    return data_package

def create_session_file(data):
    # fname
    f = h5py.File(fname,"a")
    for data in dataset:
        gname = data['name']
        data_types = data['data_types']
        grp = f.create_group(gname)
        for key, val in data_types.items():
            dset = grp.create_dataset(key, key.shape, dtype=key.dtype)

def resample_data(kinematics, spike_ts, bin_width=0.2,fps=0.005):
    K,N = kinematics.shape
    max_time = N*fps
    start_time = 0
    end_time = np.ceil(max_time / bin_width) * bin_width
    bins = make_time_edges(start_time, end_time, bin_width)
    spikes_binned = bin_spikes(spike_ts, bins)
    factor = int(bin_width/fps)
    #kinematics_trimmed = kinematics[:, :kinematics.shape[1] - kinematics.shape[1] % factor]
    #kinematics_downsampled = kinematics_trimmed.reshape(K, -1, factor).mean(axis=2).T
    kinematics_downsampled  = decimate(kinematics,q=40,axis=1, ftype='fir',zero_phase=True)
    kinematics_downsampled = kinematics_downsampled.T
    min_time = min(spikes_binned.shape[0], kinematics_downsampled.shape[0])
    spikes_binned = spikes_binned[:min_time]
    kinematics = kinematics_downsampled[:min_time, :]
    return kinematics, spikes_binned, bin_width