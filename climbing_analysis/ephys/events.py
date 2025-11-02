import matplotlib.pyplot as plt
from open_ephys.analysis import Session
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import label, gaussian_filter1d


import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch

def lfp_filter(data, fs, band=(0.1, 100), notch_freq=50.0, notch_Q=30.0):
    """
    Bandpass + notch filter for LFP data.

    Parameters
    ----------
    data : 1D array
        LFP time series.
    fs : float
        Sampling rate (Hz).
    band : tuple (low, high)
        Bandpass range in Hz (default: 0.1–100 Hz).
    notch_freq : float
        Notch frequency in Hz (default: 50 Hz).
    notch_Q : float
        Quality factor of notch filter (default: 30).

    Returns
    -------
    filtered : 1D array
        Filtered LFP signal.
    """

    nyq = 0.5 * fs

    # --- Bandpass filter (SOS = stable) ---
    low, high = band[0] / nyq, band[1] / nyq
    sos = butter(N=4, Wn=[low, high], btype='bandpass', output='sos')
    bandpassed = sosfiltfilt(sos, data)

    # --- Notch filter ---
    b_notch, a_notch = iirnotch(w0=notch_freq / nyq, Q=notch_Q)
    filtered = sosfiltfilt([[b_notch[0], b_notch[1], b_notch[2], 1.0, a_notch[1], a_notch[2]]], bandpassed)

    return filtered
def bandpass_filter(signal, fs, freq, bandwidth=20, order=3):
    nyq = 0.5 * fs
    low = (freq - bandwidth / 2) / nyq
    high = (freq + bandwidth / 2) / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def detect_camera_on(signal, fs, frame_rate=200, threshold_ratio=0.3, min_bout_duration=0.1):
    # FLIR camera generates square wave when frames are taken

    # Bandpass filter around the square wave frequency
    filtered = bandpass_filter(signal, fs, frame_rate, bandwidth=40)

    # Envelope via Hilbert transform
    envelope = np.abs(hilbert(filtered))

    # Threshold to detect active (camera on) regions
    threshold = threshold_ratio * np.max(envelope)
    active = envelope > threshold

    # Label contiguous active regions
    labeled_array, num_features = label(active)

    bouts = []
    frame_captures = []
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        duration_sec = len(indices) / fs
        if duration_sec >= min_bout_duration:
            start_idx = indices[0]
            end_idx = indices[-1]
            bouts.append((start_idx, end_idx))
            #bouts.append((indices[0], indices[-1]))
            bout_signal = signal[start_idx:end_idx]
            high_thresh = 0.5*(np.max(bout_signal)+np.min(bout_signal))
            binary_ = bout_signal > high_thresh
            edges = np.where(np.diff(binary_.astype(int)) == 1)[0]+1
            frame_starts = edges+start_idx
            frame_captures.append(frame_starts)

    return bouts, envelope, frame_captures

def get_camera_events(directory, node_idx=1, rec_idx=0, event_channel= 67, threshold_ratio=0.3):
    print('Check analog channel for frames')
    session = Session(directory)
    continuous = session.recordnodes[node_idx].recordings[rec_idx].continuous[0]
    fs = continuous.metadata['sample_rate']
    event_data = continuous.samples[:,event_channel]
    ts = continuous.sample_numbers/fs

    bouts, envelope, frame_captures = detect_camera_on(event_data, fs)

    return event_data, ts, bouts, frame_captures, continuous



# def plot_spikes_basic_range(unit_ids, sorting, dflist, frame_captures, stances, node='r_forepaw', epoch_loc='start', xlim_=[-0.5,0.5], bin_size=0.02, smooth_sigma=1.0, save_fig=None):
#     fig, ax =plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
#     spike_train_total = []
#     for unit_id in unit_ids:
#         spike_train = sorting.get_unit_spike_train(unit_id=unit_id)#, return_times=True)#clus_ids[3])
#         spike_train_total.append(spike_train)
#     spike_train_total = np.concatenate(spike_train_total)
#     spike_train = spike_train_total/30000#-recording.get_start_time()
#     spikes_to_store = []
#     spks_per_trial_total = []
#     all_counts = []
#     #bin_size = 20
#     bins = np.arange(xlim_[0], xlim_[1] + bin_size, bin_size)
#     iii = 0
#     trial_ids = []
#     for df in dflist:
#         tid = int(df.attrs['Trial'].split('T')[-1])
#         trial_ids.append(tid)
#     trial_ids_sort = np.argsort(trial_ids)
#     for ii in range(len(frame_captures)):#, bt in enumerate(frame_captures):
#         bt = frame_captures[ii]/30000
#         bout_start_id = len(bt) - (dflist[trial_ids_sort[ii]].__len__())
#         times_ = np.array(stances[trial_ids_sort[ii]][node][epoch_loc])
#         aligned_spikes = spike_train - bt[bout_start_id]
#         spks_per_trial = []
#         for i, tstart in enumerate(times_):
#             tstart = tstart/200.0
#             spikes_in_window = aligned_spikes[(aligned_spikes>(tstart-0.5)) & (aligned_spikes <=(tstart+0.5))]
#             spikes_to_store.append(spikes_in_window-tstart)
#             spks_per_trial.append([i,spikes_in_window-tstart])

#             counts, _ = np.histogram(spikes_in_window-tstart, bins=np.arange(xlim_[0],xlim_[1]+bin_size,bin_size))
#             all_counts.append(counts)
#         spks_per_trial_total.append(spks_per_trial)

#     sorted_spikes = sorted(spikes_to_store,key=len, reverse=True)
#     for iii, x in enumerate(sorted_spikes):
#         ax[0].vlines(x, iii + 0, iii + 1, color='black', lw=1)
#     ax[0].axvline(0.0, linestyle='--', color='red', linewidth=0.75,alpha=0.5)
#     ax[0].set_xlim(xlim_)
#     ax[0].set_ylabel('Trial')
#     if len(all_counts) > 0:
#         all_counts = np.array(all_counts)
#         firing_rate = (all_counts / bin_size)
#         mean_rate = np.mean(firing_rate,axis=0)
#         std_rate = np.std(firing_rate,axis=0)
#         smoothed_rate = gaussian_filter1d(mean_rate, sigma=smooth_sigma)
#         #smoothed_std = gaussian_filter1d(std_rate,sigma=smooth_sigma)
#         tbins = bins[:-1] + bin_size/2
#         #ax[1].bar(tbins,mean_rate, width=bin_size,color='purple',alpha=0.5)
#         ax[1].plot(tbins,smoothed_rate,color='black')
#         #ax[1].plot(tbins,smoothed_rate-smoothed_std,color='black',linewidth=0.75)
#         ax[1].axvline(0, linestyle='--',color='red', linewidth=0.75, alpha=0.5)
#         ax[1].set_ylabel('Firing rate (spikes/s)')
#         ax[1].set_xlabel('Time (s)')

#     plt.title(f'{node} movement {epoch_loc}')
#     if save_fig:
#         plt.savefig(save_fig+f'/{node}_movement-{epoch_loc}_unit_id{unit_ids[0]}_spikeraster.png')
#         plt.savefig(save_fig+f'/{node}_movement-{epoch_loc}_unit_id{unit_ids[0]}_spikeraster.pdf')

#     plt.show()
#     return spikes_to_store, spks_per_trial_total

