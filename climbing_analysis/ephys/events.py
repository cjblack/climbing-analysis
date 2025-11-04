from open_ephys.analysis import Session
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt, iirnotch
from scipy.ndimage import label
import numpy as np


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
    """
    Bandpass filter for handling event data
    """
    nyq = 0.5 * fs
    low = (freq - bandwidth / 2) / nyq
    high = (freq + bandwidth / 2) / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_camera_on(signal, fs, frame_rate=200, threshold_ratio=0.3, min_bout_duration=0.1):
    """
    Extracts event times from analog channel recording camera strobe based on rising edge
    """
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
    """
    High level function for getting timestamps of camera frames taken during recording
    """
    print('Check analog channel for frames')
    session = Session(directory)
    continuous = session.recordnodes[node_idx].recordings[rec_idx].continuous[0]
    fs = continuous.metadata['sample_rate']
    event_data = continuous.samples[:,event_channel]
    ts = continuous.sample_numbers/fs

    bouts, envelope, frame_captures = detect_camera_on(event_data, fs)

    return event_data, ts, bouts, frame_captures, continuous
