from open_ephys.analysis import Session
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import label
import numpy as np

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
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        duration_sec = len(indices) / fs
        if duration_sec >= min_bout_duration:
            bouts.append((indices[0], indices[-1]))

    return bouts, envelope

def get_camera_events(directory, node_idx, rec_idx, threshold_ratio=0.3):
    print('Check analog channel for frames')
    session = Session(directory)
    continuous = session.recordnodes[node_idx].recordings[rec_idx].continuous[0]
    fs = continuous.metadata['sample_rate']
    event_channel = continuous.samples
    ts = continuous.sample_numbers/fs

    bouts, _ = detect_camera_on(event_channel, fs)

    return event_channel, ts, bouts