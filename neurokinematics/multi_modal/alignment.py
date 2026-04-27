from open_ephys.analysis import Session
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import label
import numpy as np
import pandas as pd
import dask.dataframe as dd

from neurokinematics.io import load_config
from neurokinematics.ephys.io import get_continuous
from neurokinematics.utils import check_and_make_directory # this is quite redundant
#from neurokinematics.pose.utils import load_pickle, load_df_list


def bandpass_filter(signal, fs, freq, bandwidth=20, order=3):
    """
    Bandpass filter for handling event data
    """
    nyq = 0.5 * fs
    low = (freq - bandwidth / 2) / nyq
    high = (freq + bandwidth / 2) / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_camera_on(signal, fs, directory:str, detection_settings:dict, save_events=True):
    """
    Extracts event times from analog channel recording camera strobe based on rising edge
    """
    # FLIR camera generates square wave when frames are taken
    # get detection settings
    frame_rate = detection_settings['fps']
    threshold_ratio = detection_settings['threshold_ratio']
    min_bout_duration = detection_settings['minimum_bout_duration']
    
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
    frame_rows = []
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
            df = pd.DataFrame({
                "video_index": i-1,
                "frame_id": np.arange(len(frame_starts)),
                "sample_index": frame_starts

            })
            frame_rows.append(df)
    
    if not frame_rows:
        raise ValueError("frame_rows is empty; no frames were identified")
    frame_map = pd.concat(frame_rows, ignore_index=True)

    if save_events:
        events_directory = directory+'/events'
        check_and_make_directory(events_directory) # check that directory exists, if not then create it
        frame_map.to_csv(events_directory+"/video_alignment.csv", index=False)

    return bouts, envelope, frame_captures, frame_map

def get_camera_events(directory, camera_cfg_file: str):
    """
    High level function for getting timestamps of camera frames taken during recording
    """
    
    # load config
    cfg = load_config(camera_cfg_file, config_type='multimodal') # get camera config from multimodal_cfgs folder
    record_settings = cfg['acquisition_settings'] # settings for ephys acquisition channel used for alignment
    detection_settings = cfg['detection_settings'] # detection settings for identifying frames

    node_idx = record_settings['record_node']
    rec_idx = record_settings['record_index']
    event_channel = record_settings['event_channel']

    print('Check analog channel for frames')
    continuous, _ = get_continuous(directory, node_idx=node_idx, rec_idx=rec_idx)
    sample_rate = continuous.metadata.sample_rate #continuous.metadata['sample_rate']

    event_data = continuous.samples[:,event_channel]
    ts = continuous.sample_numbers/sample_rate

    bouts, envelope, frame_captures, frame_map = detect_camera_on(event_data, sample_rate, directory, detection_settings)

    return event_data, ts, bouts, frame_captures, continuous