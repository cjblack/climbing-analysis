from pathlib import Path

from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import label
import numpy as np
import pandas as pd


from neurokinematics.io import load_config, load_csv, load_pickle
from neurokinematics.ephys.io import get_continuous
from neurokinematics.utils import check_and_make_directory # this is quite redundant



def bandpass_filter(signal, fs: float, freq: float, bandwidth=20, order=3):
    """Simple bandpass filter. Used for extracting frame times from analog data.

    Args:
        signal (np.memmap): Analog time series for channel with events
        fs (float): Sampling rate of analog signal
        freq (float): Frequency of camera (i.e. frame rate).
        bandwidth (float, optional): Bandwidth to set low and high pass. Defaults to 20.
        order (int, optional): Filter order. Defaults to 3.

    Returns:
        ndarray: Filtered signal.
    """
    nyq = 0.5 * fs
    low = (freq - bandwidth / 2) / nyq
    high = (freq + bandwidth / 2) / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_camera_on(signal, fs, directory:str, detection_settings:dict, save_events=True):
    """Extracts frame times on analog channel based on strobe method.

    Args:
        signal (np.memmap): Analog time series for channel with events
        fs (float): Sampling rate of analog signal
        directory (str): Directory path for recording
        detection_settings (dict): Dictionary of detection settings, either created manually or obtained from load_config
        save_events (bool, optional): Boolean to determine if resulting frame captures should be stored. Defaults to True.

    Raises:
        ValueError: Check on appending frame captures. If signal is not long enough, or no strobe signal is present, error will be raised.

    Returns:
        bouts (list): List of tuples containing start and end times of video recording
        envelope (ndarray): Envelope of filtered analog signal
        frame_captures (list): List of stored frame captures
        frame_map (dataframe): Same as frame captures but stored as a dataframe. Much more conveient for long-term storage and use
    """
    # FLIR camera generates square wave when frames are taken
    # get detection settings
    frame_rate = detection_settings['fps']
    threshold_ratio = detection_settings['threshold_ratio']
    min_bout_duration = detection_settings['minimum_bout_duration']
    bandwidth = detection_settings['bandwidth']
    
    # Bandpass filter around the square wave frequency
    filtered = bandpass_filter(signal, fs, frame_rate, bandwidth=bandwidth)

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

def get_camera_events(directory: str, camera_cfg_file: str):
    """High-level call for getting frame times (camera events) from an ephys recording. Requires strobe method to be used for tracking frame captures on an analog channel.

    Args:
        directory (str): Path for the recording to extract camera events from
        camera_cfg_file (str): Config file to use for extracting frames. Config files should be stored in the 'configs/multimodal_cfg' folder in this projects root directory

    Returns:
        event_data (np.memmap): Memory mapped analog signal used for recording camera strobe
        ts (np.memmap): Timestamps for analog channel
        bouts (list): List of tuples containing start and end times for each video recorded during ephys session
        frame_captures (list): List of stored frame captures
        continuous:
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

def align_movements_to_ephys(directory: str, movement_events: list = ['start', 'end', 'max'], fs: float = 30000., fps: float = 200.): # frame_captures_df, movements_df, pose_df,
    """Aligns movement events to ephys timestamps and saves results in `pose/events` folder.

    Args:
        directory (str): Root directory of ephys data
        movement_events (list, optional): List of movement events. Defaults to ['start', 'end', 'max'].
        fs (float, optional): Sampling rate of ephys acquisition. Defaults to 30000..
        fps (float, optional): Frame rate of camera. Defaults to 200..

    Returns:
        records_df (pd.DataFrame): Dataframe containing times of movement events translated to ephys samples/timestamps
    """
    
    # create paths
    movement_df_path = Path(directory) / 'pose' / 'movement_events.pkl'
    pose_df_path = Path(directory) / 'pose' / 'pose_data.csv'
    frame_captures_df_path = Path(directory) / 'events' / 'video_alignment.csv'
    event_alignment_df_path = Path(directory) / 'events' / 'movement_event_alignment.csv'
    
    # load into dataframes
    movements_df = load_pickle(movement_df_path, method='pandas')
    pose_df = load_csv(pose_df_path, pkg_format='pandas')
    frame_captures_df = load_csv(frame_captures_df_path, pkg_format='pandas')

    trials_array = movements_df['trial'].unique()
    nodes = movements_df.keys().drop(['date', 'trial'])
    #resolve_id = lambda x, y: x[np.argmin(np.abs(x-y))]
    records = []
    for t in trials_array:
        frames_in_trial = frame_captures_df.query('video_index==@t')['sample_index'].values
        movements_during_trial = movements_df.query('trial==@t')
        no_frame_captures = pose_df.query('Trial==@t').__len__() # number of recorded frames
        frame_capture_start_id = len(frames_in_trial) - no_frame_captures # start id of pose data in analog frame events
        for node in nodes:
            node_movements = movements_during_trial[node]
            for me in movement_events:
                node_movement_events = node_movements[me]
                for i in node_movement_events:
                    aligned_sample_samp = frames_in_trial[frame_capture_start_id+i]
                    aligned_sample_ts = aligned_sample_samp / fs

                    records.append({
                        "trial": t,
                        "node": node,
                        "movement_event": me,
                        "frame_ids": i,
                        "event_times_ts": aligned_sample_ts,
                        "event_times_samples": aligned_sample_samp
                    })
    records_df = pd.DataFrame.from_records(records)
    records_df.to_csv(event_alignment_df_path)

    return records_df
