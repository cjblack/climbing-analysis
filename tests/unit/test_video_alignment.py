from pathlib import Path
import numpy as np
from neurokinematics.multi_modal.alignment import detect_camera_on
from neurokinematics.io import load_config

def test_video_alignment():
    """Test for extracting video alignment with ephys data
    """
    
    data_path = Path(__file__).resolve().parent.parent.parent / 'examples' / 'sample_data' / 'continuous_event_test_data.npz'
    data = np.load(data_path)
    signal = data['signal']
    sample_rate = data['sample_rate']
    fps = data['camera_fps']

    # pad signal otherwise transforms will miss at least one event due to short time series
    signal_pad = np.pad(signal, 1000, mode='minimum')
    cfg = load_config('camera_alignment_test_cfg.yaml', config_type='multimodal')
    detection_settings = cfg['detection_settings']
    # no saving data, hence 'placeholder' string
    _, _, frame_captures, _ = detect_camera_on(signal_pad, sample_rate, detection_settings, save_path=None)
    
    assert len(frame_captures) == 1 # only one "video" recording should have been identified
    assert len(frame_captures[0]) == 3 # only three frame captures should have been identified

