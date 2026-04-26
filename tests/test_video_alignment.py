from pathlib import Path
import numpy as np
from climbing_analysis.multi_modal.alignment import detect_camera_on

def test_video_alignment():
    """Test for extracting video alignment with ephys data
    """
    
    data_path = Path(__file__).resolve().parent.parent / 'examples' / 'sample_data' / 'continuous_event_test_data.npz'
    data = np.load(data_path)
    signal = data['signal']
    sample_rate = data['sample_rate']
    fps = data['camera_fps']

    # pad signal otherwise transforms will miss at least one event due to short time series
    signal_pad = np.pad(signal, 1000, mode='minimum')

    # no saving data, hence 'placeholder' string
    _, _, frame_captures, _ = detect_camera_on(signal_pad, sample_rate, 'placeholder', frame_rate=fps, min_bout_duration=0.0, save_events=False)
    
    assert len(frame_captures) == 1 # only one "video" recording should have been identified
    assert len(frame_captures[0]) == 3 # only three frame captures should have been identified

