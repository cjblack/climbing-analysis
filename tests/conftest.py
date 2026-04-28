import pandas as pd
import numpy as np
import pytest

def create_test_pose_df(n_frames=12, trial=0):
    """Creates test pose data

    Args:
        n_frames (int, optional): _description_. Defaults to 12.
        trial (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return pd.DataFrame({
        "Unnamed: 0": np.arange(n_frames),
        "node1_X": np.random.randn(n_frames),
        "node1_Y": np.random.randn(n_frames),
        "node2_X": np.random.randn(n_frames),
        "node2_Y": np.random.randn(n_frames),
        "frame_id": np.arange(n_frames),
        "Type": ['start'] * n_frames,
        'Date': ['2025-01-01'] * n_frames,
        'Trial': [trial] * n_frames
    })

def create_test_movement_events_df(n_events = 2, n_trials=1):
    """Creates test movement events data

    Args:
        n_events (int, optional): _description_. Defaults to 2.
        n_trials (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    movement_events = []
    for i in range(n_trials):
        movement_events.append(
            pd.DataFrame.from_dict(
                {
                    'node1':{
                        'start': list(np.arange(n_events)), 
                        'end': list(np.arange(n_events)+5)
                        },
                    'node2':{
                        'start': list(np.arange(n_events)+2),
                        'end': list(np.arange(n_events)+7)
                    },
                    'trial': i,
                    'date': '2025-01-01'
                }
            )
        )
    return pd.concat(movement_events)

def create_frame_captures_df(n_frames = 20):
    """Creates test frame captures data

    Args:
        n_frames (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    return pd.DataFrame({
        "video_index": np.zeros(n_frames, dtype=int),
        "frame_id": np.arange(n_frames),
        "sample_index": np.arange(10000,10000+(n_frames*150),150)
    })

@pytest.fixture
def generate_alignment_inputs(tmp_path):
    """Create test data and store temporarily

    Args:
        tmp_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    events_dir = tmp_path / 'events'
    pose_dir = tmp_path / 'pose'
    alignment_dir = tmp_path / 'alignment'

    events_dir.mkdir()
    pose_dir.mkdir()
    alignment_dir.mkdir()

    dirs = {
        "events": events_dir,
        "pose": pose_dir,
        "alignment": alignment_dir
    }

    # create test data

    # movement_df
    movement_df = create_test_movement_events_df()
    movement_df.to_pickle(events_dir / 'movement_events.pkl')

    # pose_df
    pose_df = create_test_pose_df()
    pose_df.to_csv(pose_dir / 'pose_data.csv', index=False)

    # frame_capture_df
    frame_capture_df = create_frame_captures_df(n_frames=20)
    frame_capture_df.to_csv(alignment_dir / 'video_alignment.csv', index=False)

    return dirs