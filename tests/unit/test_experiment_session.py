from pathlib import Path
import pytest

from neurokinematics.data.session import ExperimentSession

def test_experiment_session_create_and_reload(tmp_path):
    """Test for creating and reloading ExperimentSession object.

    Args:
        tmp_path (Path): Temporary path for pytest.
    """

    # create dummy dirs
    ephys_path = tmp_path / 'ephys_data_path'
    pose_path = tmp_path / 'pose_data_path'
    output_path = tmp_path / 'output_root_path'

    ephys_path.mkdir()
    pose_path.mkdir()
    output_path.mkdir()

    # create session

    session = ExperimentSession(
        session_id = "pytest_session",
        ephys_data_path = ephys_path,
        pose_data_path = pose_path,
        output_root_path = output_path,
        cfg = 'demo_session.yaml'
    )

    # base checks
    assert session.session_path.exists() # check session path created
    assert (session.session_path / "session_config.yaml").exists() # check session config created during instantiation

    # reload session
    reloaded = ExperimentSession.from_existing(session.session_path)

    # consistency
    assert reloaded.session_id == session.session_id
    assert reloaded.ephys_data_path == session.ephys_data_path
    assert reloaded.pose_data_path == session.pose_data_path
    assert reloaded.session_path == session.session_path

def test_invalid_paths(tmp_path):
    """Test for catching non-existent file paths.

    Args:
        tmp_path (Path): Temporary path for pytest.
    """
    with pytest.raises(FileNotFoundError):
        ExperimentSession(
            session_id = "pytest_session_invalid",
            ephys_data_path = tmp_path / 'ephys_data_path',
            pose_data_path = tmp_path / 'pose_data_path',
            output_root_path = tmp_path,
            cfg = 'demo_session.yaml'
        )