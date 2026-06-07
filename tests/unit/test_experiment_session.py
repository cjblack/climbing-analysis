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

def test_no_paths_raises_value_error(tmp_path):
    """Both ephys_data_path and pose_data_path are None -> ValueError."""
    with pytest.raises(ValueError):
        ExperimentSession(
            session_id="pytest_no_paths",
            ephys_data_path=None,
            pose_data_path=None,
            output_root_path=tmp_path,
            cfg="demo_session.yaml",
        )


def test_str_representation(tmp_path):
    ephys_path = tmp_path / "ephys"
    pose_path = tmp_path / "pose"
    ephys_path.mkdir()
    pose_path.mkdir()

    session = ExperimentSession(
        session_id="str_test_session",
        ephys_data_path=ephys_path,
        pose_data_path=pose_path,
        output_root_path=tmp_path,
        cfg="demo_session.yaml",
    )
    result = str(session)
    assert "str_test_session" in result
    assert "Experiment Session Object" in result


def test_handle_existing_output_skip_when_missing(tmp_path):
    """_handle_existing_output returns True when the path doesn't exist yet."""
    ephys_path = tmp_path / "ephys"
    pose_path = tmp_path / "pose"
    ephys_path.mkdir()
    pose_path.mkdir()

    session = ExperimentSession(
        session_id="handle_missing",
        ephys_data_path=ephys_path,
        pose_data_path=pose_path,
        output_root_path=tmp_path,
        cfg="demo_session.yaml",
    )
    non_existent = tmp_path / "does_not_exist.csv"
    assert session._handle_existing_output(non_existent, "skip") is True


def test_handle_existing_output_skip_when_exists(tmp_path):
    """_handle_existing_output returns False when the path exists and mode is 'skip'."""
    ephys_path = tmp_path / "ephys"
    pose_path = tmp_path / "pose"
    ephys_path.mkdir()
    pose_path.mkdir()

    session = ExperimentSession(
        session_id="handle_skip",
        ephys_data_path=ephys_path,
        pose_data_path=pose_path,
        output_root_path=tmp_path,
        cfg="demo_session.yaml",
    )
    existing = tmp_path / "existing.csv"
    existing.touch()
    assert session._handle_existing_output(existing, "skip") is False


def test_handle_existing_output_overwrite(tmp_path):
    """_handle_existing_output returns True and removes the file when mode is 'overwrite'."""
    ephys_path = tmp_path / "ephys"
    pose_path = tmp_path / "pose"
    ephys_path.mkdir()
    pose_path.mkdir()

    session = ExperimentSession(
        session_id="handle_overwrite",
        ephys_data_path=ephys_path,
        pose_data_path=pose_path,
        output_root_path=tmp_path,
        cfg="demo_session.yaml",
    )
    existing = tmp_path / "to_overwrite.csv"
    existing.touch()
    result = session._handle_existing_output(existing, "overwrite")
    assert result is True
    assert not existing.exists()


def test_handle_existing_output_error_when_exists(tmp_path):
    """_handle_existing_output raises FileExistsError when mode is 'error' and path exists."""
    ephys_path = tmp_path / "ephys"
    pose_path = tmp_path / "pose"
    ephys_path.mkdir()
    pose_path.mkdir()

    session = ExperimentSession(
        session_id="handle_error",
        ephys_data_path=ephys_path,
        pose_data_path=pose_path,
        output_root_path=tmp_path,
        cfg="demo_session.yaml",
    )
    existing = tmp_path / "conflict.csv"
    existing.touch()
    with pytest.raises(FileExistsError):
        session._handle_existing_output(existing, "error")


def test_handle_existing_output_invalid_mode(tmp_path):
    """_handle_existing_output raises ValueError for an unrecognised mode."""
    ephys_path = tmp_path / "ephys"
    pose_path = tmp_path / "pose"
    ephys_path.mkdir()
    pose_path.mkdir()

    session = ExperimentSession(
        session_id="handle_invalid_mode",
        ephys_data_path=ephys_path,
        pose_data_path=pose_path,
        output_root_path=tmp_path,
        cfg="demo_session.yaml",
    )
    with pytest.raises(ValueError):
        session._handle_existing_output(tmp_path / "any.csv", "rerun")


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