"""Tests that markerless confidence scores propagate through the pose pipeline.

Covers:
- ``load_file`` extracting point/instance/tracking scores into the dataframe
- ``create_df`` storing scores (and staying backward compatible without them)
- per-node confidence flowing extract_movements -> pad_movements ->
  build_movement_dataset into the ``confidence`` variable of the dataset
"""

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from neurokinematics.pose.io import load_file, create_df
from neurokinematics.pose.movement_events import extract_movements
from neurokinematics.pose.features import pad_movements, build_movement_dataset


NODES = ["paw", "nose", "tail"]


# ── fixtures / helpers ────────────────────────────────────────────────────────

def _write_sleap_h5(path, n_frames=120, n_inst=1):
    """Write a SLEAP-analysis-style HDF5 with tracks + the three score arrays.

    Orientation mirrors the loader's ``.T`` convention: ``tracks`` indexes as
    (frame, node, coord, instance) after transpose, so it is written as
    (instance, coord, node, frame).
    """
    n_nodes = len(NODES)
    tracks = np.random.rand(n_inst, 2, n_nodes, n_frames).astype("float32")
    # node 0 gets a known ramp so we can lock the score->node orientation
    point_scores = np.zeros((n_inst, n_nodes, n_frames), dtype="float32")
    for nd in range(n_nodes):
        point_scores[0, nd, :] = np.linspace(0.1 * (nd + 1), 1.0, n_frames)
    instance_scores = np.full((n_inst, n_frames), 0.5, dtype="float32")
    tracking_scores = np.full((n_inst, n_frames), 0.8, dtype="float32")
    with h5py.File(path, "w") as h:
        h["tracks"] = tracks
        h["point_scores"] = point_scores
        h["instance_scores"] = instance_scores
        h["tracking_scores"] = tracking_scores
        h["node_names"] = np.array([n.encode() for n in NODES])
    return n_frames


@pytest.fixture
def sleap_file(tmp_path):
    # filename must be <id>_<type>_<date>_<trial>.h5 for the loader's parser
    path = tmp_path / "M1_climb_20260101_T01.h5"
    n = _write_sleap_h5(path)
    return path, n


def _pose_df_with_movements(n=700, with_scores=True):
    """Pose dataframe with clear movement bouts (sharp ramps clear find_peaks)."""
    def sig():
        y = np.zeros(n)
        for c in (100, 300, 500):
            y[c:c + 20] = np.arange(20) * 30
            y[c + 20:c + 40] = np.arange(20)[::-1] * 30
        return y

    data = {}
    for nd in ["paw", "nose"]:
        data[f"{nd}_X"] = np.random.randn(n)
        data[f"{nd}_Y"] = sig() + np.random.randn(n) * 0.1
        if with_scores:
            data[f"{nd}_score"] = np.clip(np.random.rand(n) * 0.1 + 0.88, 0, 1)
    data.update({"Trial": 1, "Date": pd.Timestamp("2026-01-01"),
                 "Id": "M1", "Type": "climb", "SampleRate": 200.0})
    return pd.DataFrame(data)


# ── load_file / create_df ─────────────────────────────────────────────────────

def test_load_file_includes_scores(sleap_file):
    path, n = sleap_file
    df = load_file(str(path))

    for nd in NODES:
        assert f"{nd}_X" in df.columns
        assert f"{nd}_Y" in df.columns
        assert f"{nd}_score" in df.columns
    assert "instance_score" in df.columns
    assert "tracking_score" in df.columns
    assert len(df) == n

    # orientation locked: node 0 (paw) score is the ramp we wrote
    np.testing.assert_allclose(df["paw_score"].to_numpy(),
                               np.linspace(0.1, 1.0, n), atol=1e-5)
    np.testing.assert_allclose(df["instance_score"].to_numpy(), 0.5, atol=1e-5)
    np.testing.assert_allclose(df["tracking_score"].to_numpy(), 0.8, atol=1e-5)


def test_create_df_without_scores_is_backward_compatible():
    n = 10
    locs = np.random.rand(n, 2, 2, 1)          # (frame, node, coord, instance)
    df = create_df(locs, {"a": 0, "b": 1})
    assert {"a_X", "a_Y", "b_X", "b_Y", "frame_id"} <= set(df.columns)
    assert not any(c.endswith("_score") for c in df.columns)
    assert "instance_score" not in df.columns


def test_create_df_with_scores_adds_columns():
    n = 10
    locs = np.random.rand(n, 2, 2, 1)
    point_scores = np.random.rand(n, 2, 1)     # (frame, node, instance)
    df = create_df(locs, {"a": 0, "b": 1},
                   point_scores=point_scores,
                   instance_scores=np.full((n, 1), 0.9),
                   tracking_scores=np.full((n, 1), 0.7))
    assert "a_score" in df.columns and "b_score" in df.columns
    assert "instance_score" in df.columns and "tracking_score" in df.columns
    np.testing.assert_allclose(df["a_score"].to_numpy(), point_scores[:, 0, 0])


# ── movement pipeline ─────────────────────────────────────────────────────────

def test_confidence_propagates_to_movement_dataset():
    df = _pose_df_with_movements(with_scores=True)
    _, movement_list = extract_movements(df, ["paw", "nose"])

    assert len(movement_list) > 0
    assert movement_list[0]["score_array"] is not None
    assert movement_list[0]["score_array"].shape[1] == 2     # n_nodes

    padded, mov_list, valid, lengths, padded_scores = pad_movements(movement_list)
    assert padded_scores is not None
    assert padded_scores.shape == (len(movement_list), valid.shape[1], 2)

    ds = build_movement_dataset(padded, mov_list, valid, lengths,
                                padded_scores=padded_scores)
    assert "confidence" in ds.data_vars
    assert tuple(ds["confidence"].dims) == ("event", "time", "node")
    assert "confidence" in ds.attrs["features"]


def test_no_scores_means_no_confidence_var():
    df = _pose_df_with_movements(with_scores=False)
    _, movement_list = extract_movements(df, ["paw", "nose"])
    assert movement_list[0]["score_array"] is None

    padded, mov_list, valid, lengths, padded_scores = pad_movements(movement_list)
    assert padded_scores is None

    ds = build_movement_dataset(padded, mov_list, valid, lengths,
                                padded_scores=padded_scores)
    assert "confidence" not in ds.data_vars
    assert "confidence" not in ds.attrs["features"]
