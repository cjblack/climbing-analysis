"""Tests for pre-movement window extension and labelling.

Covers:
- ``extract_movements`` extending the event window backwards while preserving
  the detected onset (``onset`` / ``n_pre``)
- ``build_movement_dataset`` exposing ``pre_movement`` / ``onset_idx`` / ``n_pre``
- the binned-dataset builders storing the ``pre_movement`` mask
- ``find_latest_glm_predictions`` locating the newest predictions store
"""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from neurokinematics.pose.movement_events import extract_movements
from neurokinematics.pose.features import pad_movements, build_movement_dataset
from neurokinematics.multi_modal.features import (
    build_aligned_spike_binned_dataset,
    build_resampled_movements_dataset,
)


FPS = 200.0


def _pose_df_with_movements(n=700, with_scores=False):
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
                 "Id": "M1", "Type": "climb", "SampleRate": FPS})
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# extract_movements — backward window extension
# ---------------------------------------------------------------------------

class TestExtractMovementsPreWindow:

    def test_zero_window_is_backward_compatible(self):
        df = _pose_df_with_movements()
        _, ml = extract_movements(df, ["paw", "nose"], pre_window_s=0.0)
        assert len(ml) > 0
        m = ml[0]
        assert m["n_pre"] == 0
        assert m["start"] == m["onset"]

    def test_window_extends_backwards_and_preserves_onset(self):
        df = _pose_df_with_movements()
        _, ml0 = extract_movements(df, ["paw", "nose"], pre_window_s=0.0)
        _, ml1 = extract_movements(df, ["paw", "nose"], pre_window_s=0.1)

        assert len(ml0) == len(ml1)
        pre_frames = int(round(0.1 * FPS))  # 20

        m0, m1 = ml0[0], ml1[0]
        # detection is unchanged: same onset
        assert m1["onset"] == m0["onset"] == m0["start"]
        # window now starts pre_frames earlier (first onset is well past frame 20)
        assert m1["n_pre"] == pre_frames
        assert m1["start"] == m1["onset"] - pre_frames
        # the stored trajectory is longer by exactly the pre-movement frames
        assert m1["node_array"].shape[0] == m0["node_array"].shape[0] + pre_frames

    def test_window_clamped_at_trial_start(self):
        df = _pose_df_with_movements()
        # 1.0 s = 200 frames, larger than the first onset (~frame 100)
        _, ml = extract_movements(df, ["paw", "nose"], pre_window_s=1.0)
        m = ml[0]
        assert m["start"] == 0
        assert m["n_pre"] == m["onset"]   # clamped: all available pre-frames used


# ---------------------------------------------------------------------------
# build_movement_dataset — pre-movement labelling
# ---------------------------------------------------------------------------

class TestBuildMovementDatasetPreMovement:

    def _build(self, pre_window_s):
        df = _pose_df_with_movements()
        _, ml = extract_movements(df, ["paw", "nose"], pre_window_s=pre_window_s)
        padded, mov_list, valid, lengths, scores = pad_movements(ml)
        ds = build_movement_dataset(padded, mov_list, valid, lengths,
                                    padded_scores=scores)
        return ds, mov_list

    def test_vars_present(self):
        ds, _ = self._build(0.1)
        for var in ("pre_movement", "onset_idx", "n_pre"):
            assert var in ds.data_vars
        assert tuple(ds["pre_movement"].dims) == ("event", "time")

    def test_pre_movement_mask_matches_n_pre(self):
        ds, _ = self._build(0.1)
        n_pre = int(ds["n_pre"].values[0])
        assert n_pre > 0
        pre = ds["pre_movement"].values[0]
        assert pre[:n_pre].all()          # first n_pre frames flagged pre-movement
        assert not pre[n_pre:].any()      # nothing after onset flagged

    def test_zero_window_has_no_pre_movement(self):
        ds, _ = self._build(0.0)
        assert not ds["pre_movement"].values.any()
        assert (ds["n_pre"].values == 0).all()


# ---------------------------------------------------------------------------
# binned-dataset builders — pre_movement storage
# ---------------------------------------------------------------------------

def _min_movement_ds(n_events):
    return xr.Dataset(
        data_vars={
            "id": (["event"], np.array(["M1"] * n_events)),
            "date": (["event"], np.array([np.datetime64("2026-01-01")] * n_events)),
            "reference_node": (["event"], np.array(["paw"] * n_events)),
            "trial": (["event"], np.arange(n_events)),
        },
        coords={"event": np.arange(n_events), "node": ["paw"], "coord": ["x", "y"]},
    )


def _movement_ds_with_features(n_events=3, n_time=8, n_nodes=4):
    """Movement dataset carrying a directional (coord) feature and a scalar
    (per-node, no coord) feature — mirrors 'velocity' + 'confidence'."""
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars={
            "velocity": (["event", "time", "node", "coord"],
                         rng.normal(size=(n_events, n_time, n_nodes, 2))),
            "confidence": (["event", "time", "node"],
                           rng.random((n_events, n_time, n_nodes))),
            "id": (["event"], np.array(["M1"] * n_events)),
            "date": (["event"], np.array([np.datetime64("2026-01-01")] * n_events)),
            "reference_node": (["event"], np.array(["n0"] * n_events)),
            "trial": (["event"], np.arange(n_events)),
        },
        coords={
            "event": np.arange(n_events),
            "time": np.arange(n_time),
            "node": [f"n{i}" for i in range(n_nodes)],
            "coord": ["x", "y"],
        },
    )


class TestScalarFeatureDims:
    """Regression: scalar-per-node features (speed/confidence) must NOT get a
    'coord' dim, while directional features must. Previously only 'speed' was
    special-cased, so 'confidence' raised a broadcast error during binning."""

    def test_confidence_has_no_coord_velocity_does(self):
        E, T, N = 3, 5, 4
        mds = _movement_ds_with_features(n_events=E, n_nodes=N)
        valid = np.ones((E, T), bool)
        movement_dict = {
            "velocity": np.zeros((E, T, N, 2)),
            "confidence": np.zeros((E, T, N)),
        }
        ds = build_resampled_movements_dataset(
            movement_dict, valid, mds, np.linspace(0, 0.1, T),
            {"pose_features": ["velocity", "confidence"]}, None)
        assert tuple(ds["velocity"].dims) == ("event", "time_bin", "node", "coord")
        assert tuple(ds["confidence"].dims) == ("event", "time_bin", "node")


class TestBuildersStorePreMovement:

    def test_spike_builder_stores_mask(self):
        E, T, U = 3, 5, 2
        mds = _min_movement_ds(E)
        valid = np.ones((E, T), bool)
        pm = np.zeros((E, T), bool); pm[:, :2] = True
        ds = build_aligned_spike_binned_dataset(
            np.zeros((E, T, U)), valid, mds, np.linspace(0, 0.1, T),
            np.arange(U), {}, None, pre_movement=pm)
        assert "pre_movement" in ds.data_vars
        np.testing.assert_array_equal(ds["pre_movement"].values, pm)

    def test_spike_builder_defaults_to_all_false(self):
        E, T, U = 3, 5, 2
        mds = _min_movement_ds(E)
        ds = build_aligned_spike_binned_dataset(
            np.zeros((E, T, U)), np.ones((E, T), bool), mds,
            np.linspace(0, 0.1, T), np.arange(U), {}, None)
        assert "pre_movement" in ds.data_vars
        assert not ds["pre_movement"].values.any()

    def test_pose_builder_stores_mask(self):
        E, T = 3, 5
        mds = _min_movement_ds(E)
        valid = np.ones((E, T), bool)
        pm = np.zeros((E, T), bool); pm[:, :2] = True
        ds = build_resampled_movements_dataset(
            {"position": np.zeros((E, T, 1, 2))}, valid, mds,
            np.linspace(0, 0.1, T), {"pose_features": ["position"]}, None,
            pre_movement=pm)
        assert "pre_movement" in ds.data_vars
        np.testing.assert_array_equal(ds["pre_movement"].values, pm)


# ---------------------------------------------------------------------------
# extract_movement_features — re-extraction entry point used by the GUI Bin step
# ---------------------------------------------------------------------------

class TestExtractMovementFeatures:

    def _cfg(self, pre_window_s):
        return {
            "group_cols": ["Date", "Trial"],
            "sort_cols": ["Date", "Trial"],
            "node_list": ["paw", "nose"],
            "pre_window_s": pre_window_s,
        }

    def test_writes_zarr_with_pre_movement(self, tmp_path):
        from neurokinematics.pose.preprocessing.base import extract_movement_features
        df = _pose_df_with_movements()
        ds, outs = extract_movement_features(df, self._cfg(0.1), tmp_path)
        assert (tmp_path / "movement_features.zarr").exists()
        assert "movement_features" in outs and "movement_events" in outs
        assert "pre_movement" in ds.data_vars
        assert ds["pre_movement"].values.any()       # pre_window>0 -> pre-movement frames
        assert (ds["n_pre"].values > 0).any()

    def test_zero_window_has_no_pre_movement(self, tmp_path):
        from neurokinematics.pose.preprocessing.base import extract_movement_features
        df = _pose_df_with_movements()
        ds, _ = extract_movement_features(df, self._cfg(0.0), tmp_path)
        assert not ds["pre_movement"].values.any()
        assert (ds["n_pre"].values == 0).all()


# ---------------------------------------------------------------------------
# find_latest_glm_predictions  (imports plot_viewer -> needs PySide6)
# ---------------------------------------------------------------------------

class TestFindLatestGlmPredictions:

    def _helper(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")
        from neurokinematics.gui.plot_viewer import find_latest_glm_predictions
        return find_latest_glm_predictions

    def test_none_when_no_glm_dir(self, tmp_path):
        find = self._helper()
        assert find(tmp_path / "models") is None
        assert find(None) is None

    def test_picks_most_recent(self, tmp_path):
        find = self._helper()
        glm = tmp_path / "models" / "glm" / "encoder"
        older = glm / "run_old" / "predictions.zarr"
        newer = glm / "run_new" / "predictions.zarr"
        older.mkdir(parents=True)
        newer.mkdir(parents=True)
        now = 1_900_000_000
        os.utime(older, (now - 100, now - 100))
        os.utime(newer, (now + 100, now + 100))
        assert find(tmp_path / "models").samefile(newer)

    def test_filters_by_glm_type(self, tmp_path):
        find = self._helper()
        enc = tmp_path / "models" / "glm" / "encoder" / "run" / "predictions.zarr"
        dec = tmp_path / "models" / "glm" / "decoder" / "run" / "predictions.zarr"
        enc.mkdir(parents=True)
        dec.mkdir(parents=True)
        assert find(tmp_path / "models", glm_type="decoder").samefile(dec)
        assert find(tmp_path / "models", glm_type="encoder").samefile(enc)
