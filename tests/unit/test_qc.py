"""Tests for the data QC layer (neurokinematics.qc).

Focuses on the behaviours that were recently corrected:
- pose integrity detects the real ``<node>_X`` / ``<node>_Y`` columns
- pose confidence reads the ``<node>_score`` point-score columns
- movement-features QC judges *confidence*, not the ``valid`` padding mask
plus the core report behaviour (recorded outputs, dispatch, JSON export).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neurokinematics.qc import run_qc, run_session_qc, QCStatus
from neurokinematics.qc.session_qc import (
    check_pose_integrity,
    check_pose_likelihood,
    check_movement_features,
    check_recorded_outputs,
    check_pose_alignment_consistency,
)

NODES = ["paw", "nose", "tail"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _pose_df(n=200, low_conf=False):
    data = {}
    for nd in NODES:
        data[f"{nd}_X"] = np.random.randn(n)
        data[f"{nd}_Y"] = np.random.randn(n)
        data[f"{nd}_score"] = (np.random.rand(n) * 0.5 if low_conf
                               else np.clip(np.random.rand(n) * 0.05 + 0.93, 0, 1))
    data["instance_score"] = 0.99
    data["tracking_score"] = 0.99
    data["frame_id"] = np.arange(n)
    return pd.DataFrame(data)


def _session(tmp_path, pose_df=None, outputs=None, alignment_rows=None):
    pose = tmp_path / "pose"; pose.mkdir()
    align = tmp_path / "alignment"; align.mkdir()
    if pose_df is not None:
        pose_df.to_csv(pose / "pose_data.csv", index=False)
    if alignment_rows is not None:
        pd.DataFrame({"frame": np.arange(alignment_rows)}).to_csv(
            align / "video_alignment.csv", index=False)
    return SimpleNamespace(
        session_id="s01", session_path=tmp_path,
        dirs={"pose": pose, "alignment": align,
              "spikes": tmp_path / "spikes", "lfp": tmp_path / "lfp"},
        session_outputs=outputs or {},
    )


def _make_movement_zarr(pose_dir, conf_value, n_ev=4, max_t=30):
    """Write a movement_features.zarr that is mostly padding (valid half False)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    n_nodes = len(NODES)
    pos = np.random.rand(n_ev, max_t, n_nodes, 2)
    valid = np.zeros((n_ev, max_t), dtype=bool)
    valid[:, :max_t // 2] = True                     # half the samples are padding
    conf = np.full((n_ev, max_t, n_nodes), conf_value, dtype=float)
    conf[~valid] = np.nan
    ds = xr.Dataset(
        {"position": (["event", "time", "node", "coord"], pos),
         "valid": (["event", "time"], valid),
         "confidence": (["event", "time", "node"], conf)},
        coords={"event": np.arange(n_ev), "time": np.arange(max_t),
                "node": NODES, "coord": ["x", "y"]},
    )
    ds.to_zarr(pose_dir / "movement_features.zarr", mode="w")


# ── pose integrity: column naming fix ─────────────────────────────────────────

def test_pose_integrity_detects_XY_columns(tmp_path):
    sess = _session(tmp_path, pose_df=_pose_df(n=150))
    res = check_pose_integrity(sess)
    assert res.status is QCStatus.PASS
    assert res.metrics["n_nodes"] == len(NODES)
    assert res.metrics["n_frames"] == 150


def test_pose_integrity_na_when_no_pose(tmp_path):
    sess = _session(tmp_path)        # no pose_data.csv written
    assert check_pose_integrity(sess).status is QCStatus.NA


# ── pose confidence: reads <node>_score ───────────────────────────────────────

def test_pose_confidence_pass_when_high(tmp_path):
    sess = _session(tmp_path, pose_df=_pose_df(low_conf=False))
    res = check_pose_likelihood(sess)
    assert res.status is QCStatus.PASS
    assert res.metrics["low_conf_fraction"] == 0.0


def test_pose_confidence_warn_when_low(tmp_path):
    sess = _session(tmp_path, pose_df=_pose_df(low_conf=True))
    res = check_pose_likelihood(sess)
    assert res.status is QCStatus.WARN
    assert res.metrics["low_conf_fraction"] > 0.2


# ── pose <-> alignment consistency ────────────────────────────────────────────

def test_alignment_pass_when_frames_match(tmp_path):
    sess = _session(tmp_path, pose_df=_pose_df(n=200), alignment_rows=200)
    assert check_pose_alignment_consistency(sess).status is QCStatus.PASS


def test_alignment_warn_when_frames_mismatch(tmp_path):
    sess = _session(tmp_path, pose_df=_pose_df(n=200), alignment_rows=150)
    assert check_pose_alignment_consistency(sess).status is QCStatus.WARN


# ── movement features: 'valid' is padding, not a defect ───────────────────────

def test_movement_features_padding_not_flagged(tmp_path):
    sess = _session(tmp_path, pose_df=_pose_df())
    _make_movement_zarr(sess.dirs["pose"], conf_value=0.97)   # high confidence
    res = check_movement_features(sess)
    # half the samples are padding (valid=False) — must NOT warn on that
    assert res.status is QCStatus.PASS
    assert res.metrics["n_events"] == 4


def test_movement_features_warn_on_low_confidence(tmp_path):
    sess = _session(tmp_path, pose_df=_pose_df())
    _make_movement_zarr(sess.dirs["pose"], conf_value=0.2)    # low confidence
    res = check_movement_features(sess)
    assert res.status is QCStatus.WARN
    assert res.metrics["low_conf_fraction"] > 0.2


# ── recorded outputs ──────────────────────────────────────────────────────────

def test_recorded_outputs_pass_when_present(tmp_path):
    sess = _session(tmp_path, pose_df=_pose_df(),
                    outputs={"pose_data": {"path": "pose/pose_data.csv"}})
    res = check_recorded_outputs(sess)
    assert res.status is QCStatus.PASS
    assert res.metrics["missing"] == 0


def test_recorded_outputs_fail_when_missing(tmp_path):
    sess = _session(tmp_path,
                    outputs={"lfp_data": {"path": "lfp/missing.zarr"}})
    res = check_recorded_outputs(sess)
    assert res.status is QCStatus.FAIL
    assert "lfp_data" in res.metrics["missing_names"]


# ── runner: dispatch, aggregation, JSON export ────────────────────────────────

def test_run_qc_dispatch_and_aggregation(tmp_path):
    sess = _session(tmp_path, pose_df=_pose_df(low_conf=True))
    subject = SimpleNamespace(subject_id="m1", sessions=[sess])
    group = SimpleNamespace(group_id="G1", subjects=[subject])

    sess_report = run_qc(sess)
    assert sess_report.level == "session"

    grp_report = run_qc(group)
    assert grp_report.level == "group"
    # low-confidence pose -> at least a WARN bubbles up to the group
    assert grp_report.status in (QCStatus.WARN, QCStatus.FAIL)
    # counts roll up across the hierarchy
    assert sum(grp_report.counts().values()) > 0


def test_report_is_json_serialisable(tmp_path):
    import json
    sess = _session(tmp_path, pose_df=_pose_df())
    payload = run_session_qc(sess).to_dict()
    json.dumps(payload)            # must not raise
    assert payload["level"] == "session"
    assert "results" in payload


def test_checks_never_raise_on_empty_session():
    """A bare session with nothing on disk should produce a report, not crash."""
    sess = SimpleNamespace(session_id="empty", session_path=None,
                           dirs={}, session_outputs={})
    report = run_session_qc(sess)
    assert report.level == "session"
    # everything is NA — nothing to check, but no failures from crashes
    assert report.status is QCStatus.NA
