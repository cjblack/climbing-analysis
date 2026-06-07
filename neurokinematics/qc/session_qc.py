"""Quality-control checks for neurokinematics data objects.

A *check* is a function ``(session) -> QCResult | None`` that inspects one
session and returns a structured result (``None`` to skip itself entirely).
Every check is wrapped so a failure inside it becomes a FAIL result rather than
crashing the whole report. Checks return ``NA`` when the relevant data simply
has not been produced yet, so unprocessed sessions are reported as
"nothing to check" rather than failing.

Run with :func:`run_qc`, which dispatches on the object type (session / subject
/ group) and returns a :class:`QCReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


# ── Result model ────────────────────────────────────────────────────────────

class QCStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NA   = "na"      # not applicable — data not produced yet

    @property
    def severity(self) -> int:
        return {"na": 0, "pass": 1, "warn": 2, "fail": 3}[self.value]


def _worst(statuses) -> QCStatus:
    """Worst status in an iterable; NA only if everything is NA/empty."""
    statuses = list(statuses)
    non_na = [s for s in statuses if s is not QCStatus.NA]
    if not non_na:
        return QCStatus.NA
    return max(non_na, key=lambda s: s.severity)


@dataclass
class QCResult:
    name: str
    status: QCStatus
    message: str
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QCResult":
        return cls(
            name=d.get("name", ""),
            status=QCStatus(d.get("status", "na")),
            message=d.get("message", ""),
            metrics=d.get("metrics", {}) or {},
        )


@dataclass
class QCReport:
    target: str                       # e.g. session_id / subject_id / group_id
    level: str                        # 'session' | 'subject' | 'group'
    results: list = field(default_factory=list)        # list[QCResult]
    children: list = field(default_factory=list)       # list[QCReport]

    @property
    def status(self) -> QCStatus:
        own = [r.status for r in self.results]
        kids = [c.status for c in self.children]
        return _worst(own + kids)

    def counts(self) -> dict:
        """Tally of own + descendant result statuses (excludes report nodes)."""
        tally = {s.value: 0 for s in QCStatus}
        for r in self.results:
            tally[r.status.value] += 1
        for c in self.children:
            for k, v in c.counts().items():
                tally[k] += v
        return tally

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "level": self.level,
            "status": self.status.value,
            "counts": self.counts(),
            "results": [r.to_dict() for r in self.results],
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QCReport":
        """Rebuild a report from a saved to_dict() (status/counts are derived)."""
        return cls(
            target=d.get("target", ""),
            level=d.get("level", ""),
            results=[QCResult.from_dict(r) for r in d.get("results", [])],
            children=[cls.from_dict(c) for c in d.get("children", [])],
        )


# ── Thresholds (module-level so they're easy to tune / document) ─────────────

POSE_NAN_WARN_FRAC        = 0.20   # warn if >20% of pose coords are NaN
LIKELIHOOD_THRESHOLD      = 0.90   # "low-confidence" pose point cutoff
LIKELIHOOD_LOW_WARN_FRAC  = 0.20   # warn if >20% of points below threshold
FRAME_MISMATCH_WARN_FRAC  = 0.01   # warn if pose vs alignment differ by >1%


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dirs(session) -> dict:
    return getattr(session, "dirs", {}) or {}


def _resolve_output(session, info: dict):
    """Resolve a session_outputs entry to an existing Path, or None."""
    if not isinstance(info, dict):
        return None
    stored = info.get("path")
    if not stored:
        return None
    p = Path(stored)
    if p.exists():
        return p
    sess_dir = getattr(session, "session_path", None)
    if sess_dir:
        p2 = Path(sess_dir) / stored
        if p2.exists():
            return p2
    return None


def _pose_file(session):
    folder = _dirs(session).get("pose")
    if not folder:
        return None
    p = Path(folder) / "pose_data.csv"
    return p if p.exists() else None


# ── Individual checks ─────────────────────────────────────────────────────────

def check_recorded_outputs(session) -> Optional[QCResult]:
    """Every output recorded in session_outputs.yaml is present on disk."""
    outputs = getattr(session, "session_outputs", {}) or {}
    if not outputs:
        return QCResult("Recorded outputs", QCStatus.NA,
                        "No outputs recorded yet.")
    missing = [name for name, info in outputs.items()
               if _resolve_output(session, info) is None]
    total = len(outputs)
    if not missing:
        return QCResult("Recorded outputs", QCStatus.PASS,
                        f"All {total} recorded output(s) present on disk.",
                        {"total": total, "missing": 0})
    return QCResult(
        "Recorded outputs", QCStatus.FAIL,
        f"{len(missing)}/{total} recorded output(s) missing: "
        f"{', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}",
        {"total": total, "missing": len(missing), "missing_names": missing},
    )


def check_pose_integrity(session) -> Optional[QCResult]:
    """pose_data.csv loads, and NaN fraction across coordinates is acceptable."""
    pose = _pose_file(session)
    if pose is None:
        return QCResult("Pose integrity", QCStatus.NA,
                        "No pose_data.csv — pose not processed.")
    import pandas as pd
    df = pd.read_csv(pose)
    n_frames = len(df)
    # columns are written as "<node>_X" / "<node>_Y" by create_df
    coord_cols = [c for c in df.columns if c.lower().endswith(("_x", "_y"))]
    if not coord_cols:
        return QCResult("Pose integrity", QCStatus.WARN,
                        f"pose_data.csv has {n_frames} rows but no _X/_Y columns.",
                        {"n_frames": n_frames})
    nan_frac = float(df[coord_cols].isna().mean().mean())
    metrics = {"n_frames": n_frames, "n_nodes": len(coord_cols) // 2,
               "nan_fraction": round(nan_frac, 4)}
    if n_frames == 0:
        return QCResult("Pose integrity", QCStatus.FAIL,
                        "pose_data.csv is empty.", metrics)
    if nan_frac > POSE_NAN_WARN_FRAC:
        return QCResult("Pose integrity", QCStatus.WARN,
                        f"{nan_frac:.1%} of pose coordinates are NaN "
                        f"(>{POSE_NAN_WARN_FRAC:.0%}).", metrics)
    return QCResult("Pose integrity", QCStatus.PASS,
                    f"{n_frames} frames, {metrics['n_nodes']} nodes, "
                    f"{nan_frac:.1%} NaN.", metrics)


def check_pose_likelihood(session) -> Optional[QCResult]:
    """Fraction of tracked points below the confidence/likelihood threshold.

    Prefers the per-node ``<node>_score`` point-score columns written from the
    markerless tracker; falls back to DLC-style likelihood columns.
    """
    pose = _pose_file(session)
    if pose is None:
        return None   # covered by NA in pose_integrity; skip to avoid noise
    import pandas as pd
    df = pd.read_csv(pose)
    like_cols = [c for c in df.columns
                 if c.endswith("_score") and c not in ("instance_score", "tracking_score")]
    if not like_cols:   # DLC-style fallback
        like_cols = [c for c in df.columns
                     if "likelihood" in c.lower() or c.lower().startswith("p_")]
    if not like_cols:
        return QCResult("Pose confidence", QCStatus.NA,
                        "No per-node score/likelihood columns in pose_data.csv.")
    vals = df[like_cols].to_numpy().ravel()
    import numpy as np
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return QCResult("Pose confidence", QCStatus.NA, "No likelihood values.")
    low_frac = float((vals < LIKELIHOOD_THRESHOLD).mean())
    metrics = {"low_conf_fraction": round(low_frac, 4),
               "threshold": LIKELIHOOD_THRESHOLD,
               "median_likelihood": round(float(np.median(vals)), 4)}
    if low_frac > LIKELIHOOD_LOW_WARN_FRAC:
        return QCResult("Pose confidence", QCStatus.WARN,
                        f"{low_frac:.1%} of points below likelihood "
                        f"{LIKELIHOOD_THRESHOLD} (>{LIKELIHOOD_LOW_WARN_FRAC:.0%}).",
                        metrics)
    return QCResult("Pose confidence", QCStatus.PASS,
                    f"{low_frac:.1%} of points below likelihood "
                    f"{LIKELIHOOD_THRESHOLD}.", metrics)


def check_pose_alignment_consistency(session) -> Optional[QCResult]:
    """Pose frame count matches the video-alignment row count."""
    pose = _pose_file(session)
    align_dir = _dirs(session).get("alignment")
    if pose is None or not align_dir:
        return None
    align_file = Path(align_dir) / "video_alignment.csv"
    if not align_file.exists():
        return QCResult("Pose↔video alignment", QCStatus.NA,
                        "No video_alignment.csv — alignment not run.")
    import pandas as pd
    n_pose = len(pd.read_csv(pose))
    n_align = len(pd.read_csv(align_file))
    metrics = {"n_pose_frames": n_pose, "n_alignment_rows": n_align}
    if n_pose == 0 or n_align == 0:
        return QCResult("Pose↔video alignment", QCStatus.FAIL,
                        "Pose or alignment table is empty.", metrics)
    diff = abs(n_pose - n_align) / max(n_pose, n_align)
    metrics["mismatch_fraction"] = round(diff, 4)
    if diff > FRAME_MISMATCH_WARN_FRAC:
        return QCResult("Pose↔video alignment", QCStatus.WARN,
                        f"Frame counts differ by {diff:.1%} "
                        f"(pose={n_pose}, alignment={n_align}).", metrics)
    return QCResult("Pose↔video alignment", QCStatus.PASS,
                    f"Frame counts match (pose={n_pose}, alignment={n_align}).",
                    metrics)


def check_movement_features(session) -> Optional[QCResult]:
    """movement_features.zarr loads; judge per-node confidence (not padding).

    Note: the ``valid`` mask only marks real-vs-padding samples (events are
    padded to a common length), so a low valid fraction is expected and is
    *not* a defect — we use it to ignore padding, never to fail the check.
    """
    folder = _dirs(session).get("pose")
    if not folder:
        return None
    zarr_path = Path(folder) / "movement_features.zarr"
    if not zarr_path.exists():
        return QCResult("Movement features", QCStatus.NA,
                        "No movement_features.zarr.")
    from neurokinematics.io import load_zarr
    ds = load_zarr(zarr_path, method="xarray")
    data_vars = getattr(ds, "data_vars", {})
    metrics = {}
    try:
        metrics["n_events"] = int(ds.sizes.get("event", 0))
    except Exception:
        pass

    if "confidence" in data_vars:
        conf = ds["confidence"]
        if "valid" in data_vars:
            conf = conf.where(ds["valid"])          # ignore padding
        low = (conf < LIKELIHOOD_THRESHOLD).where(~conf.isnull())
        low_frac = float(low.mean(skipna=True))
        metrics["low_conf_fraction"] = round(low_frac, 4)
        metrics["mean_confidence"] = round(float(conf.mean(skipna=True)), 4)
        if low_frac > LIKELIHOOD_LOW_WARN_FRAC:
            return QCResult("Movement features", QCStatus.WARN,
                            f"{metrics.get('n_events', '?')} events; "
                            f"{low_frac:.1%} of tracked points below confidence "
                            f"{LIKELIHOOD_THRESHOLD}.", metrics)
        return QCResult("Movement features", QCStatus.PASS,
                        f"{metrics.get('n_events', '?')} events; mean confidence "
                        f"{metrics['mean_confidence']:.2f}.", metrics)

    return QCResult("Movement features", QCStatus.PASS,
                    f"{metrics.get('n_events', '?')} events "
                    "(no confidence stored).", metrics)


def check_spike_outputs(session) -> Optional[QCResult]:
    """Spike sorting analyzer is present."""
    folder = _dirs(session).get("spikes")
    if not folder:
        return None
    analyzer = Path(folder) / "sorting_analyzer"
    if analyzer.exists():
        return QCResult("Spike sorting", QCStatus.PASS,
                        "sorting_analyzer present.")
    return QCResult("Spike sorting", QCStatus.NA,
                    "No sorting_analyzer — spikes not sorted.")


def check_lfp_sampling_rate(session) -> Optional[QCResult]:
    """Preprocessed LFP carries a positive sampling rate in its attrs."""
    folder = _dirs(session).get("lfp")
    if not folder:
        return None
    lfp_dir = Path(folder)
    zarrs = list(lfp_dir.glob("*.zarr")) if lfp_dir.exists() else []
    if not zarrs:
        return QCResult("LFP sampling rate", QCStatus.NA,
                        "No LFP zarr — LFP not processed.")
    from neurokinematics.io import load_zarr
    _, attrs = load_zarr(zarrs[0])
    fs = (attrs or {}).get("fs")
    if fs and float(fs) > 0:
        return QCResult("LFP sampling rate", QCStatus.PASS,
                        f"fs = {float(fs):g} Hz.", {"fs": float(fs)})
    return QCResult("LFP sampling rate", QCStatus.FAIL,
                    "LFP zarr is missing a valid 'fs' attribute.",
                    {"fs": fs})


def check_spike_bad_channels(session) -> Optional[QCResult]:
    """Report bad-channel detection from spike preprocessing, if it ran.

    Reads ``bad_channels.json`` written by the preprocessing step. Status scales
    with the fraction of channels flagged bad; the message notes how many were
    actually removed before sorting (per the chosen policy).
    """
    folder = _dirs(session).get("spikes")
    if not folder:
        return None
    from neurokinematics.ephys.spikes.preprocessing import read_bad_channel_report
    report = read_bad_channel_report(folder)
    if not report:
        return QCResult("Spike bad channels", QCStatus.NA,
                        "Bad-channel detection has not been run.")

    n_total = int(report.get("n_channels", 0) or 0)
    n_bad   = int(report.get("n_bad", 0) or 0)
    n_removed = int(report.get("n_removed", 0) or 0)
    policy  = report.get("policy", "?")
    frac    = (n_bad / n_total) if n_total else 0.0
    metrics = {
        "n_channels": n_total, "n_bad": n_bad, "n_removed": n_removed,
        "policy": policy, "bad_fraction": round(frac, 4),
        "label_counts": report.get("label_counts", {}),
    }
    from neurokinematics.ephys.spikes.preprocessing import BAD_FRAC_WARN, BAD_FRAC_FAIL

    msg = (f"{n_bad}/{n_total} channel(s) flagged bad "
           f"({frac:.0%}); {n_removed} removed (policy: {policy}).")
    if n_bad == 0:
        return QCResult("Spike bad channels", QCStatus.PASS,
                        f"No bad channels detected across {n_total} channel(s).",
                        metrics)
    if frac >= BAD_FRAC_FAIL:
        return QCResult("Spike bad channels", QCStatus.FAIL, msg, metrics)
    if frac >= BAD_FRAC_WARN:
        return QCResult("Spike bad channels", QCStatus.WARN, msg, metrics)
    return QCResult("Spike bad channels", QCStatus.PASS, msg, metrics)


def check_session_id(session) -> Optional[QCResult]:
    """The session_id recorded in session_outputs.yaml matches the session.

    A mismatch usually means an outputs file was copied/moved between sessions,
    so it's a soft warning rather than a hard failure.
    """
    path = getattr(session, "session_outputs_path", None)
    if not path:
        sp = getattr(session, "session_path", None)
        path = Path(sp) / "session_outputs.yaml" if sp else None
    if not path or not Path(path).exists():
        return QCResult("Session ID", QCStatus.NA, "No session_outputs.yaml yet.")

    import yaml
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except Exception as e:
        return QCResult("Session ID", QCStatus.FAIL,
                        f"Could not read session_outputs.yaml: {e}")

    recorded = data.get("session_id") if isinstance(data, dict) else None
    actual   = getattr(session, "session_id", None)
    if recorded is None:
        return QCResult("Session ID", QCStatus.NA,
                        "No session_id recorded in outputs file.")
    if recorded == actual:
        return QCResult("Session ID", QCStatus.PASS,
                        f"Recorded session_id matches ('{actual}').",
                        {"session_id": actual})
    return QCResult("Session ID", QCStatus.WARN,
                    f"Recorded session_id '{recorded}' does not match "
                    f"session '{actual}'.",
                    {"recorded": recorded, "actual": actual})


SESSION_CHECKS: list = [
    check_session_id,
    check_recorded_outputs,
    check_pose_integrity,
    check_pose_likelihood,
    check_pose_alignment_consistency,
    check_movement_features,
    check_spike_outputs,
    check_spike_bad_channels,
    check_lfp_sampling_rate,
]


# ── Runners ───────────────────────────────────────────────────────────────────

def run_session_qc(session, checks: Optional[list] = None) -> QCReport:
    """Run every check against a session, never raising."""
    checks = checks if checks is not None else SESSION_CHECKS
    sess_id = getattr(session, "session_id", str(session))
    results: list = []
    for check in checks:
        try:
            res = check(session)
        except Exception as exc:  # a broken check must not sink the report
            res = QCResult(getattr(check, "__name__", "check"), QCStatus.FAIL,
                           f"Check raised {type(exc).__name__}: {exc}")
        if res is not None:
            results.append(res)
    return QCReport(target=str(sess_id), level="session", results=results)


def run_subject_qc(subject, checks: Optional[list] = None) -> QCReport:
    subj_id = getattr(subject, "subject_id", str(subject))
    sessions = getattr(subject, "sessions", None) or []
    children = [run_session_qc(s, checks) for s in sessions]
    return QCReport(target=str(subj_id), level="subject", children=children)


def run_group_qc(group, checks: Optional[list] = None) -> QCReport:
    grp_id = getattr(group, "group_id", str(group))
    subjects = getattr(group, "subjects", None) or []
    children = [run_subject_qc(s, checks) for s in subjects]
    return QCReport(target=str(grp_id), level="group", children=children)


def run_qc(obj, checks: Optional[list] = None) -> QCReport:
    """Dispatch to the right runner based on what *obj* looks like."""
    if hasattr(obj, "subjects"):
        return run_group_qc(obj, checks)
    if hasattr(obj, "sessions"):
        return run_subject_qc(obj, checks)
    return run_session_qc(obj, checks)
