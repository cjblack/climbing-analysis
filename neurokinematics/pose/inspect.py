"""Helpers for the GUI pose-quality inspector.

Loads the *raw* SLEAP arrays straight from disk and runs the same cleaning
functions the pipeline uses, so the inspector can preview raw-vs-processed for
any threshold / gap setting without committing anything. Kept separate from the
batch ``process_sleap`` path so it can operate on a single file at a time.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from neurokinematics.pose.preprocessing.cleaning import (
    remove_low_confidence, remove_high_velocity,
)


def find_pose_files(data_path, file_format: str = "h5") -> list:
    """Return the sorted list of pose files in *data_path* (e.g. SLEAP .h5)."""
    if not data_path:
        return []
    p = Path(data_path)
    if not p.exists():
        return []
    return sorted(glob.glob((p / f"*{file_format}").as_posix()))


def load_sleap_arrays(h5_path):
    """Read raw positions, confidence scores, and node names from a SLEAP .h5.

    Mirrors the read in ``neurokinematics.pose.io`` so the array layout matches
    what the pipeline's cleaning functions expect:
        locations : (frames, nodes, 2)   x/y per node
        scores    : (frames, nodes, 1)   per-node point score
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        locations = f["tracks"][:].T
        scores    = f["point_scores"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
    return np.asarray(locations, dtype=float), np.asarray(scores, dtype=float), node_names


def preview_cleaning(locations, scores, *, thresh: float = 0.7,
                     max_gap=None, remove_velocity: bool = False,
                     vel_thresh: float = 20.0):
    """Apply the pipeline's cleaning to raw arrays for a what-if preview.

    Returns ``(processed_locations, stats)`` where stats summarises how much data
    the chosen settings touch. ``locations``/``scores`` are not modified.
    """
    proc = remove_low_confidence(locations, scores, thresh=thresh, max_gap=max_gap)
    if remove_velocity:
        proc = remove_high_velocity(proc, thresh=vel_thresh, max_gap=max_gap)

    below = scores[:, :, 0] < thresh
    nan_after = np.isnan(proc[:, :, 0])
    n = below.size or 1
    stats = {
        "frac_below":  float(below.sum()) / n,        # flagged low-confidence
        "frac_missing": float(nan_after.sum()) / n,   # still NaN after cleaning
        "n_frames":    int(locations.shape[0]),
        "n_nodes":     int(locations.shape[1]),
    }
    return proc, stats


def node_below_fraction(scores, node_idx: int, thresh: float) -> float:
    """Fraction of frames below *thresh* for a single node."""
    s = scores[:, node_idx, 0]
    if s.size == 0:
        return 0.0
    return float((s < thresh).sum()) / s.size
