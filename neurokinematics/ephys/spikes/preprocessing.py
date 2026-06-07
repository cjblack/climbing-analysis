"""Pre-sort preprocessing & bad-channel QC for spike sorting.

Runs the cheap, pre-sort part of the pipeline — read the recording, set the
probe, bandpass-filter, and detect bad channels with SpikeInterface — so a human
(or a fixed policy) can decide whether to drop bad channels before the expensive
sort. Detection results feed both a GUI review step and a QC artifact on disk.

The heavy ``sort()`` in ``sorting.py`` is left untouched apart from an optional
``bad_channels`` argument; this module owns everything up to that point.

Design notes
------------
* ``detect_bad_channels`` returns trace *snippets* (numpy) for plotting — these
  live in memory and are handed straight to the GUI, never serialised.
* ``write_bad_channel_report`` persists only a small JSON summary (ids, labels,
  counts, the applied policy and which channels were removed) — that's what the
  QC layer reads back.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurokinematics.ephys.utils import read_data, create_probe

# default snippet / filter parameters (kept here so they're easy to tune)
SNIPPET_SECONDS   = 1.0
FILTER_FREQ_MIN   = 300.0
FILTER_FREQ_MAX   = 6000.0
BAD_CHANNELS_FILE = "bad_channels.json"
# warn/fail thresholds for QC on the fraction of channels flagged bad
BAD_FRAC_WARN = 0.10
BAD_FRAC_FAIL = 0.30


def _detect_on_recording(recording, *, snippet_seconds: float = SNIPPET_SECONDS,
                         seed: int = 0) -> dict:
    """Filter *recording*, detect bad channels, and grab a snippet for plotting.

    Split out from :func:`detect_bad_channels` so it can be exercised on a
    synthetic SpikeInterface recording without real Open Ephys data on disk.
    """
    import spikeinterface.preprocessing as sp

    filt = sp.bandpass_filter(recording, freq_min=FILTER_FREQ_MIN,
                              freq_max=FILTER_FREQ_MAX)
    bad_ids, labels = sp.detect_bad_channels(filt, seed=seed)

    fs          = float(recording.get_sampling_frequency())
    channel_ids = [str(c) for c in recording.get_channel_ids()]
    label_list  = [str(x) for x in list(labels)]
    bad_list    = [str(b) for b in bad_ids]

    n_frames = max(1, int(snippet_seconds * fs))
    traces   = np.asarray(filt.get_traces(start_frame=0, end_frame=n_frames))
    t        = np.arange(traces.shape[0]) / fs

    return {
        "channel_ids": channel_ids,        # all channels, in order
        "labels":      label_list,         # per-channel label: good/dead/noise/out
        "bad_ids":     bad_list,           # subset flagged bad
        "fs":          fs,
        "snippet": {                       # bandpassed snippet (in-memory only)
            "t":        t,                 # (n_frames,) seconds
            "traces":   traces,            # (n_frames, n_channels)
            "channels": channel_ids,
        },
    }


def detect_bad_channels(data_path, cfg, *, snippet_seconds: float = SNIPPET_SECONDS,
                        seed: int = 0) -> dict:
    """Read a recording from *data_path* and detect bad channels.

    *cfg* is a spike-sorting config dict (or a config filename/path). Returns the
    dict described in :func:`_detect_on_recording`. Performs no disk writes.
    """
    cfg = cfg if isinstance(cfg, dict) else _load_cfg(cfg)

    recording = read_data(
        data_path   = Path(data_path),
        rec_type    = cfg["rec_type"],
        stream_name = cfg["stream_name"],
    )
    probe = create_probe(cfg["probe_manufacturer"], cfg["probe_id"],
                         cfg["channel_map"])
    recording = recording.set_probe(probe, group_mode=cfg.get("group_mode", "auto"))

    return _detect_on_recording(recording, snippet_seconds=snippet_seconds, seed=seed)


def summarise_detection(detection: dict, removed=None, policy: str = "keep") -> dict:
    """Build the small, serialisable QC summary from a detection result."""
    channel_ids = detection.get("channel_ids", [])
    bad_ids     = [str(b) for b in detection.get("bad_ids", [])]
    removed     = [str(r) for r in (removed or [])]
    n_total     = len(channel_ids)
    # per-label tally (good/dead/noise/out)
    label_counts: dict = {}
    for lab in detection.get("labels", []):
        label_counts[lab] = label_counts.get(lab, 0) + 1
    return {
        "n_channels":   n_total,
        "n_bad":        len(bad_ids),
        "bad_channels": bad_ids,
        "label_counts": label_counts,
        "policy":       policy,
        "removed":      removed,
        "n_removed":    len(removed),
        "fs":           detection.get("fs"),
    }


def write_bad_channel_report(spikes_dir, detection: dict, removed=None,
                             policy: str = "keep") -> Path:
    """Write the bad-channel QC summary into *spikes_dir* and return its path."""
    spikes_dir = Path(spikes_dir)
    spikes_dir.mkdir(parents=True, exist_ok=True)
    path = spikes_dir / BAD_CHANNELS_FILE
    summary = summarise_detection(detection, removed=removed, policy=policy)
    path.write_text(json.dumps(summary, indent=2))
    return path


def read_bad_channel_report(spikes_dir) -> dict | None:
    """Read the bad-channel QC summary from *spikes_dir*, or None if absent."""
    path = Path(spikes_dir) / BAD_CHANNELS_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_cfg(cfg_file):
    """Resolve a spike-sorting config filename/path to a dict."""
    from neurokinematics.ephys.io import get_sorting_cfg
    return get_sorting_cfg(cfg_file)
