"""Automatic, label-only spike curation from quality metrics.

This is deliberately *non-destructive*: it does not merge, split, or remove
units. It assigns each unit a label (``good`` / ``mua`` / ``noise``) based on
quality-metric thresholds and writes those labels into the phy output
(``cluster_group.tsv``) so a manual curator opens phy with the auto-curation as
a starting point and remains the final authority on unit identity.

Workflow it supports:
    metrics  ->  auto_label_from_metrics  ->  write to phy cluster_group.tsv
             ->  (human curates in phy)    ->  read_phy back (elsewhere)

Rules format (``rules`` dict): ``{metric_name: {"min": x, "max": y}}``. A unit
passes a rule if the metric is within [min, max]; only metrics actually present
in the metrics table are evaluated (missing ones are ignored), and a NaN value
fails that rule. A unit that passes *all* applicable rules is labelled ``good``,
otherwise ``fail_label`` (default ``mua``).
"""

from pathlib import Path

import numpy as np
import pandas as pd


# sensible starting thresholds; tune via the spike-sorting config's `curation` block
DEFAULT_RULES = {
    "snr":                  {"min": 2.0},
    "amplitude_cutoff":     {"max": 0.1},
    "isi_violations_ratio": {"max": 0.5},
    "presence_ratio":       {"min": 0.9},
    "firing_rate":          {"min": 0.1},
}


def auto_label_from_metrics(metrics: pd.DataFrame, rules: dict | None = None,
                            fail_label: str = "mua") -> pd.DataFrame:
    """Label units good/<fail_label> from quality-metric thresholds.

    Args:
        metrics: quality-metrics DataFrame indexed by unit id (as returned by
            spikeinterface ``compute_quality_metrics``).
        rules: ``{metric: {"min": .., "max": ..}}``; defaults to DEFAULT_RULES.
        fail_label: label for units failing any rule ('mua' or 'noise').

    Returns:
        DataFrame with columns ``unit_id``, ``group``, and one ``pass_<metric>``
        boolean per evaluated metric.
    """
    rules = rules if rules is not None else DEFAULT_RULES
    unit_ids = list(metrics.index)
    n = len(metrics)

    applicable = {m: b for m, b in rules.items() if m in metrics.columns}
    passes = np.ones(n, dtype=bool)
    flag_cols = {}
    for metric, bounds in applicable.items():
        col = pd.to_numeric(metrics[metric], errors="coerce")
        ok = ~col.isna().to_numpy()                      # NaN -> fail this rule
        if bounds.get("min") is not None:
            ok &= (col >= bounds["min"]).to_numpy()
        if bounds.get("max") is not None:
            ok &= (col <= bounds["max"]).to_numpy()
        flag_cols[f"pass_{metric}"] = ok
        passes &= ok

    out = pd.DataFrame({
        "unit_id": unit_ids,
        "group":   ["good" if p else fail_label for p in passes],
    })
    for name, vals in flag_cols.items():
        out[name] = vals
    return out


def write_phy_cluster_groups(labels: pd.DataFrame, phy_folder: Path | str) -> Path:
    """Write labels to ``<phy_folder>/cluster_group.tsv`` (phy reads this on open).

    phy expects columns ``cluster_id`` and ``group``. Overwrites any existing
    file so re-running auto-curation refreshes the suggestions.
    """
    phy_folder = Path(phy_folder)
    phy_folder.mkdir(parents=True, exist_ok=True)
    out = phy_folder / "cluster_group.tsv"
    tsv = labels[["unit_id", "group"]].rename(columns={"unit_id": "cluster_id"})
    tsv.to_csv(out, sep="\t", index=False)
    return out


def auto_curate(metrics: pd.DataFrame, rules: dict | None = None,
                phy_folder: Path | str | None = None,
                save_path: Path | str | None = None,
                fail_label: str = "mua"):
    """Run label-only auto-curation and persist results.

    Writes ``cluster_group.tsv`` into *phy_folder* (so phy shows the labels) and
    ``curated_units.csv`` into *save_path*. Returns ``(labels_df, csv_path)``.
    """
    labels = auto_label_from_metrics(metrics, rules=rules, fail_label=fail_label)
    if phy_folder is not None:
        write_phy_cluster_groups(labels, phy_folder)
    csv_path = None
    if save_path is not None:
        csv_path = Path(save_path) / "curated_units.csv"
        labels.to_csv(csv_path, index=False)
    return labels, csv_path
