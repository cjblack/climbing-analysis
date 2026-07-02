"""Quantify single-/multi-unit modulation around discrete movement events.

This module turns the movement-aligned spike rasters produced by
:func:`neurokinematics.ephys.spikes.rasters.get_movement_aligned_rasters` into a
per-unit, per-node, per-epoch modulation table.

Each *epoch* is one of the movement-event alignment points extracted upstream
(typically ``'start'`` = movement initiation, ``'max'`` = peak velocity, and
``'end'`` = movement cessation). For every (unit, node, epoch) combination the
function builds a fixed-window peri-event time histogram (PETH), z-scores it
against a pre-event baseline (for display / cross-unit comparison), and tests
whether the unit is *significantly* modulated using a per-event circular-shift
permutation null — the same null philosophy used by the GLM decoder
(:func:`neurokinematics.models.glm._circular_shift_within_groups`).

The signed per-(unit, node, epoch) modulation it returns is also the natural
input to a limb-laterality analysis (ipsilateral vs contralateral tuning): group
the output by node and compare ipsi vs contra limbs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from neurokinematics.io import save_dataset


def _bin_rasters(spike_rasters, bin_edges):
    """Bin a list of per-event spike-time arrays into a (n_events, n_bins) count matrix.

    Args:
        spike_rasters (Iterable[np.ndarray]): One array of event-relative spike
            times (seconds) per movement instance.
        bin_edges (np.ndarray): Histogram edges spanning the analysis window.

    Returns:
        np.ndarray: Integer count matrix, shape ``(n_events, len(bin_edges) - 1)``.
    """
    n_bins = len(bin_edges) - 1
    counts = np.zeros((len(spike_rasters), n_bins), dtype=float)
    for i, spikes in enumerate(spike_rasters):
        spikes = np.asarray(spikes, dtype=float)
        if spikes.size:
            counts[i], _ = np.histogram(spikes, bins=bin_edges)
    return counts


def _modulation_stat(psths, bin_centers, resp_mask, base_mask, detrend):
    """Signed response-minus-baseline rate change for each row of ``psths``.

    With ``detrend=False`` this is simply ``mean(response bins) - mean(baseline
    bins)``. With ``detrend=True`` a line is fit to each row's baseline-window
    bins, extrapolated across the whole window, and subtracted before measuring —
    so a unit whose firing simply *ramps* smoothly through the event contributes
    ~0 (the response window sits on its own extrapolated trend), and only a
    *departure* from the pre-event trend is counted as modulation. This separates
    a genuine event-locked transient from a slow ramp or the decaying tail of an
    earlier peak (relevant chiefly at the offset epoch).

    Args:
        psths (np.ndarray): ``(n, n_bins)`` (or 1-D, length ``n_bins``) PETH rates.
        bin_centers (np.ndarray): Bin-centre times (s), length ``n_bins``.
        resp_mask (np.ndarray): Boolean mask selecting response-window bins.
        base_mask (np.ndarray): Boolean mask selecting baseline-window bins.
        detrend (bool): Subtract the per-row baseline linear trend before measuring.

    Returns:
        np.ndarray: One statistic per row, shape ``(n,)``.
    """
    psths = np.atleast_2d(psths)
    if detrend:
        tb = bin_centers[base_mask]
        tb_c = tb - tb.mean()
        denom = float((tb_c ** 2).sum())
        yb = psths[:, base_mask]
        if denom > 0:
            slope = (yb * tb_c).sum(axis=1) / denom          # (n,)
        else:
            slope = np.zeros(psths.shape[0])
        intercept = yb.mean(axis=1) - slope * tb.mean()      # (n,)
        trend = intercept[:, None] + slope[:, None] * bin_centers[None, :]
        psths = psths - trend
    return psths[:, resp_mask].mean(1) - psths[:, base_mask].mean(1)


def _shuffle_null(counts, bin_size, bin_centers, resp_mask, base_mask, n_shuffle, rng, detrend):
    """Build a null distribution of the signed modulation statistic.

    Circularly shifts each event's binned spike train by a random non-zero offset
    — destroying the event's time-lock to the alignment point while preserving its
    spike count and coarse temporal structure — then recomputes the modulation
    statistic (see :func:`_modulation_stat`, including optional baseline
    detrending). Vectorised across both shuffles and events (no per-shuffle Python
    loop), so 1000s of permutations per cell stay cheap. This is the same
    per-event circular-shift null used by the GLM decoder
    (``neurokinematics.models.glm._circular_shift_within_groups``), applied to the
    binned PETH.

    Args:
        counts (np.ndarray): ``(n_events, n_bins)`` count matrix.
        bin_size (float): Bin width in seconds (counts -> rate).
        bin_centers (np.ndarray): Bin-centre times (s), length ``n_bins``.
        resp_mask (np.ndarray): Boolean mask selecting response-window bins.
        base_mask (np.ndarray): Boolean mask selecting baseline-window bins.
        n_shuffle (int): Number of permutations.
        rng (np.random.Generator): Random source.
        detrend (bool): Subtract each shuffle's baseline linear trend before
            measuring (must match the observed statistic).

    Returns:
        np.ndarray: Null statistics, shape ``(n_shuffle,)``.
    """
    n_events, n_bins = counts.shape
    # random non-zero circular shift per (shuffle, event), in [1, n_bins)
    shifts = rng.integers(1, n_bins, size=(n_shuffle, n_events))
    cols = (np.arange(n_bins)[None, None, :] - shifts[:, :, None]) % n_bins
    ev = np.arange(n_events)[None, :, None]
    shifted = counts[ev, cols]                      # (n_shuffle, n_events, n_bins)
    psth = shifted.mean(axis=1) / bin_size          # (n_shuffle, n_bins)
    return _modulation_stat(psth, bin_centers, resp_mask, base_mask, detrend)


def event_modulation(
    rasters_df: pd.DataFrame,
    *,
    epochs: list | None = None,
    nodes: list | None = None,
    units: list | None = None,
    window: tuple = (-0.5, 0.5),
    bin_size: float = 0.02,
    baseline_window: tuple = (-0.5, -0.1),
    response_window: tuple = (0.0, 0.2),
    detrend_baseline: bool = False,
    n_shuffle: int = 1000,
    fdr: bool = True,
    fdr_alpha: float = 0.05,
    min_events: int = 3,
    seed: int = 0,
    progress: bool = True,
    save_path: Path | str | None = None,
    filename: str | None = None,
) -> xr.Dataset:
    """Per-unit modulation around discrete movement epochs, with a shuffle null.

    For every (unit, node, epoch) combination this builds a fixed-window PETH,
    z-scores it against the pre-event baseline window, measures a signed
    response-minus-baseline rate change, and tests it against a per-event
    circular-shift permutation null (two-sided). Across all tested combinations
    the p-values are optionally Benjamini-Hochberg FDR corrected.

    Args:
        rasters_df (pd.DataFrame): Output of
            :func:`~neurokinematics.ephys.spikes.rasters.get_movement_aligned_rasters`
            — one row per (unit, node, epoch, movement instance) with an
            event-relative ``spike_raster`` array. Required columns:
            ``unit_id``, ``node``, ``movement_event``, ``spike_raster``.
        epochs (list | None): Movement-event types to analyse. Defaults to all
            present (e.g. ``['start', 'max', 'end']``).
        nodes (list | None): Body nodes (limbs) to analyse. Defaults to all present.
        units (list | None): Unit ids to analyse. Defaults to all present.
        window (tuple): ``(start_s, end_s)`` PETH window around the event.
        bin_size (float): PETH bin width in seconds.
        baseline_window (tuple): Window (within ``window``) used as the z-scoring
            reference (mean/std across its bins) and as the modulation baseline.
        response_window (tuple): Window (within ``window``) over which the
            modulation statistic is measured.
        detrend_baseline (bool): If True, fit a line to each cell's baseline-window
            PETH, extrapolate it across the window, and subtract it before measuring
            modulation (and in the null). A unit that simply ramps smoothly through
            the event then contributes ~0; only a *departure* from the pre-event
            trend counts. Use to avoid flagging slow ramps or the decaying tail of
            an earlier peak as event-locked modulation — most relevant at the
            offset epoch. Display PETHs (``psth_hz``/``psth_z``) are left untouched.
        n_shuffle (int): Permutations for the circular-shift null. ``0`` skips
            testing (p-values become NaN).
        fdr (bool): Benjamini-Hochberg correct p-values across all tested cells.
        fdr_alpha (float): FDR level for the ``significant`` flag.
        min_events (int): Minimum movement instances required to test a cell;
            sparser cells are filled with NaN.
        seed (int): Seed for the permutation RNG.
        progress (bool): Show a tqdm progress bar over (unit, node, epoch) cells.
        save_path (Path | str | None): If given, write the result as a ``.zarr``
            store under this directory.
        filename (str | None): Store name to write under ``save_path`` (e.g. a
            timestamped ``event_modulation_20260625_141500.zarr`` so repeated runs
            don't overwrite each other). Defaults to ``event_modulation.zarr``.

    Returns:
        xr.Dataset: Dims ``(unit, node, epoch, time_bin)``. Variables:
            ``psth_hz`` / ``psth_z`` (PETHs), and per-cell scalars
            ``modulation`` (Hz, signed), ``response_z``, ``baseline_rate`` (Hz),
            ``n_events``, ``p_value``, ``p_fdr``, ``significant``.

    Example:
        >>> mod = event_modulation(rasters_df, n_shuffle=1000)
        >>> mod['significant'].sel(epoch='start').sum().item()   # k units modulated at onset
    """
    required = {"unit_id", "node", "movement_event", "spike_raster"}
    missing = required - set(rasters_df.columns)
    if missing:
        raise ValueError(f"rasters_df missing required columns: {sorted(missing)}")

    units = list(units) if units is not None else sorted(rasters_df["unit_id"].unique())
    nodes = list(nodes) if nodes is not None else sorted(rasters_df["node"].unique())
    epochs = list(epochs) if epochs is not None else sorted(rasters_df["movement_event"].unique())

    bin_edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2
    n_bins = len(bin_centers)

    base_mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    resp_mask = (bin_centers >= response_window[0]) & (bin_centers < response_window[1])
    if not base_mask.any() or not resp_mask.any():
        raise ValueError("baseline_window and response_window must each cover >=1 bin within window")

    shape = (len(units), len(nodes), len(epochs))
    psth_hz = np.full((*shape, n_bins), np.nan)
    psth_z = np.full((*shape, n_bins), np.nan)
    modulation = np.full(shape, np.nan)
    response_z = np.full(shape, np.nan)
    baseline_rate = np.full(shape, np.nan)
    n_events_arr = np.zeros(shape, dtype=int)
    p_value = np.full(shape, np.nan)

    rng = np.random.default_rng(seed)
    # group once for fast lookup rather than re-querying the frame in the triple loop
    grouped = rasters_df.groupby(["unit_id", "node", "movement_event"])

    combos = [(ui, uid, ni, nd, ei, ep)
              for ui, uid in enumerate(units)
              for ni, nd in enumerate(nodes)
              for ei, ep in enumerate(epochs)]
    iterator = tqdm(combos, desc="Event modulation", unit="cell") if progress else combos

    for ui, uid, ni, nd, ei, ep in iterator:
        try:
            rows = grouped.get_group((uid, nd, ep))
        except KeyError:
            continue
        n_ev = len(rows)
        n_events_arr[ui, ni, ei] = n_ev
        if n_ev < min_events:
            continue

        counts = _bin_rasters(rows["spike_raster"].tolist(), bin_edges)
        rates = counts / bin_size
        psth = rates.mean(axis=0)
        psth_hz[ui, ni, ei] = psth

        base_mu = psth[base_mask].mean()
        base_sd = psth[base_mask].std()
        baseline_rate[ui, ni, ei] = base_mu
        if base_sd > 0:
            z = (psth - base_mu) / base_sd
        else:
            z = np.zeros_like(psth)
        psth_z[ui, ni, ei] = z
        response_z[ui, ni, ei] = z[resp_mask].mean()

        stat = _modulation_stat(psth, bin_centers, resp_mask, base_mask, detrend_baseline)[0]
        modulation[ui, ni, ei] = stat

        if n_shuffle and n_ev > 1 and n_bins > 1:
            null = _shuffle_null(counts, bin_size, bin_centers, resp_mask, base_mask,
                                 n_shuffle, rng, detrend_baseline)
            p_value[ui, ni, ei] = (np.sum(np.abs(null) >= np.abs(stat)) + 1) / (n_shuffle + 1)

    # FDR across every tested cell
    p_fdr = np.full(shape, np.nan)
    significant = np.zeros(shape, dtype=bool)
    flat_p = p_value.reshape(-1)
    tested = np.isfinite(flat_p)
    if tested.any():
        if fdr:
            rej, p_corr, _, _ = multipletests(flat_p[tested], alpha=fdr_alpha, method="fdr_bh")
            pc = np.full_like(flat_p, np.nan)
            sig = np.zeros_like(flat_p, dtype=bool)
            pc[tested] = p_corr
            sig[tested] = rej
            p_fdr = pc.reshape(shape)
            significant = sig.reshape(shape)
        else:
            p_fdr = p_value.copy()
            significant = (p_value <= fdr_alpha)

    ds = xr.Dataset(
        data_vars={
            "psth_hz": (["unit", "node", "epoch", "time_bin"], psth_hz),
            "psth_z": (["unit", "node", "epoch", "time_bin"], psth_z),
            "modulation": (["unit", "node", "epoch"], modulation),
            "response_z": (["unit", "node", "epoch"], response_z),
            "baseline_rate": (["unit", "node", "epoch"], baseline_rate),
            "n_events": (["unit", "node", "epoch"], n_events_arr),
            "p_value": (["unit", "node", "epoch"], p_value),
            "p_fdr": (["unit", "node", "epoch"], p_fdr),
            "significant": (["unit", "node", "epoch"], significant),
        },
        coords={
            "unit": np.asarray(units),
            "node": np.asarray(nodes),
            "epoch": np.asarray(epochs),
            "time_bin": bin_centers,
        },
        attrs={
            "window": list(window),
            "bin_size": bin_size,
            "baseline_window": list(baseline_window),
            "response_window": list(response_window),
            "detrend_baseline": bool(detrend_baseline),
            "n_shuffle": int(n_shuffle),
            "fdr": bool(fdr),
            "fdr_alpha": float(fdr_alpha),
            "min_events": int(min_events),
        },
    )

    if save_path:
        save_path = Path(save_path) / (filename or "event_modulation.zarr")
        save_dataset(
            ds,
            save_path,
            chunks={"unit": -1, "node": -1, "epoch": -1, "time_bin": -1},
        )

    return ds
