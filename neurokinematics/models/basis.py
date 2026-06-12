"""Temporal basis functions for neural encoding GLMs.

An encoding GLM that only regresses spikes in bin *t* on a kinematic feature in
the *same* bin *t* can only detect instantaneous coupling. To ask whether a unit
encodes a movement feature with a lead or a lag — e.g. firing that *precedes*
velocity (preparatory / motor) versus firing that *follows* it (sensory /
feedback) — the feature must enter the design matrix at a range of temporal
offsets relative to the spike bin.

Rather than one free parameter per offset (which is noisy and collinear), the
standard approach in the spike-train GLM literature (Pillow et al., 2008) is to
project the lagged feature onto a small set of smooth, overlapping
**raised-cosine** basis functions. The GLM then learns one weight per basis
function, and the learned temporal filter is ``basis @ weights``. The location of
the filter's mass tells you the lead/lag.

Sign convention used throughout this module
--------------------------------------------
The design is built so that, for a feature ``x`` and spike bin ``t``::

    design[t] = sum_i  basis[i] * x[t + offsets[i]]

so an *offset* is the time of the kinematic sample **relative to the spike bin**,
measured in bins:

* ``offset > 0`` → the spike bin depends on a *future* kinematic sample
  (the unit *leads* the movement).
* ``offset < 0`` → the spike bin depends on a *past* kinematic sample
  (the unit *lags* the movement).

Bins whose filter would reach outside the event boundary are filled with NaN so
the caller's finite-value mask drops them (no wraparound, no zero-padding bias).
"""

import numpy as np


def offsets_from_window(window, bin_size):
    """Convert a window in seconds to an inclusive array of integer bin offsets.

    Args:
        window (tuple | list): ``(start_s, end_s)`` offsets in seconds of the
            kinematic sample relative to the spike bin. Negative values look into
            the past (unit lags movement), positive into the future (unit leads).
        bin_size (float): Bin width in seconds.

    Returns:
        np.ndarray: 1-D array of integer bin offsets, inclusive of both ends.

    Raises:
        ValueError: If ``window`` is not increasing or ``bin_size`` is not positive.
    """
    start_s, end_s = float(window[0]), float(window[1])
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}.")
    if end_s < start_s:
        raise ValueError(f"window must be increasing, got {window}.")

    lo = int(np.floor(start_s / bin_size))
    hi = int(np.ceil(end_s / bin_size))
    return np.arange(lo, hi + 1)


def raised_cosine_basis(offsets, n_basis, spacing="linear", overlap=2.0):
    """Build a raised-cosine temporal basis over a set of integer bin offsets.

    Each basis function is a smooth, non-negative bump,
    ``0.5 * (1 + cos(theta))`` clipped to its support, with the bumps evenly
    spaced (in linear or log-warped offset space) across the window and
    overlapping so their sum is roughly flat.

    Args:
        offsets (np.ndarray): 1-D array of integer bin offsets (e.g. from
            :func:`offsets_from_window`).
        n_basis (int): Number of basis functions. Must be >= 1.
        spacing (str, optional): ``'linear'`` places bump centres evenly across
            the offsets; ``'log'`` warps the offset axis so resolution is finer
            near the smallest offset (the canonical Pillow spacing — most useful
            for one-sided/causal windows). Defaults to ``'linear'``.
        overlap (float, optional): Half-width of each bump as a multiple of the
            centre spacing. Larger values give smoother, more overlapping bumps.
            Defaults to 2.0.

    Returns:
        np.ndarray: Basis matrix of shape ``(len(offsets), n_basis)``, columns
        ordered by increasing centre offset.

    Raises:
        ValueError: If ``n_basis < 1`` or ``spacing`` is unknown.
    """
    offsets = np.asarray(offsets, dtype=float)
    if n_basis < 1:
        raise ValueError(f"n_basis must be >= 1, got {n_basis}.")
    n_off = offsets.shape[0]

    if spacing == "linear":
        warped = offsets
    elif spacing == "log":
        # warp so equal spacing in 'warped' is finer near the start of the window
        shift = offsets - offsets.min() + 1.0  # strictly positive
        warped = np.log(shift)
    else:
        raise ValueError(f"spacing must be 'linear' or 'log', got {spacing!r}.")

    lo, hi = warped.min(), warped.max()
    if n_basis == 1:
        centers = np.array([(lo + hi) / 2.0])
        spacing_w = (hi - lo) or 1.0
    else:
        centers = np.linspace(lo, hi, n_basis)
        spacing_w = centers[1] - centers[0]

    half_width = overlap * spacing_w
    basis = np.zeros((n_off, n_basis), dtype=float)
    for j, c in enumerate(centers):
        theta = (warped - c) * (np.pi / half_width)
        theta = np.clip(theta, -np.pi, np.pi)  # support = |warped - c| <= half_width
        basis[:, j] = 0.5 * (1.0 + np.cos(theta))
    return basis


def _shift(x, offset):
    """Shift a (n_events, n_bins) array along time so out[:, t] = x[:, t + offset].

    Out-of-bounds positions are filled with NaN. Shifting is done per event row;
    there is no wraparound across the time axis or between events.
    """
    n_events, n_bins = x.shape
    out = np.full((n_events, n_bins), np.nan, dtype=float)
    if offset == 0:
        out[:] = x
    elif offset > 0:
        if offset < n_bins:
            out[:, : n_bins - offset] = x[:, offset:]
    else:  # offset < 0
        k = -offset
        if k < n_bins:
            out[:, k:] = x[:, : n_bins - k]
    return out


def lagged_feature_design(x, offsets, basis):
    """Project a per-event feature onto a temporal basis at a range of offsets.

    For each spike bin ``t`` the feature is gathered at ``t + offsets`` (within
    the same event) and projected onto ``basis``. Bins whose filter would reach
    outside the event are NaN, so a downstream finite-value mask drops exactly the
    event-edge bins that lack full temporal context.

    Args:
        x (np.ndarray): Feature array of shape ``(n_events, n_bins)``.
        offsets (np.ndarray): Integer bin offsets, shape ``(n_offsets,)``.
        basis (np.ndarray): Basis matrix of shape ``(n_offsets, n_basis)``.

    Returns:
        np.ndarray: Design tensor of shape ``(n_events, n_bins, n_basis)``.

    Raises:
        ValueError: If ``basis`` rows do not match ``offsets``.
    """
    x = np.asarray(x, dtype=float)
    offsets = np.asarray(offsets)
    if basis.shape[0] != offsets.shape[0]:
        raise ValueError(
            f"basis has {basis.shape[0]} rows but there are {offsets.shape[0]} offsets."
        )

    # (n_events, n_bins, n_offsets): feature gathered at each offset
    shifted = np.stack([_shift(x, int(o)) for o in offsets], axis=-1)
    # project onto basis -> (n_events, n_bins, n_basis); NaNs at edges propagate
    design = np.tensordot(shifted, basis, axes=([2], [0]))
    return design
