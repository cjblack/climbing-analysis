import numpy as np
from scipy.interpolate import interp1d


def _consecutive_runs(mask):
    """Indices of each contiguous run of True in a 1-D boolean mask."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)


def fill_missing(Y, kind="linear", max_gap=None):
    """*Adapted from sleap's pose tools* — fill missing values independently
    along each dimension after the first.

    Args:
        Y: array with time on the first axis.
        kind: interpolation kind passed to ``scipy.interpolate.interp1d``.
        max_gap (int | None): maximum gap length, in frames, to interpolate
            across. Gaps longer than this (including leading/trailing runs) are
            left as NaN instead of being interpolated — so long dropouts
            (occlusion, tracking loss) aren't fabricated into a straight line and
            instead propagate downstream as missing data. ``None`` (default)
            interpolates every gap regardless of length (original behaviour).
    """
    # Store initial shape.
    initial_shape = Y.shape
    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))
    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        nan_mask = np.isnan(y)          # original gaps (used for max_gap below)
        valid = np.flatnonzero(~nan_mask)
        if valid.size == 0:
            continue                    # all-NaN slice: nothing to fill from

        # Interpolate interior gaps
        if valid.size >= 2:
            f = interp1d(valid, y[valid], kind=kind,
                         fill_value=np.nan, bounds_error=False)
            xq = np.flatnonzero(nan_mask)
            y[xq] = f(xq)

        # Fill any remaining (leading/trailing) NaNs with nearest non-NaN values
        rem = np.isnan(y)
        if rem.any():
            y[rem] = np.interp(np.flatnonzero(rem),
                               np.flatnonzero(~rem), y[~rem])

        # Re-open gaps longer than max_gap so they stay missing
        if max_gap is not None:
            for run in _consecutive_runs(nan_mask):
                if run.size > max_gap:
                    y[run] = np.nan

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

def remove_low_confidence(locations, scores, thresh = 0.7, max_gap=None):
    filtered_locations = np.copy(locations)
    for i in range(scores.shape[1]):
        mask = scores[:,i,0] < thresh
        filtered_locations[mask,i,0] = np.nan
        filtered_locations[mask,i,1] = np.nan
    filtered_locations = fill_missing(filtered_locations, max_gap=max_gap)
    return filtered_locations

def remove_high_velocity(locations, thresh = 20, max_gap=None):
    filtered_locations = np.copy(locations)
    for i in range(filtered_locations.shape[1]):
        mask = np.abs(np.gradient(filtered_locations[:,i,1],axis=0)) > thresh
        mask = np.squeeze(mask.T)
        filtered_locations[mask,i,0] = np.nan
        filtered_locations[mask,i,1] = np.nan
    filtered_locations = fill_missing(filtered_locations, max_gap=max_gap)
    return filtered_locations