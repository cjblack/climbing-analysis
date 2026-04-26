import numpy as np
from scipy.interpolate import interp1d


def fill_missing(Y, kind="linear"):
    """*Taken from sleap's pose estimation tools** Fills missing values independently along each dimension after the first."""
    # Store initial shape.
    initial_shape = Y.shape
    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))
    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

def filter_confidence(locations, scores, thresh = 0.7):
    filtered_locations = np.copy(locations)
    for i in range(scores.shape[1]):
        mask = scores[:,i,0] < thresh
        filtered_locations[mask,i,0] = np.nan
        filtered_locations[mask,i,1] = np.nan
    filtered_locations = fill_missing(filtered_locations)
    return filtered_locations

def filter_velocity(locations, thresh = 20):
    filtered_locations = np.copy(locations)
    for i in range(filtered_locations.shape[1]):
        mask = np.abs(np.gradient(filtered_locations[:,i,1],axis=0)) > thresh
        mask = np.squeeze(mask.T)
        filtered_locations[mask,i,0] = np.nan
        filtered_locations[mask,i,1] = np.nan
    filtered_locations = fill_missing(filtered_locations)
    return filtered_locations