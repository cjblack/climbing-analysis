import numpy as np
import pytest
from neurokinematics.pose.preprocessing.cleaning import remove_low_confidence


def test_low_confidence_points_are_replaced():
    """Points with score below threshold should be interpolated, not kept as-is."""
    # Shape: (T=5, N=1, 2) locations and (T=5, N=1, 1) scores
    locations = np.array([[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]], [[4.0, 4.0]]])
    scores = np.array([[[0.9]], [[0.9]], [[0.1]], [[0.9]], [[0.9]]])  # frame 2 is low-confidence

    result = remove_low_confidence(locations, scores, thresh=0.7)

    # Frame 2 was low-confidence — after fill_missing it should be interpolated (~[2.0, 2.0])
    assert not np.any(np.isnan(result)), "No NaNs should remain after interpolation"
    np.testing.assert_allclose(result[2, 0, :], [2.0, 2.0], atol=1e-6)


def test_high_confidence_points_are_unchanged():
    """Points with score at or above threshold should not be modified."""
    locations = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
    scores = np.ones((3, 1, 1)) * 0.9  # all high confidence

    result = remove_low_confidence(locations, scores, thresh=0.7)

    np.testing.assert_array_equal(result, locations)


def test_custom_threshold():
    """A stricter threshold should flag more points."""
    locations = np.array([[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]], [[4.0, 4.0]]])
    scores = np.array([[[0.9]], [[0.9]], [[0.8]], [[0.9]], [[0.9]]])

    # With default thresh=0.7, frame 2 (score=0.8) passes — values unchanged
    result_default = remove_low_confidence(locations, scores, thresh=0.7)
    np.testing.assert_array_equal(result_default[2, 0, :], [2.0, 2.0])

    # With thresh=0.85, frame 2 (score=0.8) is flagged — value is interpolated (still ~2.0 here)
    result_strict = remove_low_confidence(locations, scores, thresh=0.85)
    assert not np.any(np.isnan(result_strict))
