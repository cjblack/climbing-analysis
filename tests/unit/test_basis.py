"""Unit tests for neurokinematics.models.basis (temporal basis functions)."""

import numpy as np
import pytest

from neurokinematics.models.basis import (
    offsets_from_window,
    raised_cosine_basis,
    lagged_feature_design,
    _shift,
)


# ---------------------------------------------------------------------------
# offsets_from_window
# ---------------------------------------------------------------------------

class TestOffsetsFromWindow:

    def test_symmetric_window(self):
        offsets = offsets_from_window((-0.1, 0.1), bin_size=0.05)
        assert list(offsets) == [-2, -1, 0, 1, 2]

    def test_zero_window_is_single_zero_offset(self):
        offsets = offsets_from_window((0.0, 0.0), bin_size=0.02)
        assert list(offsets) == [0]

    def test_causal_only_window_has_no_positive_offsets(self):
        offsets = offsets_from_window((-0.1, 0.0), bin_size=0.05)
        assert offsets.max() == 0
        assert offsets.min() == -2

    def test_rounds_outward(self):
        # 0.06 / 0.05 -> ceil 2 ; -0.06 / 0.05 -> floor -2
        offsets = offsets_from_window((-0.06, 0.06), bin_size=0.05)
        assert list(offsets) == [-2, -1, 0, 1, 2]

    def test_non_increasing_window_raises(self):
        with pytest.raises(ValueError, match="increasing"):
            offsets_from_window((0.1, -0.1), bin_size=0.05)

    def test_non_positive_bin_size_raises(self):
        with pytest.raises(ValueError, match="bin_size"):
            offsets_from_window((-0.1, 0.1), bin_size=0.0)


# ---------------------------------------------------------------------------
# raised_cosine_basis
# ---------------------------------------------------------------------------

class TestRaisedCosineBasis:

    def test_shape(self):
        offsets = np.arange(-5, 6)
        basis = raised_cosine_basis(offsets, n_basis=4)
        assert basis.shape == (len(offsets), 4)

    def test_non_negative(self):
        offsets = np.arange(-5, 6)
        basis = raised_cosine_basis(offsets, n_basis=4)
        assert (basis >= 0).all()

    def test_single_basis_is_centered_bump(self):
        offsets = np.arange(-5, 6)
        basis = raised_cosine_basis(offsets, n_basis=1)
        assert basis.shape == (len(offsets), 1)
        # peak should be at the central offset (0)
        peak_idx = np.argmax(basis[:, 0])
        assert offsets[peak_idx] == 0

    def test_bump_centers_increase_across_columns(self):
        offsets = np.arange(-5, 6)
        basis = raised_cosine_basis(offsets, n_basis=4)
        peak_offsets = [offsets[np.argmax(basis[:, j])] for j in range(basis.shape[1])]
        # centres are monotonically non-decreasing left-to-right
        assert peak_offsets == sorted(peak_offsets)
        assert peak_offsets[0] < peak_offsets[-1]

    def test_columns_are_full_rank(self):
        offsets = np.arange(-8, 9)
        basis = raised_cosine_basis(offsets, n_basis=5)
        assert np.linalg.matrix_rank(basis) == 5

    def test_log_spacing_runs_and_shapes(self):
        offsets = np.arange(0, 11)
        basis = raised_cosine_basis(offsets, n_basis=4, spacing="log")
        assert basis.shape == (len(offsets), 4)
        assert (basis >= 0).all()

    def test_invalid_spacing_raises(self):
        with pytest.raises(ValueError, match="spacing"):
            raised_cosine_basis(np.arange(-3, 4), n_basis=3, spacing="bogus")

    def test_invalid_n_basis_raises(self):
        with pytest.raises(ValueError, match="n_basis"):
            raised_cosine_basis(np.arange(-3, 4), n_basis=0)


# ---------------------------------------------------------------------------
# _shift
# ---------------------------------------------------------------------------

class TestShift:

    def test_zero_offset_is_identity(self):
        x = np.arange(12, dtype=float).reshape(2, 6)
        np.testing.assert_array_equal(_shift(x, 0), x)

    def test_positive_offset_pulls_future_into_present(self):
        x = np.arange(6, dtype=float).reshape(1, 6)  # [0,1,2,3,4,5]
        out = _shift(x, 1)  # out[t] = x[t+1]
        np.testing.assert_array_equal(out[0, :5], [1, 2, 3, 4, 5])
        assert np.isnan(out[0, 5])

    def test_negative_offset_pulls_past_into_present(self):
        x = np.arange(6, dtype=float).reshape(1, 6)
        out = _shift(x, -1)  # out[t] = x[t-1]
        np.testing.assert_array_equal(out[0, 1:], [0, 1, 2, 3, 4])
        assert np.isnan(out[0, 0])

    def test_offset_beyond_length_is_all_nan(self):
        x = np.arange(4, dtype=float).reshape(1, 4)
        assert np.isnan(_shift(x, 10)).all()
        assert np.isnan(_shift(x, -10)).all()

    def test_no_cross_event_bleed(self):
        x = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
        out = _shift(x, 1)
        # row 1 bin 2 must be NaN, not row 1's first element wrapping in
        assert np.isnan(out[0, 2])
        assert np.isnan(out[1, 2])
        np.testing.assert_array_equal(out[:, 0], [1.0, 11.0])


# ---------------------------------------------------------------------------
# lagged_feature_design
# ---------------------------------------------------------------------------

class TestLaggedFeatureDesign:

    def test_shape(self):
        x = np.random.default_rng(0).normal(size=(5, 8))
        offsets = np.array([-1, 0, 1])
        basis = raised_cosine_basis(offsets, n_basis=3)
        design = lagged_feature_design(x, offsets, basis)
        assert design.shape == (5, 8, 3)

    def test_zero_offset_identity_basis_reproduces_feature(self):
        x = np.arange(20, dtype=float).reshape(4, 5)
        offsets = np.array([0])
        basis = np.array([[1.0]])
        design = lagged_feature_design(x, offsets, basis)
        np.testing.assert_array_equal(design[:, :, 0], x)

    def test_edges_are_nan_when_window_extends(self):
        x = np.ones((3, 6))
        offsets = np.array([-1, 0, 1])
        basis = raised_cosine_basis(offsets, n_basis=2)
        design = lagged_feature_design(x, offsets, basis)
        # first and last bins lack full context -> NaN across all basis columns
        assert np.isnan(design[:, 0, :]).all()
        assert np.isnan(design[:, -1, :]).all()
        # interior bins are finite
        assert np.isfinite(design[:, 1:-1, :]).all()

    def test_single_offset_basis_scales_shifted_feature(self):
        x = np.arange(6, dtype=float).reshape(1, 6)
        offsets = np.array([1])
        basis = np.array([[2.0]])  # single bump, weight 2
        design = lagged_feature_design(x, offsets, basis)
        # design[t] = 2 * x[t+1]
        np.testing.assert_array_equal(design[0, :5, 0], 2 * np.array([1, 2, 3, 4, 5]))
        assert np.isnan(design[0, 5, 0])

    def test_basis_offset_mismatch_raises(self):
        x = np.ones((2, 4))
        offsets = np.array([-1, 0, 1])
        bad_basis = np.ones((2, 3))  # 2 rows != 3 offsets
        with pytest.raises(ValueError, match="rows"):
            lagged_feature_design(x, offsets, bad_basis)
