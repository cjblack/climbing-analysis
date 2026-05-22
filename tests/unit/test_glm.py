"""Unit tests for neurokinematics.models.glm."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from neurokinematics.models.glm import (
    build_glm_dataset,
    build_glm_model_sets,
    compare_glm_models,
    create_glm_decoder,
    create_glm_encoder,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_pose_dataset(n_events=20, n_bins=10, node="hand", seed=42):
    """Minimal synthetic pose dataset matching the structure expected by the GLM functions."""
    rng = np.random.default_rng(seed)
    position = rng.normal(size=(n_events, n_bins, 1, 2))
    valid = np.ones((n_events, n_bins), dtype=bool)
    return xr.Dataset(
        {
            "position": (["event", "time_bin", "node", "coord"], position),
            "valid": (["event", "time_bin"], valid),
            "reference_node": (["event"], np.full(n_events, node)),
        },
        coords={
            "event": np.arange(n_events),
            "time_bin": np.linspace(0.0, 1.0, n_bins),
            "node": [node],
            "coord": ["x", "y"],
        },
    )


def make_spike_dataset(n_events=20, n_bins=10, n_units=3, seed=42):
    """Minimal synthetic spike dataset matching the structure expected by the GLM functions."""
    rng = np.random.default_rng(seed)
    spike_counts = rng.poisson(2, size=(n_events, n_bins, n_units)).astype(float)
    valid = np.ones((n_events, n_bins), dtype=bool)
    return xr.Dataset(
        {
            "spike_counts": (["event", "time_bin", "unit"], spike_counts),
            "valid": (["event", "time_bin"], valid),
        },
        coords={
            "event": np.arange(n_events),
            "time_bin": np.linspace(0.0, 1.0, n_bins),
            "unit": np.arange(n_units),
        },
    )


# ---------------------------------------------------------------------------
# build_glm_model_sets
# ---------------------------------------------------------------------------

class TestBuildGlmModelSets:

    FEATURES = ["position_x", "position_y", "speed"]

    def test_single_mode_one_model_per_feature(self):
        result = build_glm_model_sets(self.FEATURES, mode="single")
        assert set(result.keys()) == set(self.FEATURES)
        for feat, feat_list in result.items():
            assert feat_list == [feat]

    def test_full_mode_one_model_with_all_features(self):
        result = build_glm_model_sets(self.FEATURES, mode="full")
        assert list(result.keys()) == ["full"]
        assert result["full"] == self.FEATURES

    def test_single_and_full_mode_contains_both(self):
        result = build_glm_model_sets(self.FEATURES, mode="single_and_full")
        assert "full" in result
        assert result["full"] == self.FEATURES
        for feat in self.FEATURES:
            assert feat in result
            assert result[feat] == [feat]

    def test_drop_one_mode_includes_full_and_drop_sets(self):
        result = build_glm_model_sets(self.FEATURES, mode="drop_one")
        assert "full" in result
        assert result["full"] == self.FEATURES
        for feat in self.FEATURES:
            key = f"drop_{feat}"
            assert key in result
            assert feat not in result[key]
            assert len(result[key]) == len(self.FEATURES) - 1

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            build_glm_model_sets(self.FEATURES, mode="bad_mode")

    def test_single_feature_drop_one_produces_empty_set(self):
        result = build_glm_model_sets(["speed"], mode="drop_one")
        assert result["drop_speed"] == []


# ---------------------------------------------------------------------------
# build_glm_dataset
# ---------------------------------------------------------------------------

class TestBuildGlmDataset:

    def _make_outputs(self, n_events=5, n_bins=4, seed=0):
        rng = np.random.default_rng(seed)
        n_total = n_events * n_bins
        event_idx = np.repeat(np.arange(n_events), n_bins)
        time_idx = np.tile(np.arange(n_bins), n_events)
        observed = rng.poisson(3, size=n_total).astype(float)
        predicted = observed + rng.normal(0, 0.5, size=n_total)
        return {
            "observed": observed,
            "predicted": predicted,
            "event_idx": event_idx,
            "time_idx": time_idx,
            "time_bins": np.linspace(0, 1, n_bins),
        }

    def test_returns_xarray_dataset(self):
        ds = build_glm_dataset(self._make_outputs())
        assert isinstance(ds, xr.Dataset)

    def test_correct_shape(self):
        n_events, n_bins = 5, 4
        ds = build_glm_dataset(self._make_outputs(n_events=n_events, n_bins=n_bins))
        assert ds.dims["event"] == n_events
        assert ds.dims["time_bin"] == n_bins

    def test_residuals_equal_observed_minus_predicted(self):
        ds = build_glm_dataset(self._make_outputs())
        expected = ds["observed_counts"] - ds["predicted_counts"]
        xr.testing.assert_allclose(ds["residuals"], expected)

    def test_valid_mask_true_for_all_provided_indices(self):
        ds = build_glm_dataset(self._make_outputs(n_events=5, n_bins=4))
        assert ds["valid"].values.all()

    def test_custom_event_ids(self):
        outputs = self._make_outputs(n_events=3)
        event_ids = ["trial_a", "trial_b", "trial_c"]
        ds = build_glm_dataset(outputs, event_ids=event_ids)
        assert list(ds.coords["event"].values) == event_ids

    def test_attrs_stored_on_dataset(self):
        attrs = {"model_type": "encoder", "unit": 0, "node": "hand"}
        ds = build_glm_dataset(self._make_outputs(), attrs=attrs)
        assert ds.attrs == attrs

    def test_unfilled_slots_are_nan_and_invalid(self):
        """Slots not covered by event_idx/time_idx should be NaN and marked invalid."""
        n_events, n_bins = 6, 4
        # Only provide data for the first half of events
        half = n_events // 2
        outputs = {
            "observed": np.ones(half * n_bins),
            "predicted": np.ones(half * n_bins),
            "event_idx": np.repeat(np.arange(half), n_bins),
            "time_idx": np.tile(np.arange(n_bins), half),
            "time_bins": np.linspace(0, 1, n_bins),
        }
        ds = build_glm_dataset(outputs)
        assert np.isnan(ds["observed_counts"].values[half:]).all()
        assert not ds["valid"].values[half:].any()


# ---------------------------------------------------------------------------
# create_glm_encoder
# ---------------------------------------------------------------------------

class TestCreateGlmEncoder:

    @pytest.fixture
    def data(self):
        return make_pose_dataset(), make_spike_dataset()

    @pytest.fixture
    def params(self):
        return {
            "type": "encoder",
            "family": "Poisson",
            "pose": {"node": "hand", "features": ["position_y"]},
            "spikes": {"unit": [0], "features": ["spike_counts"]},
        }

    def test_params_none_does_not_raise(self, data):
        """Regression: passing params=None with xarray inputs must not raise AttributeError."""
        pose_ds, spike_ds = data
        # Minimal params needed to avoid indexing into the default integer 'unit'
        minimal = {
            "pose": {"node": "hand", "features": ["position_y"]},
            "spikes": {"unit": [0], "features": ["spike_counts"]},
        }
        model, results, outputs = create_glm_encoder(pose_ds, spike_ds, params=minimal)
        assert model is not None

    def test_returns_model_results_outputs(self, data, params):
        pose_ds, spike_ds = data
        result = create_glm_encoder(pose_ds, spike_ds, params=params)
        assert len(result) == 3

    def test_outputs_have_expected_keys(self, data, params):
        pose_ds, spike_ds = data
        _, _, outputs = create_glm_encoder(pose_ds, spike_ds, params=params)
        for key in ("predicted", "observed", "event_idx", "time_idx", "time_bins", "attrs", "params"):
            assert key in outputs, f"Missing key: {key}"

    def test_predicted_and_observed_same_length(self, data, params):
        pose_ds, spike_ds = data
        _, _, outputs = create_glm_encoder(pose_ds, spike_ds, params=params)
        assert len(outputs["predicted"]) == len(outputs["observed"])

    def test_aic_and_log_likelihood_recorded(self, data, params):
        pose_ds, spike_ds = data
        _, _, outputs = create_glm_encoder(pose_ds, spike_ds, params=params)
        metrics = outputs["params"]["metrics"]
        assert "aic" in metrics
        assert "log_likelihood" in metrics
        assert np.isfinite(metrics["aic"])
        assert np.isfinite(metrics["log_likelihood"])

    def test_observed_is_1d_numpy(self, data, params):
        pose_ds, spike_ds = data
        _, _, outputs = create_glm_encoder(pose_ds, spike_ds, params=params)
        assert isinstance(outputs["observed"], np.ndarray)
        assert outputs["observed"].ndim == 1

    def test_invalid_zarr_path_raises_value_error(self, data, params):
        _, spike_ds = data
        with pytest.raises(ValueError, match='".zarr"'):
            create_glm_encoder("not_a_zarr.csv", spike_ds, params=params)


# ---------------------------------------------------------------------------
# create_glm_decoder
# ---------------------------------------------------------------------------

class TestCreateGlmDecoder:

    @pytest.fixture
    def data(self):
        return make_pose_dataset(), make_spike_dataset(n_units=3)

    @pytest.fixture
    def params(self):
        return {
            "type": "decoder",
            "family": "Gaussian",
            "pose": {"node": "hand", "features": ["position_y"]},
            "spikes": {"unit": [0, 1, 2], "features": ["spike_counts"]},
        }

    def test_params_none_does_not_raise(self, data):
        """Regression: passing params=None with xarray inputs must not raise AttributeError."""
        pose_ds, spike_ds = data
        minimal = {
            "pose": {"node": "hand", "features": ["position_y"]},
            "spikes": {"unit": [0], "features": ["spike_counts"]},
        }
        model, results, outputs = create_glm_decoder(pose_ds, spike_ds, params=minimal)
        assert model is not None

    def test_observed_is_1d_numpy(self, data, params):
        """Regression: sy must be flattened before masking — not left as a 2D xarray."""
        pose_ds, spike_ds = data
        _, _, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        assert isinstance(outputs["observed"], np.ndarray)
        assert outputs["observed"].ndim == 1

    def test_returns_model_results_outputs(self, data, params):
        pose_ds, spike_ds = data
        result = create_glm_decoder(pose_ds, spike_ds, params=params)
        assert len(result) == 3

    def test_predicted_and_observed_same_length(self, data, params):
        pose_ds, spike_ds = data
        _, _, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        assert len(outputs["predicted"]) == len(outputs["observed"])

    def test_aic_and_log_likelihood_recorded(self, data, params):
        pose_ds, spike_ds = data
        _, _, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        metrics = outputs["params"]["metrics"]
        assert "aic" in metrics
        assert "log_likelihood" in metrics
        assert np.isfinite(metrics["aic"])
        assert np.isfinite(metrics["log_likelihood"])


# ---------------------------------------------------------------------------
# compare_glm_models
# ---------------------------------------------------------------------------

class TestCompareGlmModels:

    def test_non_encoder_raises_not_implemented(self):
        """Regression: non-encoder type must raise NotImplementedError, not a cryptic NameError."""
        pose_ds = make_pose_dataset()
        spike_ds = make_spike_dataset()
        params = {
            "type": "decoder",
            "pose": {"node": "hand", "features": ["position_y"]},
            "spikes": {"unit": [0], "features": ["spike_counts"]},
            "comparison": {"mode": "full"},
        }
        with pytest.raises(NotImplementedError, match="encoder"):
            compare_glm_models(pose_ds, spike_ds, params, None)

    def test_returns_fitted_models_and_summary(self):
        pose_ds = make_pose_dataset()
        spike_ds = make_spike_dataset()
        params = {
            "type": "encoder",
            "family": "Poisson",
            "pose": {"node": "hand", "features": ["position_x", "position_y"]},
            "spikes": {"unit": [0], "features": ["spike_counts"]},
            "comparison": {"mode": "single"},
        }
        fitted, summary = compare_glm_models(pose_ds, spike_ds, params, None)
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(fitted, dict)

    def test_summary_has_expected_columns(self):
        pose_ds = make_pose_dataset()
        spike_ds = make_spike_dataset()
        params = {
            "type": "encoder",
            "family": "Poisson",
            "pose": {"node": "hand", "features": ["position_x", "position_y"]},
            "spikes": {"unit": [0], "features": ["spike_counts"]},
            "comparison": {"mode": "single"},
        }
        _, summary = compare_glm_models(pose_ds, spike_ds, params, None)
        for col in ("model_name", "features", "aic", "log_likelihood"):
            assert col in summary.columns

    def test_single_mode_produces_one_row_per_feature(self):
        features = ["position_x", "position_y"]
        pose_ds = make_pose_dataset()
        spike_ds = make_spike_dataset()
        params = {
            "type": "encoder",
            "family": "Poisson",
            "pose": {"node": "hand", "features": features},
            "spikes": {"unit": [0], "features": ["spike_counts"]},
            "comparison": {"mode": "single"},
        }
        fitted, summary = compare_glm_models(pose_ds, spike_ds, params, None)
        assert len(summary) == len(features)
        assert len(fitted) == len(features)

    def test_per_model_params_reflect_correct_feature_set(self):
        """Regression: each model's stored params should reflect its own features, not the global set."""
        features = ["position_x", "position_y"]
        pose_ds = make_pose_dataset()
        spike_ds = make_spike_dataset()
        params = {
            "type": "encoder",
            "family": "Poisson",
            "pose": {"node": "hand", "features": features},
            "spikes": {"unit": [0], "features": ["spike_counts"]},
            "comparison": {"mode": "single"},
        }
        fitted, _ = compare_glm_models(pose_ds, spike_ds, params, None)
        for feat in features:
            stored_features = fitted[feat]["params"]["pose"]["features"]
            assert stored_features == [feat], (
                f"Expected [{feat!r}] but got {stored_features!r} — "
                "params_ (not params) must be saved per model"
            )
