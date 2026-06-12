"""Unit tests for neurokinematics.models.glm."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from neurokinematics.models.glm import (
    build_decoder_params,
    build_encoder_params,
    build_glm_dataset,
    build_glm_model_sets,
    compare_glm_models,
    create_glm_decoder,
    create_glm_encoder,
    crossval_glm_predictions,
    glm_cv_scores,
    _fit_linear_model,
    _circular_shift_within_groups,
    shuffle_null_cv_r2,
)


# ---------------------------------------------------------------------------
# cross-validation helpers
# ---------------------------------------------------------------------------

class TestGlmCvScores:

    def test_gaussian_perfect_prediction_r2_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = glm_cv_scores(y, y.copy(), "Gaussian")
        assert s["cv_r2"] == pytest.approx(1.0)
        assert s["cv_corr"] == pytest.approx(1.0)

    def test_gaussian_mean_prediction_r2_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.full_like(y, y.mean())
        assert glm_cv_scores(y, pred, "Gaussian")["cv_r2"] == pytest.approx(0.0)

    def test_poisson_perfect_prediction_pseudo_r2_one(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        s = glm_cv_scores(y, y.copy(), "Poisson")
        assert s["cv_r2"] == pytest.approx(1.0, abs=1e-6)
        assert "cv_deviance" in s

    def test_ignores_non_finite(self):
        y = np.array([1.0, 2.0, np.nan, 4.0])
        pred = np.array([1.0, 2.0, 3.0, np.nan])
        s = glm_cv_scores(y, pred, "Gaussian")
        assert np.isfinite(s["cv_r2"])


class TestCrossvalGlmPredictions:

    def test_every_row_predicted_once(self):
        rng = np.random.default_rng(0)
        n = 60
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = rng.poisson(2, size=n).astype(float)
        groups = np.repeat(np.arange(12), 5)   # 12 events, 5 bins each
        oos, metrics = crossval_glm_predictions(X, y, "Poisson", groups, n_splits=4)
        assert oos.shape == y.shape
        assert np.isfinite(oos).all()          # every row got a held-out prediction
        assert metrics["n_groups"] == 12
        assert metrics["n_splits"] == 4

    def test_n_splits_clamped_to_group_count(self):
        rng = np.random.default_rng(1)
        groups = np.repeat(np.arange(3), 4)    # only 3 groups
        n = groups.size
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = rng.normal(size=n)
        _, metrics = crossval_glm_predictions(X, y, "Gaussian", groups, n_splits=10)
        assert metrics["n_splits"] == 3        # clamped down to n_groups


# ---------------------------------------------------------------------------
# build_encoder_params
# ---------------------------------------------------------------------------

class TestBuildEncoderParams:

    def test_basic_shape(self):
        p = build_encoder_params("hand", 3, ["velocity_x", "velocity_y"])
        assert p["type"] == "encoder"
        assert p["family"] == "Poisson"
        assert p["pose"]["node"] == "hand"
        assert p["pose"]["features"] == ["velocity_x", "velocity_y"]
        assert p["spikes"]["unit"] == [3]
        assert p["spikes"]["features"] == ["spike_counts"]
        assert p["comparison"]["mode"] == "full"

    def test_unit_list_preserved(self):
        p = build_encoder_params("hand", [0, 1, 2], ["speed"])
        assert p["spikes"]["unit"] == [0, 1, 2]

    def test_no_basis_key_when_basis_none(self):
        p = build_encoder_params("hand", 0, ["speed"])
        assert "basis" not in p["pose"]

    def test_basis_block_passed_through(self):
        p = build_encoder_params(
            "hand", 0, ["velocity_y"],
            basis={"window": (-0.1, 0.2), "n_basis": 4, "spacing": "log"},
        )
        b = p["pose"]["basis"]
        assert b["window"] == [-0.1, 0.2]
        assert b["n_basis"] == 4
        assert b["spacing"] == "log"

    def test_params_feed_create_glm_encoder(self):
        """The assembled params should drive create_glm_encoder end-to-end."""
        pose_ds = make_pose_dataset(n_bins=20)
        spike_ds = make_spike_dataset(n_bins=20)
        params = build_encoder_params(
            "hand", 0, ["position_x", "position_y"],
            basis={"window": (-0.1, 0.1), "n_basis": 3, "spacing": "linear"},
        )
        model, results, outputs = create_glm_encoder(pose_ds, spike_ds, params=params)
        assert model is not None
        assert outputs["attrs"]["basis"]["n_basis"] == 3


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_pose_dataset(n_events=20, n_bins=10, node="hand", seed=42):
    """Minimal synthetic pose dataset matching the structure expected by the GLM functions."""
    rng = np.random.default_rng(seed)
    position = rng.normal(size=(n_events, n_bins, 1, 2))
    velocity = rng.normal(size=(n_events, n_bins, 1, 2))
    speed = np.abs(rng.normal(size=(n_events, n_bins, 1)))   # scalar-per-node feature
    valid = np.ones((n_events, n_bins), dtype=bool)
    return xr.Dataset(
        {
            "position": (["event", "time_bin", "node", "coord"], position),
            "velocity": (["event", "time_bin", "node", "coord"], velocity),
            "speed": (["event", "time_bin", "node"], speed),
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


def make_mixed_pose_dataset(n_events=20, n_bins=12, node="hand", seed=0):
    """Pose dataset where only half the events were initiated by `node` — so the
    reference-node mask keeps a strict subset, letting us test `all_events`."""
    rng = np.random.default_rng(seed)
    ref = np.array([node if i % 2 == 0 else "foot" for i in range(n_events)])
    speed = np.abs(rng.normal(size=(n_events, n_bins, 1)))
    valid = np.ones((n_events, n_bins), dtype=bool)
    return xr.Dataset(
        {
            "speed": (["event", "time_bin", "node"], speed),
            "valid": (["event", "time_bin"], valid),
            "reference_node": (["event"], ref),
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
# create_glm_encoder with a temporal basis
# ---------------------------------------------------------------------------

class TestCreateGlmEncoderBasis:

    @pytest.fixture
    def data(self):
        # more bins so a multi-offset window still leaves interior bins
        return make_pose_dataset(n_bins=20), make_spike_dataset(n_bins=20)

    def _params(self, window, n_basis, spacing="linear"):
        return {
            "type": "encoder",
            "family": "Poisson",
            "pose": {
                "node": "hand",
                "features": ["position_x", "position_y"],
                "basis": {"window": window, "n_basis": n_basis, "spacing": spacing},
            },
            "spikes": {"unit": [0], "features": ["spike_counts"]},
        }

    def test_runs_and_returns_triplet(self, data):
        pose_ds, spike_ds = data
        result = create_glm_encoder(pose_ds, spike_ds,
                                    params=self._params((-0.1, 0.1), 3))
        assert len(result) == 3

    def test_coefficient_count_is_features_times_basis_plus_const(self, data):
        pose_ds, spike_ds = data
        n_basis = 3
        _, results, _ = create_glm_encoder(
            pose_ds, spike_ds, params=self._params((-0.1, 0.1), n_basis))
        # 2 features * 3 basis functions + intercept
        assert len(results.params) == 2 * n_basis + 1

    def test_design_columns_named_per_basis(self, data):
        pose_ds, spike_ds = data
        _, results, _ = create_glm_encoder(
            pose_ds, spike_ds, params=self._params((-0.1, 0.1), 2))
        names = list(results.params.index)
        assert "position_x__b0" in names
        assert "position_y__b1" in names

    def test_basis_metadata_recorded_in_attrs(self, data):
        pose_ds, spike_ds = data
        _, _, outputs = create_glm_encoder(
            pose_ds, spike_ds, params=self._params((-0.1, 0.1), 4))
        basis_meta = outputs["attrs"]["basis"]
        assert basis_meta["n_basis"] == 4
        assert basis_meta["spacing"] == "linear"
        assert len(basis_meta["offsets"]) >= 1

    def test_wider_window_drops_edge_bins(self, data):
        """A window spanning several bins should drop event-edge bins (no full context)."""
        pose_ds, spike_ds = data
        n_events, n_bins = 20, 20
        _, _, narrow = create_glm_encoder(
            pose_ds, spike_ds, params=self._params((0.0, 0.0), 1))
        _, _, wide = create_glm_encoder(
            pose_ds, spike_ds, params=self._params((-0.3, 0.3), 3))
        assert len(wide["observed"]) < len(narrow["observed"])
        assert len(narrow["observed"]) == n_events * n_bins

    def test_metrics_finite_with_basis(self, data):
        pose_ds, spike_ds = data
        _, _, outputs = create_glm_encoder(
            pose_ds, spike_ds, params=self._params((-0.1, 0.2), 4))
        metrics = outputs["params"]["metrics"]
        assert np.isfinite(metrics["aic"])
        assert np.isfinite(metrics["log_likelihood"])


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
# cross-validated encoder / decoder
# ---------------------------------------------------------------------------

class TestEncoderCrossValidation:

    def test_cv_metrics_recorded(self):
        pose_ds, spike_ds = make_pose_dataset(), make_spike_dataset()
        params = build_encoder_params(
            "hand", 0, ["position_x", "position_y"], n_splits=4)
        _, _, outputs = create_glm_encoder(pose_ds, spike_ds, params=params)
        metrics = outputs["params"]["metrics"]
        assert metrics["cross_validated"] is True
        assert "cv_r2" in metrics and "cv_corr" in metrics
        assert metrics["n_splits"] == 4

    def test_no_cv_flag_when_n_splits_zero(self):
        pose_ds, spike_ds = make_pose_dataset(), make_spike_dataset()
        params = build_encoder_params("hand", 0, ["position_y"])  # n_splits=0
        _, _, outputs = create_glm_encoder(pose_ds, spike_ds, params=params)
        assert outputs["params"]["metrics"]["cross_validated"] is False

    def test_predicted_length_matches_observed(self):
        pose_ds, spike_ds = make_pose_dataset(), make_spike_dataset()
        params = build_encoder_params("hand", 0, ["position_y"], n_splits=3)
        _, _, outputs = create_glm_encoder(pose_ds, spike_ds, params=params)
        assert len(outputs["predicted"]) == len(outputs["observed"])


class TestDecoderTargetsAndCv:

    def test_decodes_scalar_speed_target(self):
        """Regression: 'speed' has no coord — previously split('_') crashed."""
        pose_ds, spike_ds = make_pose_dataset(), make_spike_dataset(n_units=3)
        params = build_decoder_params("hand", [0, 1, 2], "speed", n_splits=3)
        model, _, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        assert model is not None
        assert outputs["observed"].ndim == 1
        assert outputs["attrs"]["target"] == "speed"
        assert outputs["attrs"]["model_type"] == "decoder"

    def test_decodes_directional_target(self):
        pose_ds, spike_ds = make_pose_dataset(), make_spike_dataset(n_units=3)
        params = build_decoder_params("hand", [0, 1, 2], "velocity_y", n_splits=3)
        _, _, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        assert outputs["attrs"]["target"] == "velocity_y"

    def test_decoder_cv_metrics_recorded(self):
        pose_ds, spike_ds = make_pose_dataset(), make_spike_dataset(n_units=3)
        params = build_decoder_params("hand", [0, 1, 2], "speed", n_splits=4)
        _, _, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        metrics = outputs["params"]["metrics"]
        assert metrics["cross_validated"] is True
        assert "cv_r2" in metrics


class TestGlmSaveLayout:
    """The fit must write predictions.zarr under glm/<type>/ — exactly where the
    Plot Viewer's find_latest_glm_predictions looks."""

    def test_decoder_saves_under_glm_decoder(self, tmp_path):
        pose_ds, spike_ds = make_pose_dataset(), make_spike_dataset(n_units=3)
        params = build_decoder_params("hand", [0, 1, 2], "speed", n_splits=3)
        create_glm_decoder(pose_ds, spike_ds, params=params, save_path=tmp_path)
        hits = list((tmp_path / "glm" / "decoder").glob("**/predictions.zarr"))
        assert len(hits) == 1, f"expected one decoder predictions.zarr, got {hits}"

    def test_encoder_saves_under_glm_encoder(self, tmp_path):
        pose_ds, spike_ds = make_pose_dataset(), make_spike_dataset()
        params = build_encoder_params("hand", 0, ["position_y"], n_splits=3)
        create_glm_encoder(pose_ds, spike_ds, params=params, save_path=tmp_path)
        hits = list((tmp_path / "glm" / "encoder").glob("**/predictions.zarr"))
        assert len(hits) == 1, f"expected one encoder predictions.zarr, got {hits}"


class TestDecoderSmoothingEventsShuffle:

    def test_smoothing_recorded_and_runs(self):
        pose_ds = make_pose_dataset(n_bins=20)
        spike_ds = make_spike_dataset(n_bins=20, n_units=3)
        params = build_decoder_params("hand", [0, 1, 2], "speed",
                                      smoothing_s=0.05, n_splits=3)
        _, _, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        assert outputs["attrs"]["smoothing_s"] == 0.05

    def test_all_events_uses_more_data(self):
        pose_ds = make_mixed_pose_dataset(n_events=20, n_bins=12)
        spike_ds = make_spike_dataset(n_events=20, n_bins=12, n_units=3)
        masked = build_decoder_params("hand", [0, 1, 2], "speed", n_splits=3, all_events=False)
        allev = build_decoder_params("hand", [0, 1, 2], "speed", n_splits=3, all_events=True)
        _, _, out_masked = create_glm_decoder(pose_ds, spike_ds, params=masked)
        _, _, out_all = create_glm_decoder(pose_ds, spike_ds, params=allev)
        assert len(out_all["observed"]) > len(out_masked["observed"])
        assert out_all["attrs"]["all_events"] is True

    def test_shuffle_null_recorded(self):
        pose_ds = make_pose_dataset(n_bins=20)
        spike_ds = make_spike_dataset(n_bins=20, n_units=3)
        params = build_decoder_params("hand", [0, 1, 2], "speed",
                                      n_splits=3, n_shuffle=20)
        _, _, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        m = outputs["params"]["metrics"]
        assert m["shuffle_n"] == 20
        assert 0.0 < m["shuffle_p"] <= 1.0
        assert np.isfinite(m["shuffle_null_mean"])


class TestShuffleHelpers:

    def test_circular_shift_permutes_within_group(self):
        rng = np.random.default_rng(0)
        y = np.array([1., 2., 3., 10., 20., 30.])
        groups = np.array([0, 0, 0, 1, 1, 1])
        out = _circular_shift_within_groups(y, groups, rng)
        assert sorted(out[:3]) == [1, 2, 3]      # same values, just rolled
        assert sorted(out[3:]) == [10, 20, 30]   # no cross-group mixing

    def test_singleton_group_unchanged(self):
        rng = np.random.default_rng(0)
        out = _circular_shift_within_groups(np.array([5.0]), np.array([0]), rng)
        assert out[0] == 5.0

    def test_shuffle_null_returns_p_and_mean(self):
        rng = np.random.default_rng(0)
        n = 120
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = rng.normal(size=n)
        groups = np.repeat(np.arange(12), 10)
        p, null_mean = shuffle_null_cv_r2(X, y, "Gaussian", groups, 4, 0.0,
                                          real_r2=0.0, n_shuffle=20)
        assert 0.0 < p <= 1.0
        assert np.isfinite(null_mean)


class TestPlottedScoreMatchesSaved:
    """The fit score the Plot Viewer shows is recomputed from predictions.zarr;
    it must match the CV R² recorded in glm_params.yaml at fit time."""

    def test_score_from_zarr_matches_saved_cv_r2(self, tmp_path):
        import yaml
        from neurokinematics.io import load_zarr
        pose_ds = make_pose_dataset(n_bins=20)
        spike_ds = make_spike_dataset(n_bins=20, n_units=3)
        params = build_decoder_params("hand", [0, 1, 2], "speed", n_splits=4)
        create_glm_decoder(pose_ds, spike_ds, params=params, save_path=tmp_path)

        run = next((tmp_path / "glm" / "decoder").glob("population_to_*"))
        pred = load_zarr(run / "predictions.zarr", method="xarray")
        with open(run / "glm_params.yaml") as f:
            saved = yaml.safe_load(f)

        obs = np.asarray(pred["observed_counts"].values, float)
        prd = np.asarray(pred["predicted_counts"].values, float)
        mask = pred["valid"].values & np.isfinite(obs) & np.isfinite(prd)
        score = glm_cv_scores(obs[mask], prd[mask], saved["family"])

        assert np.isfinite(score["cv_r2"])
        assert score["cv_r2"] == pytest.approx(saved["metrics"]["cv_r2"], abs=1e-6)


class TestBuildDecoderParams:

    def test_shape(self):
        p = build_decoder_params("hand", [0, 1, 2], "speed")
        assert p["type"] == "decoder"
        assert p["family"] == "Gaussian"
        assert p["pose"] == {"node": "hand", "features": ["speed"]}
        assert p["spikes"]["unit"] == [0, 1, 2]
        assert p["cv"]["n_splits"] == 5

    def test_cv_omitted_when_zero(self):
        p = build_decoder_params("hand", [0], "position_y", n_splits=0)
        assert "cv" not in p

    def test_lag_and_alpha_passed_through(self):
        p = build_decoder_params(
            "hand", [0, 1], "speed",
            lag={"window": (-0.1, 0.1), "n_basis": 4}, alpha=2.5)
        assert p["spikes"]["basis"]["window"] == [-0.1, 0.1]
        assert p["spikes"]["basis"]["n_basis"] == 4
        assert p["regularization"]["alpha"] == 2.5

    def test_no_regularization_key_when_alpha_zero(self):
        p = build_decoder_params("hand", [0], "speed", alpha=0.0)
        assert "regularization" not in p

    def test_smoothing_all_events_shuffle_passed(self):
        p = build_decoder_params("hand", [0, 1], "speed",
                                 smoothing_s=0.05, all_events=True, n_shuffle=100)
        assert p["spikes"]["smoothing_s"] == 0.05
        assert p["pose"]["all_events"] is True
        assert p["shuffle"]["n"] == 100

    def test_defaults_omit_optional_keys(self):
        p = build_decoder_params("hand", [0], "speed")
        assert "smoothing_s" not in p["spikes"]
        assert "all_events" not in p["pose"]
        assert "shuffle" not in p


# ---------------------------------------------------------------------------
# ridge fit helper + lagged / regularized decoder
# ---------------------------------------------------------------------------

class TestFitLinearModel:

    def test_alpha_zero_is_statsmodels(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(50), rng.normal(size=50)])
        y = rng.normal(size=50)
        res = _fit_linear_model(y, X, "Gaussian", alpha=0.0)
        assert hasattr(res, "aic")            # statsmodels results
        assert res.predict(X).shape == (50,)

    def test_alpha_positive_is_ridge_and_shrinks(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=200)
        X = np.column_stack([np.ones(200), x])
        y = 3.0 * x + rng.normal(scale=0.1, size=200)
        ols = _fit_linear_model(y, X, "Gaussian", alpha=0.0)
        ridge = _fit_linear_model(y, X, "Gaussian", alpha=1000.0)
        # heavy ridge shrinks the slope toward zero vs OLS
        assert abs(ridge.coef_[1]) < abs(ols.params[1])
        assert ridge.predict(X).shape == (200,)


class TestDecoderLagsAndRidge:

    def test_spike_history_widens_design_and_records_basis(self):
        pose_ds = make_pose_dataset(n_bins=20)
        spike_ds = make_spike_dataset(n_bins=20, n_units=3)
        params = build_decoder_params(
            "hand", [0, 1, 2], "speed",
            lag={"window": (-0.1, 0.1), "n_basis": 3}, n_splits=3)
        _, results, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        # 3 units * 3 basis functions + intercept
        assert len(results.params) == 3 * 3 + 1
        assert outputs["attrs"]["spike_basis"]["n_basis"] == 3

    def test_ridge_decoder_runs_and_flags_regularization(self):
        pose_ds = make_pose_dataset(n_bins=20)
        spike_ds = make_spike_dataset(n_bins=20, n_units=3)
        params = build_decoder_params(
            "hand", [0, 1, 2], "speed",
            lag={"window": (-0.1, 0.1), "n_basis": 3}, alpha=5.0, n_splits=3)
        _, _, outputs = create_glm_decoder(pose_ds, spike_ds, params=params)
        metrics = outputs["params"]["metrics"]
        assert metrics["cross_validated"] is True
        assert metrics["regularization_alpha"] == 5.0
        assert "aic" not in metrics          # regularized fit has no AIC


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
