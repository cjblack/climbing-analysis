"""Unit tests for neurokinematics.ephys.spikes.modulation.event_modulation."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from neurokinematics.ephys.spikes.modulation import event_modulation


# ---------------------------------------------------------------------------
# synthetic raster df matching get_movement_aligned_rasters output
# ---------------------------------------------------------------------------

def make_rasters_df(
    *,
    units=(0, 1),
    nodes=("l_forepaw", "r_forepaw"),
    epochs=("start", "max", "end"),
    n_events=30,
    baseline_rate=10.0,
    window=(-0.5, 0.5),
    seed=0,
    modulated=None,
):
    """Build a rasters_df. Homogeneous-Poisson spikes per event at ``baseline_rate``,
    except cells named in ``modulated`` get extra spikes injected into [0, 0.2] s.

    Args:
        modulated (dict | None): ``{(unit, node, epoch): added_rate_hz}`` — a
            positive value adds a post-event firing burst, a negative value
            deletes post-event spikes (suppression). None = all flat.
    """
    rng = np.random.default_rng(seed)
    modulated = modulated or {}
    dur = window[1] - window[0]
    rows = []
    for uid in units:
        for nd in nodes:
            for ep in epochs:
                add = modulated.get((uid, nd, ep), 0.0)
                for _ in range(n_events):
                    n = rng.poisson(baseline_rate * dur)
                    spikes = list(rng.uniform(window[0], window[1], size=n))
                    if add > 0:
                        n_extra = rng.poisson(add * 0.2)  # response window is 0.2 s
                        spikes += list(rng.uniform(0.0, 0.2, size=n_extra))
                    elif add < 0:
                        spikes = [s for s in spikes
                                  if not (0.0 <= s < 0.2 and rng.random() < min(1.0, -add / baseline_rate))]
                    rows.append({
                        "unit_id": uid,
                        "node": nd,
                        "movement_event": ep,
                        "trial": 0,
                        "event_time_ts": 0.0,
                        "spike_raster": np.asarray(sorted(spikes)),
                    })
    return pd.DataFrame(rows)


def make_ramp_df(
    *, unit=0, node="l_forepaw", epoch="start", n_events=60,
    r0=5.0, r1=45.0, window=(-0.5, 0.5), seed=0,
):
    """One cell whose firing ramps *linearly* from ``r0`` (window start) to ``r1``
    (window end) — a smooth ramp with no event-locked transient at t=0. Used to
    check that ``detrend_baseline`` stops a pure ramp reading as modulation.
    """
    rng = np.random.default_rng(seed)
    edges = np.linspace(window[0], window[1], 101)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    rates = r0 + (r1 - r0) * (centers - window[0]) / (window[1] - window[0])
    rows = []
    for _ in range(n_events):
        spikes = []
        for c, rate in zip(centers, rates):
            k = rng.poisson(rate * width)
            if k:
                spikes += list(rng.uniform(c - width / 2, c + width / 2, size=k))
        rows.append({
            "unit_id": unit, "node": node, "movement_event": epoch,
            "trial": 0, "event_time_ts": 0.0,
            "spike_raster": np.asarray(sorted(spikes)),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def flat_df():
    return make_rasters_df(seed=1)


# ---------------------------------------------------------------------------
# structure / contract
# ---------------------------------------------------------------------------

class TestStructure:

    def test_returns_dataset_with_expected_dims(self, flat_df):
        mod = event_modulation(flat_df, n_shuffle=0)
        assert isinstance(mod, xr.Dataset)
        assert set(mod.dims) == {"unit", "node", "epoch", "time_bin"}

    def test_coords_match_inputs(self, flat_df):
        mod = event_modulation(flat_df, n_shuffle=0)
        assert list(mod.coords["unit"].values) == [0, 1]
        assert list(mod.coords["epoch"].values) == ["end", "max", "start"]

    def test_expected_variables_present(self, flat_df):
        mod = event_modulation(flat_df, n_shuffle=0)
        for v in ("psth_hz", "psth_z", "modulation", "response_z",
                  "baseline_rate", "n_events", "p_value", "p_fdr", "significant"):
            assert v in mod

    def test_missing_column_raises(self):
        df = pd.DataFrame({"unit_id": [0], "node": ["x"], "movement_event": ["start"]})
        with pytest.raises(ValueError, match="missing required columns"):
            event_modulation(df)

    def test_baseline_rate_recovers_truth(self, flat_df):
        """Baseline-window mean rate should sit near the simulated 10 Hz."""
        mod = event_modulation(flat_df, n_shuffle=0)
        assert mod["baseline_rate"].mean().item() == pytest.approx(10.0, abs=2.5)


# ---------------------------------------------------------------------------
# the statistics actually fire
# ---------------------------------------------------------------------------

class TestDetection:

    def test_injected_increase_is_significant_and_positive(self):
        df = make_rasters_df(seed=2, modulated={(0, "l_forepaw", "start"): 60.0})
        mod = event_modulation(df, n_shuffle=500, seed=0)
        cell = mod.sel(unit=0, node="l_forepaw", epoch="start")
        assert cell["modulation"].item() > 0
        assert cell["p_value"].item() < 0.05
        assert bool(cell["significant"].item())

    def test_injected_suppression_is_significant_and_negative(self):
        df = make_rasters_df(seed=3, baseline_rate=20.0,
                             modulated={(1, "r_forepaw", "end"): -20.0})
        mod = event_modulation(df, n_shuffle=500, seed=0)
        cell = mod.sel(unit=1, node="r_forepaw", epoch="end")
        assert cell["modulation"].item() < 0
        assert cell["p_value"].item() < 0.05

    def test_flat_unit_mostly_not_significant(self, flat_df):
        """A purely flat dataset should rarely cross the FDR threshold."""
        mod = event_modulation(flat_df, n_shuffle=500, seed=0)
        assert mod["significant"].sum().item() == 0

    def test_two_sided_p_is_bounded(self, flat_df):
        mod = event_modulation(flat_df, n_shuffle=200, seed=0)
        p = mod["p_value"].values
        p = p[np.isfinite(p)]
        assert np.all((p > 0) & (p <= 1.0))


# ---------------------------------------------------------------------------
# baseline detrending (separate event-locked transients from slow ramps)
# ---------------------------------------------------------------------------

class TestDetrend:

    def test_pure_ramp_flagged_without_detrend_but_not_with(self):
        """A smooth ramp reads as modulation under the level test, but baseline
        detrending (extrapolating the pre-event trend) removes it."""
        df = make_ramp_df(seed=5)
        cell = dict(unit=0, node="l_forepaw", epoch="start")
        raw = event_modulation(df, n_shuffle=800, seed=0,
                               progress=False).sel(**cell)
        det = event_modulation(df, n_shuffle=800, seed=0, detrend_baseline=True,
                               progress=False).sel(**cell)
        assert raw["modulation"].item() > 0
        assert raw["p_value"].item() < 0.05
        assert abs(det["modulation"].item()) < abs(raw["modulation"].item())
        assert det["p_value"].item() > 0.05

    def test_transient_survives_detrend(self):
        """A flat-baseline post-event burst is still detected with detrend on."""
        df = make_rasters_df(units=(0, 1), n_events=40, baseline_rate=20.0, seed=6,
                             modulated={(0, "l_forepaw", "start"): 80.0})
        det = event_modulation(df, n_shuffle=800, seed=0, detrend_baseline=True,
                               progress=False)
        cell = det.sel(unit=0, node="l_forepaw", epoch="start")
        assert cell["modulation"].item() > 0
        assert cell["p_value"].item() < 0.05

    def test_detrend_recorded_in_attrs(self, flat_df):
        assert event_modulation(flat_df, n_shuffle=0).attrs["detrend_baseline"] is False
        assert event_modulation(flat_df, n_shuffle=0,
                                detrend_baseline=True).attrs["detrend_baseline"] is True


# ---------------------------------------------------------------------------
# windows / sparsity handling
# ---------------------------------------------------------------------------

class TestWindowsAndSparsity:

    def test_no_shuffle_leaves_pvalue_nan(self, flat_df):
        mod = event_modulation(flat_df, n_shuffle=0)
        assert np.isnan(mod["p_value"].values).all()

    def test_min_events_filters_sparse_cells(self):
        df = make_rasters_df(n_events=2, seed=4)
        mod = event_modulation(df, n_shuffle=0, min_events=3)
        assert np.isnan(mod["modulation"].values).all()
        assert (mod["n_events"].values == 2).all()

    def test_window_out_of_range_raises(self, flat_df):
        with pytest.raises(ValueError, match="cover >=1 bin"):
            event_modulation(flat_df, n_shuffle=0, response_window=(5.0, 6.0))

    def test_psth_bins_span_window(self, flat_df):
        mod = event_modulation(flat_df, n_shuffle=0, window=(-0.5, 0.5), bin_size=0.02)
        assert mod.dims["time_bin"] == 50


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

class TestSave:

    def test_writes_zarr(self, flat_df, tmp_path):
        event_modulation(flat_df, n_shuffle=0, save_path=tmp_path)
        assert (tmp_path / "event_modulation.zarr").exists()

    def test_custom_filename(self, flat_df, tmp_path):
        """A timestamped filename lets repeated runs coexist without overwriting."""
        name = "event_modulation_20260625_120000.zarr"
        event_modulation(flat_df, n_shuffle=0, save_path=tmp_path, filename=name)
        assert (tmp_path / name).exists()
        assert not (tmp_path / "event_modulation.zarr").exists()


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

class TestPlots:

    @pytest.fixture(scope="class")
    def mod(self, flat_df):
        return event_modulation(flat_df, n_shuffle=0)

    def test_event_modulation_heatmap_axes(self, mod):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_event_modulation
        fig = plot_event_modulation(mod, node="l_forepaw")
        # one axes per epoch (+ colorbar axes)
        assert len(fig.axes) >= len(mod.epoch)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels and all(s != "" for s in labels)   # rows labelled with unit ids

    def test_event_modulation_graded_stars(self):
        """Heatmap marks strongly-modulated cells with ** (gated by FDR sig)."""
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_event_modulation
        mod = {(0, "l_forepaw", "start"): 80.0}
        df = make_rasters_df(units=(0, 1), nodes=("l_forepaw", "r_forepaw"),
                             n_events=40, baseline_rate=20.0, seed=21, modulated=mod)
        m = event_modulation(df, n_shuffle=1000, seed=0, progress=False)
        fig = plot_event_modulation(m, node="l_forepaw")
        # gather star text drawn on the axes
        marks = {txt.get_text() for ax in fig.axes for txt in ax.texts}
        assert "**" in marks or "*" in marks

    def test_unit_limb_tuning_grid_default(self, mod):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_unit_limb_tuning
        unit = int(mod.unit.values[0])
        fig = plot_unit_limb_tuning(mod, unit)                 # default kind='grid'
        assert len(fig.axes) == len(mod.node) * len(mod.epoch)  # one panel per limb x epoch

    def test_unit_limb_tuning_overlay_axes_and_columns(self, mod):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_unit_limb_tuning
        unit = int(mod.unit.values[0])
        fig = plot_unit_limb_tuning(mod, unit, kind="overlay")
        assert len(fig.axes) == len(mod.epoch)                 # one column per epoch
        assert fig.axes[0].get_ylabel() == "firing rate (Hz)"

    def test_unit_limb_tuning_paired_segments_x_epochs(self):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_unit_limb_tuning
        # needs all four limbs so there are two segments (fore/hind)
        df = make_rasters_df(nodes=("l_forepaw", "r_forepaw", "l_hindpaw", "r_hindpaw"))
        m = event_modulation(df, n_shuffle=0)
        fig = plot_unit_limb_tuning(m, int(m.unit.values[0]), kind="paired")
        assert len(fig.axes) == 2 * len(m.epoch)   # 2 segments x n_epochs

    def test_unit_limb_tuning_heatmap_rows_are_limbs(self, mod):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_unit_limb_tuning
        unit = int(mod.unit.values[0])
        fig = plot_unit_limb_tuning(mod, unit, kind="heatmap", feature="psth_z")
        labels = [tl.get_text() for tl in fig.axes[0].get_yticklabels()]
        assert set(str(n) for n in mod.node.values) <= set(labels)

    def test_unit_limb_tuning_bad_kind_raises(self, mod):
        from neurokinematics.ephys.spikes.plotting import plot_unit_limb_tuning
        with pytest.raises(ValueError, match="kind must be"):
            plot_unit_limb_tuning(mod, int(mod.unit.values[0]), kind="nope")

    def test_p_stars_thresholds(self):
        from neurokinematics.ephys.spikes.plotting import _p_stars
        assert _p_stars(0.0005) == "**"      # p < 0.001
        assert _p_stars(0.02) == "*"         # 0.001 <= p < 0.05
        assert _p_stars(0.2) == ""           # n.s.
        assert _p_stars(float("nan")) == ""
        assert _p_stars(None) == ""

    def test_unit_limb_tuning_graded_stars_render(self, mod):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_unit_limb_tuning
        # should run for every layout with the p-value-driven stars
        for kind in ("grid", "overlay", "heatmap"):
            fig = plot_unit_limb_tuning(mod, int(mod.unit.values[0]), kind=kind)
            assert fig is not None


# ---------------------------------------------------------------------------
# laterality (ipsi vs contra)
# ---------------------------------------------------------------------------

FOUR_LIMBS = ("l_forepaw", "r_forepaw", "l_hindpaw", "r_hindpaw")


class TestLaterality:

    def test_structure_and_inference(self):
        from neurokinematics.ephys.spikes.laterality import laterality
        df = make_rasters_df(nodes=FOUR_LIMBS, n_events=20, seed=11)
        m = event_modulation(df, n_shuffle=0)
        lat = laterality(m, epoch="start")
        assert len(lat) == len(m.unit)
        for col in ("unit", "ipsi", "contra", "LI", "ipsi_sig", "contra_sig",
                    "bilateral", "pattern", "ipsi_up_contra_down"):
            assert col in lat.columns
        assert lat.attrs["ipsi_nodes"] == ["l_forepaw", "l_hindpaw"]
        assert lat.attrs["contra_nodes"] == ["r_forepaw", "r_hindpaw"]

    def test_detects_opponent_pattern(self):
        """Ipsi-up / contra-down → bilateral, opponent, ipsi_up_contra_down."""
        from neurokinematics.ephys.spikes.laterality import laterality
        mod = {
            (0, "l_forepaw", "start"): 55.0, (0, "l_hindpaw", "start"): 55.0,
            (0, "r_forepaw", "start"): -25.0, (0, "r_hindpaw", "start"): -25.0,
        }
        df = make_rasters_df(units=(0, 1), nodes=FOUR_LIMBS, n_events=40,
                             baseline_rate=30.0, seed=12, modulated=mod)
        m = event_modulation(df, n_shuffle=500, seed=0, progress=False)
        lat = laterality(m, epoch="start").set_index("unit")
        assert bool(lat.loc[0, "bilateral"])
        assert lat.loc[0, "pattern"] == "bilateral_opponent"
        assert bool(lat.loc[0, "ipsi_up_contra_down"])
        assert lat.loc[0, "ipsi"] > 0 and lat.loc[0, "contra"] < 0
        assert not bool(lat.loc[1, "bilateral"])          # flat unit

    def test_detects_congruent_pattern(self):
        """Both sides increasing → bilateral, congruent, NOT ipsi_up_contra_down."""
        from neurokinematics.ephys.spikes.laterality import laterality
        mod = {
            (0, "l_forepaw", "start"): 45.0, (0, "l_hindpaw", "start"): 45.0,
            (0, "r_forepaw", "start"): 45.0, (0, "r_hindpaw", "start"): 45.0,
        }
        df = make_rasters_df(units=(0, 1), nodes=FOUR_LIMBS, n_events=40,
                             baseline_rate=20.0, seed=13, modulated=mod)
        m = event_modulation(df, n_shuffle=500, seed=0, progress=False)
        lat = laterality(m, epoch="start").set_index("unit")
        assert bool(lat.loc[0, "bilateral"])
        assert lat.loc[0, "pattern"] == "bilateral_congruent"
        assert not bool(lat.loc[0, "ipsi_up_contra_down"])

    def test_missing_sides_raises(self):
        from neurokinematics.ephys.spikes.laterality import laterality
        df = make_rasters_df(nodes=("l_forepaw", "l_hindpaw"), seed=3)  # no right limbs
        m = event_modulation(df, n_shuffle=0)
        with pytest.raises(ValueError, match="ipsi/contra"):
            laterality(m, epoch="start")

    def test_units_subset(self):
        from neurokinematics.ephys.spikes.laterality import laterality
        df = make_rasters_df(units=(0, 1, 2, 3), nodes=FOUR_LIMBS, n_events=20, seed=8)
        m = event_modulation(df, n_shuffle=0)
        lat = laterality(m, epoch="start", units=[1, 3])
        assert sorted(lat["unit"].tolist()) == [1, 3]

    def test_plot_laterality_two_panels(self):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_laterality
        df = make_rasters_df(nodes=FOUR_LIMBS, n_events=20, seed=7)
        m = event_modulation(df, n_shuffle=0)
        fig = plot_laterality(m, epoch="start")
        assert len(fig.axes) == 2

    def test_plot_laterality_uniform_marker_size(self):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_laterality
        df = make_rasters_df(units=(0, 1, 2, 3), nodes=FOUR_LIMBS, n_events=20, seed=9)
        m = event_modulation(df, n_shuffle=0)
        fig = plot_laterality(m, epoch="start")
        # every scatter collection on the scatter axes uses one identical size
        sizes = set()
        for coll in fig.axes[0].collections:
            sizes.update(coll.get_sizes().tolist())
        assert len(sizes) == 1

    def test_plot_laterality_units_subset(self):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_laterality
        df = make_rasters_df(units=(0, 1, 2, 3), nodes=FOUR_LIMBS, n_events=20, seed=7)
        m = event_modulation(df, n_shuffle=0)
        fig = plot_laterality(m, epoch="start", units=[0, 2])
        n_points = sum(len(c.get_offsets()) for c in fig.axes[0].collections)
        assert n_points == 2


# ---------------------------------------------------------------------------
# laterality across sessions
# ---------------------------------------------------------------------------

class TestLateralityAcrossSessions:

    def _two_session_mods(self):
        m1 = event_modulation(make_rasters_df(units=(0, 1), nodes=FOUR_LIMBS, seed=1),
                              n_shuffle=0)
        m2 = event_modulation(make_rasters_df(units=(0, 1), nodes=FOUR_LIMBS, seed=2),
                              n_shuffle=0)
        return {"day1": m1, "day2": m2}

    def test_concatenates_with_session_column(self):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        pooled = laterality_across_sessions(self._two_session_mods(), epoch="start")
        assert "session" in pooled.columns
        assert sorted(pooled["session"].unique()) == ["day1", "day2"]
        assert len(pooled) == 4                     # 2 units x 2 sessions
        assert pooled.attrs["n_sessions"] == 2

    def test_accepts_pair_iterable(self):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        mods = list(self._two_session_mods().items())
        pooled = laterality_across_sessions(mods, epoch="start")
        assert pooled["session"].nunique() == 2

    def test_select_list_filters_sessions(self):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        pooled = laterality_across_sessions(self._two_session_mods(),
                                            select=["day2"], epoch="start")
        assert pooled["session"].unique().tolist() == ["day2"]

    def test_select_callable_filters_sessions(self):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        pooled = laterality_across_sessions(self._two_session_mods(),
                                            select=lambda s: s.endswith("1"), epoch="start")
        assert pooled["session"].unique().tolist() == ["day1"]

    def test_select_all_filtered_out_raises(self):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        with pytest.raises(ValueError, match="No sessions"):
            laterality_across_sessions(self._two_session_mods(), select=["nope"], epoch="start")

    def test_good_units_mapping_filters_units(self):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        # event_modulation here builds units 0 and 1; keep only unit 1 per session
        pooled = laterality_across_sessions(
            self._two_session_mods(), good_units={"day1": [1], "day2": [1]}, epoch="start")
        assert pooled["unit"].unique().tolist() == [1]

    def test_good_units_callable(self):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        pooled = laterality_across_sessions(
            self._two_session_mods(), good_units=lambda s: [0], epoch="start")
        assert pooled["unit"].unique().tolist() == [0]

    def test_good_units_auto_without_phy_raises(self):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        with pytest.raises(ValueError, match="cluster_group"):
            laterality_across_sessions(self._two_session_mods(), good_units="auto", epoch="start")


    def test_empty_raises(self):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        with pytest.raises(ValueError, match="No sessions"):
            laterality_across_sessions({}, epoch="start")

    def test_plot_laterality_accepts_pooled_table(self):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        from neurokinematics.ephys.spikes.plotting import plot_laterality
        pooled = laterality_across_sessions(self._two_session_mods(), epoch="start")
        fig = plot_laterality(pooled)              # DataFrame path
        assert len(fig.axes) == 2

    def test_stability_bar(self):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        from neurokinematics.ephys.spikes.plotting import plot_laterality_stability
        pooled = laterality_across_sessions(self._two_session_mods(), epoch="start")
        fig = plot_laterality_stability(pooled, normalize=True)
        assert fig is not None
        # two sessions on the x-axis
        assert len(fig.axes[0].get_xticks()) == 2

    def test_stability_requires_session_column(self):
        from neurokinematics.ephys.spikes.plotting import plot_laterality_stability
        from neurokinematics.ephys.spikes.laterality import laterality
        m = event_modulation(make_rasters_df(nodes=FOUR_LIMBS), n_shuffle=0)
        lat = laterality(m, epoch="start")         # no 'session' column
        with pytest.raises(ValueError, match="session"):
            plot_laterality_stability(lat)


class TestLateralityEpochs:

    def test_rows_per_epoch_from_dataset(self):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_laterality_epochs
        m = event_modulation(make_rasters_df(nodes=FOUR_LIMBS), n_shuffle=0)
        fig = plot_laterality_epochs(m, epochs=("start", "max", "end"))
        assert len(fig.axes) == 3 * 2               # 3 epoch rows x (scatter + hist)

    def test_accepts_per_epoch_tables(self):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        from neurokinematics.ephys.spikes.plotting import plot_laterality_epochs
        mods = {"d1": event_modulation(make_rasters_df(nodes=FOUR_LIMBS, seed=1), n_shuffle=0),
                "d2": event_modulation(make_rasters_df(nodes=FOUR_LIMBS, seed=2), n_shuffle=0)}
        tables = {ep: laterality_across_sessions(mods, epoch=ep) for ep in ("start", "max", "end")}
        fig = plot_laterality_epochs(tables)        # pooled per-epoch tables
        assert len(fig.axes) == 6

    def test_single_dataframe_raises(self):
        from neurokinematics.ephys.spikes.plotting import plot_laterality_epochs
        from neurokinematics.ephys.spikes.laterality import laterality
        m = event_modulation(make_rasters_df(nodes=FOUR_LIMBS), n_shuffle=0)
        with pytest.raises(ValueError, match="one epoch"):
            plot_laterality_epochs(laterality(m, epoch="start"))


# ---------------------------------------------------------------------------
# laterality_stats (session-level summary + tests)
# ---------------------------------------------------------------------------

class TestLateralityStats:

    @staticmethod
    def _epoch_table(n_opp, n_con, sessions=("s1", "s2", "s3", "s4")):
        """Minimal pooled laterality table: only the columns laterality_stats reads."""
        rows = []
        for s in sessions:
            for _ in range(n_opp):
                rows.append({"session": s, "pattern": "bilateral_opponent", "LI": 0.8})
            for _ in range(n_con):
                rows.append({"session": s, "pattern": "bilateral_congruent", "LI": 0.0})
            rows.append({"session": s, "pattern": "contra_only", "LI": -0.6})
            rows.append({"session": s, "pattern": "ipsi_only", "LI": 0.5})
            rows.append({"session": s, "pattern": "none", "LI": float("nan")})
        return pd.DataFrame(rows)

    def _tables(self):
        return {"start": self._epoch_table(2, 2),
                "max": self._epoch_table(2, 2),
                "end": self._epoch_table(5, 1)}      # offset: opponent-enriched

    def test_per_session_shape_and_columns(self):
        from neurokinematics.ephys.spikes.laterality import laterality_stats
        out = laterality_stats(self._tables())
        ps = out["per_session"]
        assert len(ps) == 3 * 4                       # 3 epochs x 4 sessions
        for col in ("opp_frac_bilateral", "contra_frac_unilateral", "median_LI"):
            assert col in ps.columns
        # offset opponent fraction (5/6) > onset (2/4)
        end = ps[ps.epoch == "end"]["opp_frac_bilateral"].mean()
        start = ps[ps.epoch == "start"]["opp_frac_bilateral"].mean()
        assert end > start

    def test_within_epoch_has_tests(self):
        from neurokinematics.ephys.spikes.laterality import laterality_stats
        out = laterality_stats(self._tables())
        we = out["within_epoch"]
        assert set(we["epoch"]) == {"start", "max", "end"}
        for col in ("opp_vs_congruent_wilcoxon_p", "contra_vs_ipsi_wilcoxon_p",
                    "LI_vs_zero_wilcoxon_p", "pooled_opp_vs_con_binom_p"):
            assert col in we.columns

    def test_across_epoch_friedman_per_metric(self):
        from neurokinematics.ephys.spikes.laterality import laterality_stats
        out = laterality_stats(self._tables())
        ae = out["across_epoch"]
        assert {"metric", "friedman_stat", "friedman_p", "n_sessions"} <= set(ae.columns)
        assert {"bilateral_frac", "opponent_frac", "contra_only_frac"} <= set(ae["metric"])
        assert (ae["n_sessions"] == 4).all()

    def test_across_epoch_posthoc_pairs(self):
        from neurokinematics.ephys.spikes.laterality import laterality_stats
        out = laterality_stats(self._tables())
        ph = out["across_epoch_posthoc"]
        assert {"metric", "epoch_a", "epoch_b", "wilcoxon_p", "wilcoxon_p_holm"} <= set(ph.columns)
        # 3 epoch pairs per metric
        assert (ph.groupby("metric").size() == 3).all()

    def test_accepts_single_table(self):
        from neurokinematics.ephys.spikes.laterality import laterality_stats
        out = laterality_stats(self._epoch_table(3, 1))   # single epoch
        assert len(out["within_epoch"]) == 1
        assert out["across_epoch"]["friedman_p"].isna().all()   # no Friedman with one epoch

    def test_summary_plot_single_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        from neurokinematics.ephys.spikes.plotting import plot_laterality_summary
        fig = plot_laterality_summary(self._tables())
        assert len(fig.axes) == 1                          # single summary plot
        assert fig.axes[0].get_ylabel() == "fraction of units"

    def test_session_trend_detects_increase(self):
        """bilateral_frac rising monotonically across sessions -> positive Spearman."""
        import numpy as np
        from neurokinematics.ephys.spikes.laterality import laterality_stats
        sessions = [f"s{i}" for i in range(6)]
        rows = []
        for i, s in enumerate(sessions):
            n_bi = i + 1                                   # increasing bilateral count
            for _ in range(n_bi):
                rows.append({"session": s, "pattern": "bilateral_opponent", "LI": 0.8})
            for _ in range(6):                             # constant filler
                rows.append({"session": s, "pattern": "none", "LI": float("nan")})
        df = pd.DataFrame(rows)
        out = laterality_stats({"start": df}, session_order=sessions)
        st = out["session_trend"]
        row = st[(st.epoch == "start") & (st.metric == "bilateral_frac")].iloc[0]
        assert row["spearman_rho"] > 0.9
        assert row["p"] < 0.05
        assert row["n_sessions"] == 6


# ---------------------------------------------------------------------------
# curation reader + good-unit auto-resolution
# ---------------------------------------------------------------------------

class TestCurationReader:

    def test_good_unit_ids_reads_cluster_group(self, tmp_path):
        import pandas as pd
        from neurokinematics.ephys.spikes.curation import good_unit_ids, read_phy_cluster_groups
        phy = tmp_path / "kilosort4" / "phy_output"
        phy.mkdir(parents=True)
        pd.DataFrame({"cluster_id": [0, 1, 2, 3],
                      "group": ["good", "mua", "good", "noise"]}
                     ).to_csv(phy / "cluster_group.tsv", sep="\t", index=False)
        assert good_unit_ids(phy) == [0, 2]
        groups = read_phy_cluster_groups(phy)
        assert list(groups.columns) == ["unit_id", "group"]

    def test_good_units_auto_resolves_from_session(self, tmp_path):
        """End-to-end: 'auto' reads cluster_group.tsv via a session-like object."""
        import pandas as pd
        from types import SimpleNamespace
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        from neurokinematics.io import save_dataset
        spikes = tmp_path / "spikes"
        phy = spikes / "kilosort4" / "phy_output"
        phy.mkdir(parents=True)
        pd.DataFrame({"cluster_id": [0, 1], "group": ["good", "mua"]}
                     ).to_csv(phy / "cluster_group.tsv", sep="\t", index=False)
        # save a real event_modulation.zarr (units 0,1) where the session expects it
        m = event_modulation(make_rasters_df(units=(0, 1), nodes=FOUR_LIMBS), n_shuffle=0)
        save_dataset(m, spikes / "modulation" / "event_modulation.zarr")
        sess = SimpleNamespace(session_id="dayA", dirs={"spikes": spikes})
        subject = SimpleNamespace(sessions=[sess])
        pooled = laterality_across_sessions(subject, good_units="auto", epoch="start")
        assert pooled["unit"].unique().tolist() == [0]      # only the 'good' cluster


class TestModFileSelection:

    @staticmethod
    def _subject_two_stores(tmp_path):
        """A session with two timestamped stores (distinguishable by unit ids)."""
        import os
        from types import SimpleNamespace
        from neurokinematics.io import save_dataset
        spikes = tmp_path / "spikes"
        mod = spikes / "modulation"
        older = event_modulation(make_rasters_df(units=(0, 1), nodes=FOUR_LIMBS), n_shuffle=0)
        newer = event_modulation(make_rasters_df(units=(2, 3), nodes=FOUR_LIMBS), n_shuffle=0)
        save_dataset(older, mod / "event_modulation_20260101_000000.zarr")
        save_dataset(newer, mod / "event_modulation_20260201_000000.zarr")
        os.utime(mod / "event_modulation_20260101_000000.zarr", (1_000, 1_000))
        os.utime(mod / "event_modulation_20260201_000000.zarr", (2_000, 2_000))
        sess = SimpleNamespace(session_id="dayA", dirs={"spikes": spikes})
        return SimpleNamespace(sessions=[sess])

    def test_newest_store_used_by_default(self, tmp_path):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        subject = self._subject_two_stores(tmp_path)
        pooled = laterality_across_sessions(subject, epoch="start")
        assert sorted(pooled["unit"].unique()) == [2, 3]            # the newer store

    def test_mod_file_pins_specific_store(self, tmp_path):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        subject = self._subject_two_stores(tmp_path)
        pooled = laterality_across_sessions(
            subject, epoch="start", mod_file="event_modulation_20260101_000000.zarr")
        assert sorted(pooled["unit"].unique()) == [0, 1]            # the older store

    def test_mod_file_mapping_per_session(self, tmp_path):
        from neurokinematics.ephys.spikes.laterality import laterality_across_sessions
        subject = self._subject_two_stores(tmp_path)
        pooled = laterality_across_sessions(
            subject, epoch="start",
            mod_file={"dayA": "event_modulation_20260101_000000.zarr"})
        assert sorted(pooled["unit"].unique()) == [0, 1]
