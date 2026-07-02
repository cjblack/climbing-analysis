"""Unit tests for neurokinematics.pose.interlimb."""

import numpy as np
import pandas as pd
import pytest

from neurokinematics.pose.interlimb import (
    interlimb_phase, circular_resultant, plot_interlimb_phase,
)


def _alignment(a_onsets, b_onsets, event="start", trial=0):
    """Minimal movement_event_alignment-style frame for nodes A and B."""
    rows = []
    for t in a_onsets:
        rows.append({"trial": trial, "node": "A", "movement_event": event, "event_times_ts": t})
    for t in b_onsets:
        rows.append({"trial": trial, "node": "B", "movement_event": event, "event_times_ts": t})
    return pd.DataFrame(rows)


class TestInterlimbPhase:

    def test_anti_phase_midcycle(self):
        # B onsets every 1 s; A onsets at the midpoint -> phase 0.5 (anti-phase)
        df = _alignment(a_onsets=[0.5, 1.5, 2.5], b_onsets=[0, 1, 2, 3])
        phases = interlimb_phase(df)
        ph = phases[("A", "B")]
        assert np.allclose(ph, 0.5, atol=1e-6)
        R, mean_phase = circular_resultant(ph)
        assert R == pytest.approx(1.0, abs=1e-6)        # perfectly locked
        assert mean_phase == pytest.approx(0.5, abs=1e-6)

    def test_in_phase_near_zero(self):
        df = _alignment(a_onsets=[0.05, 1.05, 2.05], b_onsets=[0, 1, 2, 3])
        ph = interlimb_phase(df)[("A", "B")]
        assert np.all(ph < 0.1)                         # near in-phase

    def test_onsets_outside_bracket_dropped(self):
        # A onset before the first / after the last B onset has no bracketing pair
        df = _alignment(a_onsets=[-0.5, 0.5, 5.0], b_onsets=[0, 1, 2])
        ph = interlimb_phase(df)[("A", "B")]
        assert ph.size == 1 and ph[0] == pytest.approx(0.5)

    def test_default_pairs_are_unordered_combinations(self):
        df = _alignment(a_onsets=[0.5], b_onsets=[0, 1])
        phases = interlimb_phase(df)
        assert list(phases.keys()) == [("A", "B")]

    def test_resultant_empty_is_nan(self):
        R, mp = circular_resultant(np.array([]))
        assert np.isnan(R) and np.isnan(mp)

    def test_plot_returns_polar_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        df = _alignment(a_onsets=[0.5, 1.4, 2.6], b_onsets=[0, 1, 2, 3])
        fig = plot_interlimb_phase(interlimb_phase(df))
        assert fig is not None
        assert fig.axes[0].name == "polar"
