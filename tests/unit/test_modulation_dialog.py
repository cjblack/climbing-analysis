"""Tests for the GUI event-modulation dialog and its rasters discovery helper.

Mirrors test_encoder_dialog.py: the pure path helper runs anywhere; the Qt
dialog construction is skipped unless pytest-qt is installed (it manages the
QApplication lifecycle so an offscreen app does not hang pytest on exit).
"""

import os
import importlib.util
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

_HAS_PYTEST_QT = importlib.util.find_spec("pytestqt") is not None

from neurokinematics.gui.dialogs import _find_movement_rasters


# ---------------------------------------------------------------------------
# _find_movement_rasters  (pure — no QApplication)
# ---------------------------------------------------------------------------

class TestFindMovementRasters:

    def test_finds_existing_pkl(self, tmp_path):
        rasters = tmp_path / "rasters"
        rasters.mkdir()
        pkl = rasters / "movement_aligned_rasters.pkl"
        pkl.write_bytes(b"")
        assert _find_movement_rasters(tmp_path) == pkl

    def test_none_when_missing(self, tmp_path):
        assert _find_movement_rasters(tmp_path) is None

    def test_none_when_dir_none(self):
        assert _find_movement_rasters(None) is None


# ---------------------------------------------------------------------------
# ModulationDialog construction (needs pytest-qt)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _HAS_PYTEST_QT,
    reason="pytest-qt manages the QApplication lifecycle; without it an "
           "offscreen QApplication hangs pytest on exit.",
)
class TestModulationDialog:

    def test_warns_and_no_result_without_rasters(self, qtbot, tmp_path):
        from neurokinematics.gui.dialogs import ModulationDialog
        session = SimpleNamespace(dirs={"spikes": tmp_path / "spikes"})
        dlg = ModulationDialog(session)
        qtbot.addWidget(dlg)
        assert dlg._rasters_path is None
        dlg._confirm()                     # no rasters -> no result
        assert dlg.result is None

    def test_builds_kwargs_when_rasters_present(self, qtbot, tmp_path):
        from neurokinematics.gui.dialogs import ModulationDialog
        rasters = tmp_path / "spikes" / "rasters"
        rasters.mkdir(parents=True)
        (rasters / "movement_aligned_rasters.pkl").write_bytes(b"")
        session = SimpleNamespace(dirs={"spikes": tmp_path / "spikes"})
        dlg = ModulationDialog(session)
        qtbot.addWidget(dlg)
        dlg._shuffle_spin.setValue(250)
        dlg._bin_spin.setValue(25.0)
        dlg._confirm()
        assert dlg.result is not None
        path, kwargs = dlg.result
        assert path.name == "movement_aligned_rasters.pkl"
        assert kwargs["n_shuffle"] == 250
        assert kwargs["bin_size"] == pytest.approx(0.025)
        assert kwargs["window"] == (-0.5, 0.5)
        assert kwargs["fdr"] is True
