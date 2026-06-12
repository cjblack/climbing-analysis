"""Tests for the GUI GLM-encoder dialog.

Two layers:

* ``TestDiscoverBinnedStores`` — the pure store-pairing helper. No Qt
  application is created, so these run under a plain ``pytest`` invocation.
* ``TestEncoderDialog`` — constructs the real Qt dialog. These need
  ``pytest-qt`` to manage the ``QApplication`` lifecycle; without it, an
  offscreen ``QApplication`` torn down at interpreter exit hangs the pytest
  process. They are therefore skipped unless ``pytest-qt`` is installed
  (``pip install pytest-qt``), in which case the ``qtbot`` fixture handles
  cleanup. The dialog's construction is otherwise covered by the manual
  smoke check in the PR notes.
"""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import importlib.util

_HAS_PYTEST_QT = importlib.util.find_spec("pytestqt") is not None

from neurokinematics.gui.dialogs import _discover_binned_stores


# ---------------------------------------------------------------------------
# _discover_binned_stores  (pure — no QApplication)
# ---------------------------------------------------------------------------

class TestDiscoverBinnedStores:

    def _make_store(self, folder, name):
        d = folder / name
        d.mkdir(parents=True, exist_ok=True)  # zarr stores are directories
        return d

    def test_pairs_matching_bin_sizes(self, tmp_path):
        pose = tmp_path / "pose"
        spikes = tmp_path / "spikes"
        self._make_store(pose, "resampled_movements_20ms.zarr")
        self._make_store(spikes, "movement_spike_counts_20ms.zarr")
        stores = _discover_binned_stores(pose, spikes)
        assert set(stores.keys()) == {20}
        pose_path, spike_path = stores[20]
        assert pose_path.name == "resampled_movements_20ms.zarr"
        assert spike_path.name == "movement_spike_counts_20ms.zarr"

    def test_only_includes_bins_present_in_both(self, tmp_path):
        pose = tmp_path / "pose"
        spikes = tmp_path / "spikes"
        self._make_store(pose, "resampled_movements_20ms.zarr")
        self._make_store(pose, "resampled_movements_50ms.zarr")
        self._make_store(spikes, "movement_spike_counts_50ms.zarr")
        stores = _discover_binned_stores(pose, spikes)
        assert set(stores.keys()) == {50}  # only 50 ms exists on both sides

    def test_empty_when_no_stores(self, tmp_path):
        stores = _discover_binned_stores(tmp_path / "pose", tmp_path / "spikes")
        assert stores == {}

    def test_missing_dirs_do_not_raise(self, tmp_path):
        stores = _discover_binned_stores(tmp_path / "nope", tmp_path / "also_nope")
        assert stores == {}


# ---------------------------------------------------------------------------
# EncoderDialog construction  (needs pytest-qt for safe QApplication teardown)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _HAS_PYTEST_QT,
    reason="pytest-qt manages the QApplication lifecycle; without it an "
           "offscreen QApplication hangs pytest on exit.",
)
class TestEncoderDialog:

    def test_disables_when_no_binned_data(self, qtbot, tmp_path):
        from neurokinematics.gui.dialogs import EncoderDialog
        session = SimpleNamespace(dirs={"pose": tmp_path / "pose",
                                        "spikes": tmp_path / "spikes",
                                        "models": tmp_path / "models"})
        dlg = EncoderDialog(session)
        qtbot.addWidget(dlg)
        assert not dlg._bin_combo.isEnabled()
        dlg._confirm()  # no stores -> no result
        assert dlg.result is None

    def test_constructs_with_stores_and_default_features(self, qtbot, tmp_path):
        from neurokinematics.gui.dialogs import EncoderDialog
        pose = tmp_path / "pose"
        spikes = tmp_path / "spikes"
        (pose / "resampled_movements_20ms.zarr").mkdir(parents=True)
        (spikes / "movement_spike_counts_20ms.zarr").mkdir(parents=True)
        session = SimpleNamespace(dirs={"pose": pose, "spikes": spikes,
                                        "models": tmp_path / "models"})
        dlg = EncoderDialog(session)
        qtbot.addWidget(dlg)
        assert dlg._bin_combo.isEnabled()
        checked = {f for f, cb in dlg._feat_checks.items() if cb.isChecked()}
        assert checked == {"velocity_x", "velocity_y"}
        assert dlg._basis_group.isChecked()
