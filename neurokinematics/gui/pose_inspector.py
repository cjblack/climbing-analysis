"""Pose-quality inspector: raw-vs-processed traces with a what-if preview.

Opened on demand (right-click a session's Pose row). Loads the raw SLEAP arrays,
lets you scrub through trials/keypoints, and live-previews how a confidence
threshold / gap cap changes the processed trace — without committing anything.
Optionally re-runs the real pose processing with the chosen settings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QSlider, QCheckBox, QSpinBox, QGroupBox, QDialogButtonBox,
    QMessageBox, QFileDialog,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from neurokinematics.gui.style import SUCCESS, WARNING, ERROR, TEXT_DIM
from neurokinematics.pose.inspect import (
    find_pose_files, load_sleap_arrays, preview_cleaning, node_below_fraction,
)

# threshold below which the model is essentially guessing — don't allow lower
_MIN_THRESH = 0.50


class PoseInspectDialog(QDialog):
    """Raw vs processed pose viewer with a confidence/gap what-if preview.

    *on_rerun(thresh, max_gap, remove_velocity, vel_thresh)* — optional callback
    invoked when the user chooses to re-run processing with the previewed
    settings. If omitted, the re-run button is hidden (view-only).
    """

    def __init__(self, session, log=None, on_rerun=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.log = log
        self._on_rerun = on_rerun
        self._arrays = None          # (locations, scores, node_names) cache for current file

        sid = getattr(session, 'session_id', 'session')
        self.setWindowTitle(f"Pose Quality — {sid}")
        self.setMinimumSize(820, 620)

        self._files = find_pose_files(getattr(session, 'pose_data_path', None))
        self._build_ui()
        if self._files:
            self._load_file(0)

    # ── config-derived defaults ──
    def _cfg_defaults(self):
        pp = (getattr(self.session, 'pose_cfg', None) or {}).get('pose_preprocessing', {})
        conf = pp.get('confidence', {}) or {}
        vel = pp.get('velocity', {}) or {}
        return {
            'thresh':   float(conf.get('thresh', 0.7)),
            'max_gap':  pp.get('max_gap', None),
            'vel_on':   bool(vel.get('enabled', False)),
            'vel_thr':  float(vel.get('thresh', 20.0)),
        }

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        if not self._files:
            warn = QLabel(
                "No raw pose files (.h5) found for this session.\n\n"
                f"Looked in: {getattr(self.session, 'pose_data_path', '(none)')}\n"
                "Link the pose data folder to enable the inspector.")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {WARNING};")
            layout.addWidget(warn)
            btns = QDialogButtonBox(QDialogButtonBox.Close)
            btns.rejected.connect(self.reject)
            layout.addWidget(btns)
            return

        d = self._cfg_defaults()

        # ── selectors ──
        sel = QHBoxLayout()
        self._file_combo = QComboBox()
        self._file_combo.addItems([Path(f).name for f in self._files])
        self._file_combo.currentIndexChanged.connect(self._load_file)
        self._node_combo = QComboBox()
        self._node_combo.currentIndexChanged.connect(self._redraw)
        sel.addWidget(QLabel("Trial:"))
        sel.addWidget(self._file_combo, 2)
        sel.addWidget(QLabel("Keypoint:"))
        sel.addWidget(self._node_combo, 1)
        layout.addLayout(sel)

        # ── plot ──
        self._fig = Figure(figsize=(7, 4.2), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas, stretch=1)

        # ── what-if controls ──
        controls = QGroupBox("What-if preview (does not change saved data)")
        form = QFormLayout(controls)

        self._thresh_slider = QSlider(Qt.Horizontal)
        self._thresh_slider.setRange(int(_MIN_THRESH * 100), 95)
        self._thresh_slider.setValue(int(round(d['thresh'] * 100)))
        self._thresh_lbl = QLabel()
        self._thresh_slider.valueChanged.connect(self._on_thresh_label)
        self._thresh_slider.sliderReleased.connect(self._redraw)
        trow = QHBoxLayout(); trow.addWidget(self._thresh_slider); trow.addWidget(self._thresh_lbl)
        form.addRow("Confidence ≥", trow)

        self._gap_check = QCheckBox("Cap interpolation gap")
        self._gap_spin = QSpinBox(); self._gap_spin.setRange(1, 2000); self._gap_spin.setSuffix(" frames")
        if d['max_gap']:
            self._gap_check.setChecked(True); self._gap_spin.setValue(int(d['max_gap']))
        else:
            self._gap_check.setChecked(False); self._gap_spin.setValue(20)
        self._gap_check.toggled.connect(self._redraw)
        self._gap_spin.valueChanged.connect(self._redraw)
        grow = QHBoxLayout(); grow.addWidget(self._gap_check); grow.addWidget(self._gap_spin); grow.addStretch()
        form.addRow("Long gaps", grow)

        self._vel_check = QCheckBox("Remove implausible jumps")
        self._vel_check.setChecked(d['vel_on'])
        self._vel_check.toggled.connect(self._redraw)
        form.addRow("Velocity", self._vel_check)
        self._vel_thr = d['vel_thr']

        layout.addWidget(controls)

        # ── stats ──
        self._stats = QLabel("")
        self._stats.setObjectName("subheading")
        layout.addWidget(self._stats)

        # ── actions ──
        btns = QHBoxLayout()
        save_btn = QPushButton("Save Figure…")
        save_btn.setObjectName("secondary")
        save_btn.clicked.connect(self._save_figure)
        btns.addWidget(save_btn)
        btns.addStretch()
        if self._on_rerun is not None:
            rerun = QPushButton("Re-run processing with these settings…")
            rerun.setObjectName("run")
            rerun.clicked.connect(self._do_rerun)
            btns.addWidget(rerun)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        btns.addWidget(close)
        layout.addLayout(btns)

        self._on_thresh_label()

    def _save_figure(self):
        """Save the current raw-vs-processed figure as a vector PDF (or SVG/PNG).

        PDF/SVG keep the traces as editable vector paths for Illustrator.
        """
        if not self._arrays:
            return
        sid   = getattr(self.session, 'session_id', 'session')
        node  = self._node_combo.currentText() or "keypoint"
        trial = Path(self._file_combo.currentText()).stem if self._file_combo.count() else "trial"
        safe  = f"pose_{sid}_{trial}_{node}".replace(" ", "_").replace("/", "-")
        from neurokinematics.gui.settings import get_default_root
        start = str(Path(get_default_root() or Path.home()) / f"{safe}.pdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pose Figure", start,
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)")
        if not path:
            return
        try:
            self._fig.savefig(path, bbox_inches="tight")
        except Exception as e:
            QMessageBox.warning(self, "Save Figure", f"Could not save figure:\n{e}")

    # ── data ──
    def _load_file(self, idx):
        if not (0 <= idx < len(self._files)):
            return
        try:
            self._arrays = load_sleap_arrays(self._files[idx])
        except Exception as e:
            self._arrays = None
            QMessageBox.warning(self, "Pose Inspector",
                                f"Could not read {Path(self._files[idx]).name}:\n{e}")
            return
        _, _, node_names = self._arrays
        self._node_combo.blockSignals(True)
        self._node_combo.clear()
        self._node_combo.addItems([str(n) for n in node_names])
        self._node_combo.blockSignals(False)
        self._redraw()

    def _current_params(self):
        thresh = self._thresh_slider.value() / 100.0
        max_gap = self._gap_spin.value() if self._gap_check.isChecked() else None
        return thresh, max_gap, self._vel_check.isChecked(), self._vel_thr

    def _on_thresh_label(self):
        self._thresh_lbl.setText(f"{self._thresh_slider.value() / 100.0:.2f}")

    def _redraw(self, *args):
        if not self._arrays:
            return
        locations, scores, node_names = self._arrays
        node = self._node_combo.currentIndex()
        if node < 0:
            return
        thresh, max_gap, rm_vel, vel_thr = self._current_params()
        proc, stats = preview_cleaning(
            locations, scores, thresh=thresh, max_gap=max_gap,
            remove_velocity=rm_vel, vel_thresh=vel_thr)

        frames = np.arange(locations.shape[0])
        raw_y  = locations[:, node, 1]
        proc_y = proc[:, node, 1]
        conf   = scores[:, node, 0]
        low    = conf < thresh

        self._fig.clear()
        ax1 = self._fig.add_subplot(2, 1, 1)
        ax1.plot(frames, raw_y, color=TEXT_DIM, lw=0.6, label="raw")
        ax1.plot(frames, proc_y, color=SUCCESS, lw=0.9, label="processed")
        # grey bands where the processed trace is left missing (long gaps)
        self._shade(ax1, frames, np.isnan(proc_y), color="#444", alpha=0.25)
        ax1.set_ylabel("y position")
        ax1.legend(loc="upper right", fontsize=7)
        ax1.set_title(f"{node_names[node]} — raw vs processed", fontsize=9)

        ax2 = self._fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(frames, conf, color="#7a9cc4", lw=0.6)
        ax2.axhline(thresh, color=ERROR, lw=0.8, ls="--")
        self._shade(ax2, frames, low, color=ERROR, alpha=0.15)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("confidence")
        ax2.set_xlabel("frame")
        self._canvas.draw_idle()

        node_frac = node_below_fraction(scores, node, thresh)
        self._stats.setText(
            f"This keypoint: {node_frac:.0%} below {thresh:.2f}.   "
            f"All keypoints: {stats['frac_below']:.0%} below, "
            f"{stats['frac_missing']:.0%} left missing after cleaning "
            f"({stats['n_frames']} frames × {stats['n_nodes']} nodes).")

    @staticmethod
    def _shade(ax, frames, mask, color, alpha):
        if not mask.any():
            return
        idx = np.flatnonzero(mask)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in runs:
            ax.axvspan(run[0], run[-1] + 1, color=color, alpha=alpha, lw=0)

    def _do_rerun(self):
        thresh, max_gap, rm_vel, vel_thr = self._current_params()
        gap_txt = f"{max_gap} frames" if max_gap else "no cap"
        ok = QMessageBox.question(
            self, "Re-run pose processing",
            f"Re-process this session's pose data with:\n\n"
            f"  • confidence ≥ {thresh:.2f}\n"
            f"  • gap cap: {gap_txt}\n"
            f"  • remove jumps: {'yes' if rm_vel else 'no'}\n\n"
            "This overwrites the processed pose output for this session.",
            QMessageBox.Ok | QMessageBox.Cancel)
        if ok != QMessageBox.Ok:
            return
        try:
            self._on_rerun(thresh, max_gap, rm_vel, vel_thr)
        finally:
            self.accept()
