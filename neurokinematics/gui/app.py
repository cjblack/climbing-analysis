"""
neurokinematics GUI — main application window.

Launch:
    python -m neurokinematics.gui.app
    # or from Python:
    from neurokinematics.gui.app import launch
    launch()
"""

import sys
import yaml
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QFileDialog,
    QMessageBox, QSizePolicy, QStackedWidget,
    QComboBox, QInputDialog, QDockWidget, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont, QIcon

from neurokinematics.gui.style import (
    STYLESHEET, PRIMARY, SECONDARY, TEXT, TEXT_DIM, BORDER, BG_LIGHT
)


# ── Assets ────────────────────────────────────────────────────────────────────

ASSETS_DIR = Path(__file__).parent / "assets"


def app_icon() -> QIcon:
    """Return the neurokinematics app icon, or an empty QIcon if none is found.

    Drop an icon file into ``neurokinematics/gui/assets/`` named one of the
    candidates below (``.ico`` is preferred on Windows for crisp taskbar
    rendering; ``.png`` works everywhere).
    """
    candidates = [
        "neurokinematics.ico", "neurokinematics.png",
        "icon.ico", "icon.png", "logo.png", "neurokinematics_icon.png"
    ]
    for name in candidates:
        path = ASSETS_DIR / name
        if path.exists():
            return QIcon(str(path))
    return QIcon()
from neurokinematics.gui.widgets import (
    PathField, HDivider, LogWidget, DetailPanel, Worker
)
from neurokinematics.gui.dialogs import SubjectDialog, GroupDialog, pick_spec_folder


# ── Tree item types ───────────────────────────────────────────────────────────

ITEM_PROJECT = 0
ITEM_GROUP   = 1
ITEM_SUBJECT = 2
ITEM_SESSION = 3


# ── Welcome panel ─────────────────────────────────────────────────────────────

class WelcomePanel(QWidget):
    """Landing page shown when nothing is loaded.

    Keeps the clean centred title/subtitle, and — when there is history —
    shows a 'Recent' list so a previous session can be reopened in one click.
    *on_open* is called with a recent-entry dict when a row is clicked.
    """

    # emoji + word per recent kind, mirroring the structure tree
    _KIND_ICON  = {"project": "📁", "group": "👥", "subject": "🐭"}
    _KIND_LABEL = {"project": "Project", "group": "Group", "subject": "Subject"}

    def __init__(self, on_open=None, parent=None):
        super().__init__(parent)
        self._on_open = on_open

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        title = QLabel("neurokinematics")
        title.setObjectName("heading")
        title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        title.setFont(font)

        subtitle = QLabel("Building reusable workflows for multimodal neuroscience")
        subtitle.setObjectName("subheading")
        subtitle.setAlignment(Qt.AlignCenter)

        hint = QLabel("File → New Project  or  File → Load Project  to get started")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")

        # ── Recent section (populated via set_recents) ──
        self._recent_box = QWidget()
        recent_layout = QVBoxLayout(self._recent_box)
        recent_layout.setAlignment(Qt.AlignHCenter)
        recent_layout.setSpacing(4)
        recent_layout.setContentsMargins(0, 0, 0, 0)

        self._recent_header = QLabel("Recent")
        self._recent_header.setAlignment(Qt.AlignCenter)
        self._recent_header.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; letter-spacing: 1px;")
        recent_layout.addWidget(self._recent_header, alignment=Qt.AlignHCenter)

        # divider under the header to separate it from the list
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedSize(380, 1)
        divider.setStyleSheet(f"background-color: {BORDER}; border: none;")
        recent_layout.addWidget(divider, alignment=Qt.AlignHCenter)
        recent_layout.addSpacing(4)

        self._recent_list = QVBoxLayout()
        self._recent_list.setSpacing(2)
        recent_layout.addLayout(self._recent_list)
        self._recent_box.setVisible(False)

        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        layout.addWidget(hint)
        layout.addSpacing(18)
        layout.addWidget(self._recent_box)
        layout.addStretch()

    def set_recents(self, recents):
        """Rebuild the recent list. Hides the section when there is nothing."""
        # clear existing rows
        while self._recent_list.count():
            item = self._recent_list.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        recents = recents or []
        self._recent_box.setVisible(bool(recents))

        for entry in recents:
            kind  = entry.get("kind")
            icon  = self._KIND_ICON.get(kind, "•")
            word  = self._KIND_LABEL.get(kind, (kind or "").title())
            label = entry.get("label") or Path(entry.get("path", "")).name
            proj  = entry.get("project_name")

            row = QPushButton()
            row.setCursor(Qt.PointingHandCursor)
            row.setFlat(True)
            row.setFixedWidth(380)
            row.setToolTip(entry.get("path", ""))
            row.setStyleSheet(
                "QPushButton { border: none; background: transparent;"
                " border-radius: 4px; }"
                f"QPushButton:hover {{ background: {BG_LIGHT}; }}"
            )
            row.clicked.connect(lambda _=False, e=entry: self._emit_open(e))

            h = QHBoxLayout(row)
            h.setContentsMargins(10, 4, 10, 4)
            h.setSpacing(8)

            # kind column: icon + word, dim, fixed width so names line up
            kind_lbl = QLabel(f"{icon}  {word}")
            kind_lbl.setFixedWidth(92)
            kind_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

            name_lbl = QLabel(label)
            name_lbl.setStyleSheet(f"color: {TEXT}; font-size: 13px;")

            h.addWidget(kind_lbl)
            h.addWidget(name_lbl)
            h.addStretch()

            if proj:
                proj_lbl = QLabel(proj)
                proj_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
                h.addWidget(proj_lbl)
            else:
                proj_lbl = None

            # let clicks fall through the labels to the button
            for lbl in (kind_lbl, name_lbl, proj_lbl):
                if lbl is not None:
                    lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)

            self._recent_list.addWidget(row, alignment=Qt.AlignHCenter)

    def _emit_open(self, entry):
        if callable(self._on_open):
            self._on_open(entry)


# ── Action bar (shown when a data item is selected) ───────────────────────────

class ActionBar(QWidget):
    """
    Shown below the data list when an item is selected.
    Displays relevant action buttons + optional mode selector.
    """
    action_requested = Signal(str, str)   # action_name, mode

    def __init__(self, actions: list, needs_mode: bool = True, parent=None):
        """
        Parameters
        ----------
        actions : list[str]
            Action names to show as buttons (e.g. ['Process', 'Align']).
        needs_mode : bool
            Whether to show the mode selector alongside the buttons.
        """
        super().__init__(parent)
        self._needs_mode = needs_mode

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(8)

        if needs_mode:
            mode_lbl = QLabel("Mode:")
            mode_lbl.setObjectName("subheading")
            self._mode_combo = QComboBox()
            self._mode_combo.addItems(["skip", "overwrite", "error"])
            layout.addWidget(mode_lbl)
            layout.addWidget(self._mode_combo)
            layout.addSpacing(8)
        else:
            self._mode_combo = None

        for action in actions:
            btn = QPushButton(action)
            btn.setObjectName("run" if action in ("Summarize", "Process", "Align") else "secondary")
            btn.clicked.connect(lambda checked=False, a=action: self._on_action(a))
            layout.addWidget(btn)

        layout.addStretch()

    def _on_action(self, action: str):
        mode = self._mode_combo.currentText() if self._mode_combo else ""
        self.action_requested.emit(action, mode)

    def set_enabled(self, enabled: bool):
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(enabled)
        if self._mode_combo:
            self._mode_combo.setEnabled(enabled)


# ── Data item widget (clickable row in a data list) ───────────────────────────

class DataItem(QWidget):
    """A single row showing a data label + status indicator."""

    clicked = Signal(object)   # emits self

    STATUS_COLOURS = {
        'available': '#4caf82',
        'missing':   '#555577',
        'running':   '#e0a050',
    }

    def __init__(self, label: str, status: str = 'available', meta: str = "",
                 always_clickable: bool = False, parent=None):
        super().__init__(parent)
        self._selected        = False
        self._always_clickable = always_clickable
        self.label  = label
        self.status = status

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {self.STATUS_COLOURS.get(status, '#555577')}; font-size: 10px;")
        dot.setFixedWidth(16)

        lbl = QLabel(label)
        lbl.setStyleSheet("font-weight: bold;" if status == 'available' else f"color: {TEXT_DIM};")

        layout.addWidget(dot)
        layout.addWidget(lbl)

        if meta:
            meta_lbl = QLabel(meta)
            meta_lbl.setObjectName("subheading")
            layout.addStretch()
            layout.addWidget(meta_lbl)

        is_clickable = (status == 'available') or always_clickable
        self.setCursor(Qt.PointingHandCursor if is_clickable else Qt.ArrowCursor)
        self._update_style()

    def mousePressEvent(self, event):
        if self.status == 'available' or self._always_clickable:
            self.clicked.emit(self)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_style()

    def _update_style(self):
        if self._selected:
            self.setStyleSheet(f"background-color: {PRIMARY}; border-radius: 4px;")
        else:
            self.setStyleSheet("background-color: transparent;")


def make_qc_section(run_cb, past_cb):
    """A collapsible 'Quality Control' section (Run QC + Past QC buttons).

    Shared by the group/subject/session panels so QC has one consistent home
    rather than a loose button row.
    """
    from neurokinematics.gui.widgets import CollapsibleSection
    section = CollapsibleSection("Quality Control", expanded=True)
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    info = QLabel("Run automated checks, or view a previously saved report.")
    info.setObjectName("subheading")
    h.addWidget(info)
    h.addStretch()
    past_btn = QPushButton("Past QC…")
    past_btn.setObjectName("secondary")
    past_btn.setFixedWidth(90)
    past_btn.clicked.connect(past_cb)
    run_btn = QPushButton("Run QC")
    run_btn.setObjectName("secondary")
    run_btn.setFixedWidth(90)
    run_btn.clicked.connect(run_cb)
    h.addWidget(past_btn)
    h.addWidget(run_btn)
    section.add_widget(row)
    return section


# ── Group detail panel ────────────────────────────────────────────────────────

class GroupPanel(DetailPanel):
    """
    Right-hand panel for a Group.

    Layout:
      - Heading
      - Subjects list (clickable — opens SubjectPanel inline or in tree)
      - Available data section (summaries that exist on disk)
        -> clicking one reveals action buttons: Summarize / Analyse
    """

    def __init__(self, group_obj, log: LogWidget, parent=None):
        super().__init__(parent)
        self.group        = group_obj
        self.log          = log
        self._selected_item = None
        self._action_bar    = None
        self._build()

    def _build(self):
        from neurokinematics.gui.widgets import CollapsibleSection
        self._add_heading(
            f"Group: {self.group.group_id}",
            f"{len(self.group.subjects)} subject(s)"
        )

        # ── Data overview (what's been computed across all sessions) ──
        self._layout.addWidget(self._build_overview_section())

        # ── Subjects section ──
        subj_section = CollapsibleSection("Subjects", expanded=True)
        for subj in self.group.subjects:
            n = len(getattr(subj, 'sessions', None) or [])
            item = DataItem(subj.subject_id, status='available', meta=f"{n} session(s)")
            item.clicked.connect(lambda it, s=subj: self.log.log(f"Subject: {s.subject_id}", 'info'))
            subj_section.add_widget(item)
        self._layout.addWidget(subj_section)

        # ── Processing section ──
        proc_section = CollapsibleSection("Processing", expanded=True)
        self._proc_items = {}
        for modality in ['pose', 'spikes', 'lfp']:
            item = DataItem(modality, status='available')
            item.clicked.connect(lambda it, m=modality: self._on_proc_selected(it, m))
            proc_section.add_widget(item)
            self._proc_items[modality] = item

        self._proc_action_container = QWidget()
        self._proc_action_layout = QVBoxLayout(self._proc_action_container)
        self._proc_action_layout.setContentsMargins(0, 0, 0, 0)
        proc_section.add_widget(self._proc_action_container)
        self._layout.addWidget(proc_section)

        # ── Alignment section (video / movement — not pose/ephys) ──
        align_section = CollapsibleSection("Alignment", expanded=True)
        self._align_items = {}
        for align_type in ['video', 'movement']:
            item = DataItem(align_type, status='available')
            item.clicked.connect(lambda it, a=align_type: self._on_align_selected(it, a))
            align_section.add_widget(item)
            self._align_items[align_type] = item

        self._align_action_container = QWidget()
        self._align_action_layout = QVBoxLayout(self._align_action_container)
        self._align_action_layout.setContentsMargins(0, 0, 0, 0)
        align_section.add_widget(self._align_action_container)
        self._layout.addWidget(align_section)

        # ── Epoch section (segment neural data around aligned events) ──
        epoch_section = CollapsibleSection("Epoch", expanded=True)
        self._epoch_items = {}
        for epoch_type in ['spikes', 'lfp']:
            item = DataItem(epoch_type, status='available')
            item.clicked.connect(lambda it, e=epoch_type: self._on_epoch_selected(it, e))
            epoch_section.add_widget(item)
            self._epoch_items[epoch_type] = item

        self._epoch_action_container = QWidget()
        self._epoch_action_layout = QVBoxLayout(self._epoch_action_container)
        self._epoch_action_layout.setContentsMargins(0, 0, 0, 0)
        epoch_section.add_widget(self._epoch_action_container)
        self._layout.addWidget(epoch_section)

        # ── Quality Control section ──
        self._layout.addWidget(make_qc_section(self._run_qc, self._view_past_qc))

        # ── Summaries section ──
        summ_section = CollapsibleSection("Summaries", expanded=True)
        self._summary_items = {}
        summaries_dir = self.group.dirs.get('summaries', None)

        for modality in ['pose', 'spikes', 'lfp']:
            parquet = Path(summaries_dir) / f"{modality}_metrics.parquet" if summaries_dir else None
            exists  = parquet and parquet.exists()
            status  = 'available' if exists else 'missing'
            meta    = "ready" if exists else "not yet generated"
            item    = DataItem(modality, status=status, meta=meta, always_clickable=True)
            item.clicked.connect(lambda it, m=modality: self._on_summary_selected(it, m))
            summ_section.add_widget(item)
            self._summary_items[modality] = item

        self._summ_action_container = QWidget()
        self._summ_action_layout = QVBoxLayout(self._summ_action_container)
        self._summ_action_layout.setContentsMargins(0, 0, 0, 0)
        summ_section.add_widget(self._summ_action_container)
        self._layout.addWidget(summ_section)

        self._layout.addStretch()

    def _run_qc(self):
        """Run QC across the whole group and show the report in a dock."""
        from neurokinematics.qc import run_qc
        main = self.window()
        if hasattr(main, '_open_qc_dock'):
            self.log.log(f"Running QC for group {self.group.group_id}…", 'info')
            main._open_qc_dock(run_qc(self.group), f"QC — {self.group.group_id}",
                               sources=self.group)

    def _view_past_qc(self):
        """Open a previously exported QC report for this group."""
        main = self.window()
        if hasattr(main, '_open_past_qc'):
            main._open_past_qc(self.group, self.group.group_id)

    def _build_overview_section(self, expanded: bool = True):
        """Roll-up of which modalities have been computed across all sessions."""
        from neurokinematics.gui.widgets import CollapsibleSection
        sessions = []
        for subj in self.group.subjects:
            sessions.extend(getattr(subj, 'sessions', None) or [])
        total = len(sessions)

        section = CollapsibleSection("Data Overview", expanded=expanded)
        if total == 0:
            empty = QLabel("No sessions loaded.")
            empty.setObjectName("subheading")
            section.add_widget(empty)
            return section

        for label, dir_key, indicator in [
            ("Pose",   "pose",   "pose_data.csv"),
            ("Spikes", "spikes", "sorting_analyzer"),
            ("LFP",    "lfp",    "lfp_preprocessed"),
        ]:
            n = 0
            for sess in sessions:
                folder = getattr(sess, 'dirs', {}).get(dir_key)
                if folder and (Path(folder) / indicator).exists():
                    n += 1
            section.add_widget(self._make_overview_row(label, n, total))
        return section

    def _make_overview_row(self, label: str, n: int, total: int) -> QWidget:
        row    = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)

        dot = QLabel("●")
        dot.setFixedWidth(14)
        colour = DataItem.STATUS_COLOURS['available' if n else 'missing']
        dot.setStyleSheet(f"color: {colour}; font-size: 10px;")

        lbl = QLabel(label)
        lbl.setMinimumWidth(80)
        lbl.setStyleSheet("font-weight: bold;" if n else f"color: {TEXT_DIM};")

        count = QLabel(f"{n}/{total} session(s)")
        count.setObjectName("subheading")

        layout.addWidget(dot)
        layout.addWidget(lbl)
        layout.addStretch()
        layout.addWidget(count)
        return row

    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _deselect_all(self, items_dict, exclude=None):
        for item in items_dict.values():
            if item is not exclude:
                item.set_selected(False)

    def _on_proc_selected(self, item: DataItem, modality: str):
        self._deselect_all(self._proc_items, exclude=item)
        self._deselect_all(self._align_items)
        self._deselect_all(self._epoch_items)
        self._deselect_all(self._summary_items)
        item.set_selected(True)
        self._clear_layout(self._summ_action_layout)
        self._clear_layout(self._align_action_layout)
        self._clear_layout(self._epoch_action_layout)
        self._clear_layout(self._proc_action_layout)

        bar = ActionBar(actions=["Process"], needs_mode=True)
        bar.action_requested.connect(lambda action, mode: self._run_proc(action, modality, mode))
        self._proc_action_layout.addWidget(bar)

    def _on_align_selected(self, item: DataItem, align_type: str):
        self._deselect_all(self._align_items, exclude=item)
        self._deselect_all(self._proc_items)
        self._deselect_all(self._epoch_items)
        self._deselect_all(self._summary_items)
        item.set_selected(True)
        self._clear_layout(self._proc_action_layout)
        self._clear_layout(self._epoch_action_layout)
        self._clear_layout(self._summ_action_layout)
        self._clear_layout(self._align_action_layout)

        bar = ActionBar(actions=["Align"], needs_mode=True)
        bar.action_requested.connect(lambda action, mode: self._run_align(align_type, mode))
        self._align_action_layout.addWidget(bar)

    def _on_epoch_selected(self, item: DataItem, epoch_type: str):
        self._deselect_all(self._epoch_items, exclude=item)
        self._deselect_all(self._proc_items)
        self._deselect_all(self._align_items)
        self._deselect_all(self._summary_items)
        item.set_selected(True)
        self._clear_layout(self._proc_action_layout)
        self._clear_layout(self._align_action_layout)
        self._clear_layout(self._summ_action_layout)
        self._clear_layout(self._epoch_action_layout)

        bar = ActionBar(actions=["Epoch"], needs_mode=True)
        bar.action_requested.connect(lambda action, mode: self._run_epoch(epoch_type, mode))
        self._epoch_action_layout.addWidget(bar)

    def _on_summary_selected(self, item: DataItem, modality: str):
        self._deselect_all(self._summary_items, exclude=item)
        self._deselect_all(self._proc_items)
        self._deselect_all(self._align_items)
        self._deselect_all(self._epoch_items)
        item.set_selected(True)
        self._clear_layout(self._proc_action_layout)
        self._clear_layout(self._align_action_layout)
        self._clear_layout(self._epoch_action_layout)
        self._clear_layout(self._summ_action_layout)

        # show Inspect/Analyse only if the parquet already exists
        summaries_dir = self.group.dirs.get('summaries', None)
        parquet       = Path(summaries_dir) / f"{modality}_metrics.parquet" if summaries_dir else None
        exists        = parquet and parquet.exists()
        actions       = ["Summarize", "Inspect", "Analyse"] if exists else ["Summarize"]

        bar = ActionBar(actions=actions, needs_mode=False)
        bar.action_requested.connect(
            lambda action, _, m=modality, p=parquet: self._run_summary_action(action, m, p)
        )
        self._summ_action_layout.addWidget(bar)

    def _refresh_summaries(self):
        """Update summary item status dots after a summarize run completes."""
        summaries_dir = self.group.dirs.get('summaries', None)
        if not summaries_dir:
            return
        for modality, item in self._summary_items.items():
            parquet = Path(summaries_dir) / f"{modality}_metrics.parquet"
            exists  = parquet.exists()
            new_status = 'available' if exists else 'missing'
            if item.status != new_status:
                item.status = new_status
                # re-colour the dot
                dot_colour = DataItem.STATUS_COLOURS.get(new_status, '#555577')
                dots = item.findChildren(QLabel)
                if dots:
                    dots[0].setStyleSheet(f"color: {dot_colour}; font-size: 10px;")
                meta_labels = [l for l in dots if l.text() in ('ready', 'not yet generated')]
                if meta_labels:
                    meta_labels[0].setText('ready' if exists else 'not yet generated')
                # make it clickable now that it exists
                if exists:
                    item._always_clickable = True
                    item.setCursor(Qt.PointingHandCursor)

    def _run_proc(self, action: str, modality: str, mode: str):
        if action != "Process":
            return
        self.log.log(f"group.process('{modality}', mode='{mode}')...", 'info')
        self._run_in_thread(lambda: self.group.process(modality, mode), self.log)

    def _run_align(self, align_type: str, mode: str):
        """Group-level alignment — video/movement, not pose/ephys modalities."""
        self.log.log(f"group.align('{align_type}', mode='{mode}')...", 'info')
        self._run_in_thread(lambda: self.group.align(align_type, mode), self.log)

    def _run_epoch(self, epoch_type: str, mode: str):
        """Group-level epoching — segment spikes/lfp around aligned events."""
        self.log.log(f"group.epoch('{epoch_type}', mode='{mode}')...", 'info')
        self._run_in_thread(lambda: self.group.epoch(epoch_type, mode), self.log)

    def _run_summary_action(self, action: str, modality: str, parquet_path=None):
        if action == "Inspect":
            if parquet_path and Path(parquet_path).exists():
                # find the MainWindow ancestor and open a dock there
                main = self.window()
                if hasattr(main, '_open_inspect_dock'):
                    main._open_inspect_dock(parquet_path, f"{modality}_metrics.parquet")
            return

        if action == "Analyse":
            self._open_analysis_dialog(modality)
            return

        fn_map = {
            "Summarize": lambda: self.group.summarize(modality),
        }
        fn = fn_map.get(action)
        if not fn:
            return
        self.log.log(f"group.{action.lower()}('{modality}')...", 'info')
        self._run_in_thread(fn, self.log)

        if action == "Summarize":
            # refresh summary items once done so status dots update
            self._worker.finished.connect(self._refresh_summaries)

    def _open_analysis_dialog(self, modality: str):
        from neurokinematics.gui.dialogs import AnalysisDialog
        dlg = AnalysisDialog(self.group, parent=self)
        # pre-select the right summary file if it matches
        idx = dlg._data_combo.findText(f"{modality}_metrics.parquet")
        if idx >= 0:
            dlg._data_combo.setCurrentIndex(idx)

        if dlg.exec() != AnalysisDialog.Accepted or dlg.result is None:
            return

        framework, model, data_file, params = dlg.result
        self.log.log(f"group.analyse('{framework}', '{model}', '{data_file}')...", 'info')

        def _run():
            result = self.group.analyse(
                framework = framework,
                model     = model,
                data      = data_file,
                params    = params,
            )
            # store trace on group for direct plotting
            if result is not None:
                _, trace = result if isinstance(result, tuple) else (None, result)
                self.group._last_trace        = trace
                self.group._last_trace_params = params
                self.group._last_trace_model  = model
                self.group._last_trace_framework = framework

        self._run_in_thread(_run, self.log)
        self._worker.finished.connect(
            lambda: self.log.log("Trace stored — use Plot Viewer to visualise.", 'success')
        )


# ── Subject detail panel ──────────────────────────────────────────────────────

class SubjectPanel(DetailPanel):
    """
    Right-hand panel for a Subject.

    Shows sessions with their processing status.
    Clicking a session reveals Process / Align action buttons.
    """

    def __init__(self, subject_obj, log: LogWidget, parent=None):
        super().__init__(parent)
        self.subject      = subject_obj
        self.log          = log
        self._sess_items  = {}
        self._build()

    def _build(self):
        from neurokinematics.gui.widgets import CollapsibleSection
        sessions = getattr(self.subject, 'sessions', None) or []
        self._add_heading(
            f"Subject: {self.subject.subject_id}",
            f"{len(sessions)} session(s)"
        )

        if not sessions:
            self._layout.addWidget(QLabel("No sessions loaded."))
            self._layout.addStretch()
            return

        sess_section = CollapsibleSection("Sessions", expanded=True)

        for sess in sessions:
            sess_id = getattr(sess, 'session_id', str(sess))
            dirs    = getattr(sess, 'dirs', {})
            tags    = []
            for modality, folder_key, indicator in [
                ('pose',   'pose',   'pose_data.csv'),
                ('spikes', 'spikes', 'sorting_analyzer'),
                ('lfp',    'lfp',    'lfp_preprocessed'),
            ]:
                folder = dirs.get(folder_key)
                if folder and (Path(folder) / indicator).exists():
                    tags.append(modality)

            meta = ", ".join(tags) if tags else "unprocessed"
            item = DataItem(sess_id, status='available', meta=meta)
            item.clicked.connect(lambda it, s=sess: self._on_session_selected(it, s))
            sess_section.add_widget(item)
            self._sess_items[sess_id] = item

        # action bar inside the section
        self._action_container = QWidget()
        self._action_layout    = QVBoxLayout(self._action_container)
        self._action_layout.setContentsMargins(0, 0, 0, 0)
        sess_section.add_widget(self._action_container)

        self._layout.addWidget(sess_section)

        # ── Quality Control section ──
        self._layout.addWidget(make_qc_section(self._run_qc, self._view_past_qc))
        self._layout.addStretch()

    def _run_qc(self):
        """Run QC across the whole subject and show the report in a dock."""
        from neurokinematics.qc import run_qc
        main = self.window()
        if hasattr(main, '_open_qc_dock'):
            self.log.log(f"Running QC for subject {self.subject.subject_id}…", 'info')
            main._open_qc_dock(run_qc(self.subject),
                               f"QC — {self.subject.subject_id}",
                               sources=self.subject)

    def _view_past_qc(self):
        """Open a previously exported QC report for this subject."""
        main = self.window()
        if hasattr(main, '_open_past_qc'):
            main._open_past_qc(self.subject, self.subject.subject_id)

    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _on_session_selected(self, item: DataItem, session):
        for it in self._sess_items.values():
            it.set_selected(False)
        item.set_selected(True)

        self._clear_layout(self._action_layout)

        bar = ActionBar(actions=["Process", "Align"], needs_mode=True)
        bar.action_requested.connect(lambda action, mode, s=session: self._run_session_action(action, mode, s))
        self._action_layout.addWidget(bar)

    def _run_session_action(self, action: str, mode: str, session):
        modality_map = {
            "Process": lambda m: session.process(m, mode),
            "Align":   lambda m: session.align(m, mode),
        }
        # For session-level, ask what modality
        from PySide6.QtWidgets import QInputDialog
        modalities = ["pose", "spikes", "lfp"] if action == "Process" else ["video", "movement"]
        modality, ok = QInputDialog.getItem(
            self, f"{action} — select modality", "Modality:", modalities, 0, False
        )
        if not ok:
            return
        fn = lambda m=modality: modality_map[action](m)
        self.log.log(f"session.{action.lower()}('{modality}', mode='{mode}')...", 'info')
        self._run_in_thread(fn, self.log)


# ── Session detail panel ──────────────────────────────────────────────────────

class SessionPanel(DetailPanel):
    """
    Right-hand panel for a single ExperimentSession.
    Shows all available operations as individual buttons — no dropdowns.
    """

    def __init__(self, session_obj, log: LogWidget, parent=None):
        super().__init__(parent)
        self.session = session_obj
        self.log     = log
        self._build()

    # suffixes we can display as a DataFrame
    INSPECTABLE_TYPES = {'.csv', '.parquet', '.pkl', '.pickle'}

    def _build(self):
        from neurokinematics.gui.widgets import CollapsibleSection
        sess_id  = getattr(self.session, 'session_id', str(self.session))
        dirs     = getattr(self.session, 'dirs', {})

        self._add_heading(f"Session: {sess_id}")

        # ── Available data section (record of what's been computed) ──
        self._layout.addWidget(self._build_outputs_section())

        # ── Processing section ──
        proc_section = CollapsibleSection("Processing", expanded=True)
        for label, dir_key, process_type, candidates in [
            ("Pose",   "pose",   "pose",   ["pose_data.csv"]),
            ("Spikes", "spikes", "spikes", ["sorting_analyzer"]),
            ("LFP",    "lfp",   "lfp",    ["lfp_preprocessed"]),
        ]:
            folder      = dirs.get(dir_key, None)
            data_path   = self._find_file(folder, candidates)
            exists      = data_path is not None
            inspectable = exists and data_path.suffix in self.INSPECTABLE_TYPES
            actions = ["Process"]
            fn_map  = {
                "Process": self._run(
                    f"session.process('{process_type}')",
                    lambda mode, p=process_type: self.session.process(p, mode)
                ),
            }
            # pose gets a 'Quality' action: raw-vs-processed what-if inspector
            if process_type == 'pose':
                actions.append("Quality")
                fn_map["Quality"] = self._inspect_pose
            if inspectable:
                actions.append("Inspect")
                fn_map["Inspect"] = lambda p=data_path, l=label: self._inspect(p, l)
            row = self._make_op_row(label=label, done=exists,
                                    actions=actions, fn_map=fn_map)
            proc_section.add_widget(row)
            # right-click a sorted Spikes row to open phy2
            if process_type == 'spikes':
                self._attach_phy_context(row, folder)
        self._layout.addWidget(proc_section)

        # ── Alignment section ──
        align_section = CollapsibleSection("Alignment", expanded=True)
        for label, dir_key, align_type, candidates in [
            ("Video",    "alignment", "video",    ["video_alignment.csv"]),
            ("Movement", "alignment", "movement", ["movement_alignment.csv"]),
        ]:
            folder      = dirs.get(dir_key, None)
            data_path   = self._find_file(folder, candidates)
            exists      = data_path is not None
            inspectable = exists and data_path.suffix in self.INSPECTABLE_TYPES
            row = self._make_op_row(
                label   = label,
                done    = exists,
                actions = ["Align"] + (["Inspect"] if inspectable else []),
                fn_map  = {
                    "Align": self._run(
                        f"session.align('{align_type}')",
                        lambda mode, a=align_type: self.session.align(a, mode)
                    ),
                    "Inspect": lambda p=data_path, l=label: self._inspect(p, l),
                }
            )
            align_section.add_widget(row)
        self._layout.addWidget(align_section)

        # ── Epoch section (segment neural data around aligned events) ──
        epoch_section = CollapsibleSection("Epoch", expanded=True)
        for label, dir_key, epoch_type, candidates in [
            ("Spikes", "spikes", "spikes",
             ["rasters/movement_aligned_rasters.pkl", "movement_aligned_rasters.pkl"]),
            ("LFP",    "lfp",    "lfp", ["lfp_epoched"]),
        ]:
            folder      = dirs.get(dir_key, None)
            data_path   = self._find_file(folder, candidates)
            exists      = data_path is not None
            inspectable = exists and data_path.suffix in self.INSPECTABLE_TYPES
            actions = ["Epoch"] + (["Inspect"] if inspectable else [])
            fn_map  = {
                "Epoch": self._run(
                    f"session.epoch('{epoch_type}')",
                    lambda mode, e=epoch_type: self.session.epoch(e, mode)
                ),
            }
            if inspectable:
                fn_map["Inspect"] = lambda p=data_path, l=label: self._inspect(p, l)
            epoch_section.add_widget(
                self._make_op_row(label=label, done=exists,
                                  actions=actions, fn_map=fn_map))
        self._layout.addWidget(epoch_section)

        # ── Modeling section (bin features, fit GLM encoder) ──
        self._layout.addWidget(self._build_modeling_section())

        # ── Quality Control section ──
        self._layout.addWidget(make_qc_section(self._run_qc, self._view_past_qc))

        # ── Other outputs section ──
        extras = []
        pose_folder = dirs.get('pose', None)
        if pose_folder and Path(pose_folder).exists():
            extras += [
                f for f in Path(pose_folder).iterdir()
                if f.suffix in self.INSPECTABLE_TYPES
                and f.name != 'pose_data.csv'
            ]
        # surface inspectable spike outputs (e.g. spike_qc_metrics.csv)
        spikes_folder = dirs.get('spikes', None)
        if spikes_folder and Path(spikes_folder).exists():
            extras += [
                f for f in Path(spikes_folder).iterdir()
                if f.is_file() and f.suffix in self.INSPECTABLE_TYPES
            ]
        if extras:
            other_section = CollapsibleSection("Other Outputs", expanded=False)
            for f in extras:
                row = self._make_op_row(
                    label   = f.name,
                    done    = True,
                    actions = ["Inspect"],
                    fn_map  = {"Inspect": lambda p=f: self._inspect(p, p.name)}
                )
                other_section.add_widget(row)
            self._layout.addWidget(other_section)

        # ── Configs ──
        # ── Configs section ──
        subject_id = getattr(self.session, 'metadata', {}).get('subject_id', '') if hasattr(self.session, 'metadata') else ''
        prefix = f"{subject_id} / {sess_id}" if subject_id else sess_id

        configs_to_show = [
            ("Session Config",    getattr(self.session, 'cfg',                   None)),
            ("Pose Config",       getattr(self.session, 'pose_cfg',              None)),
            ("Spike Config",      getattr(self.session, 'sorting_cfg',           None)),
            ("LFP Config",        getattr(self.session, 'lfp_preprocessing_cfg', None)),
            ("Multimodal Config", getattr(self.session, 'multimodal_cfg',        None)),
        ]
        available_cfgs = [(n, d) for n, d in configs_to_show if d]

        if available_cfgs:
            cfg_section = CollapsibleSection("Configs", expanded=False)
            cfg_row_widget = QWidget()
            cfg_row = QHBoxLayout(cfg_row_widget)
            cfg_row.setContentsMargins(0, 0, 0, 0)
            for cfg_name, cfg_data in available_cfgs:
                btn = QPushButton(cfg_name.replace(" Config", ""))
                btn.setObjectName("secondary")
                btn.setFixedHeight(28)
                btn.clicked.connect(
                    lambda checked=False, d=cfg_data, n=cfg_name, p=prefix:
                        self._show_config(d, f"{p} — {n}")
                )
                cfg_row.addWidget(btn)
            cfg_row.addStretch()
            cfg_section.add_widget(cfg_row_widget)
            self._layout.addWidget(cfg_section)

        self._layout.addStretch()

    # ── Modeling (binned features + GLM encoder) ───────────────────────────────
    def _build_modeling_section(self):
        """Section to bin pose/spikes and fit a GLM encoder on the result."""
        from neurokinematics.gui.widgets import CollapsibleSection
        dirs = getattr(self.session, 'dirs', {})
        section = CollapsibleSection("Modeling", expanded=True)

        spikes_dir = dirs.get('spikes')
        binned = bool(spikes_dir and Path(spikes_dir).exists()
                      and list(Path(spikes_dir).glob('movement_spike_counts_*ms.zarr')))
        section.add_widget(self._make_op_row(
            label="Binned", done=binned,
            actions=["Bin"], fn_map={"Bin": self._run_bin}))

        models_dir = dirs.get('models')
        glm_dir = Path(models_dir) / 'glm' if models_dir else None
        enc_done = bool(glm_dir and (glm_dir / 'encoder').exists())
        dec_done = bool(glm_dir and (glm_dir / 'decoder').exists())
        section.add_widget(self._make_op_row(
            label="Encoder", done=enc_done,
            actions=["Fit"], fn_map={"Fit": self._open_encoder_dialog}))
        section.add_widget(self._make_op_row(
            label="Decoder", done=dec_done,
            actions=["Fit"], fn_map={"Fit": self._open_decoder_dialog}))
        return section

    def _current_pre_window(self) -> float:
        """Pre-movement window (s) currently set in the session's pose config."""
        cfg = getattr(self.session, 'pose_cfg', None) or {}
        md  = cfg.get('movement_detection') or {}
        try:
            return float(md.get('pre_window_s', 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _run_bin(self):
        """Bin pose + spikes into matching zarr stores (encoder inputs).

        Also exposes the pre-movement window: changing it re-extracts the movement
        features (so pre-onset bins are captured) before binning.
        """
        bin_ms, ok = QInputDialog.getDouble(
            self, "Bin size", "Bin size (ms):", 20.0, 1.0, 1000.0, 1)
        if not ok:
            return
        current_pre = self._current_pre_window()
        pre_s, ok = QInputDialog.getDouble(
            self, "Pre-movement window",
            "Pre-movement lead-in (s)\n0 = onset→end;  >0 re-extracts movement events:",
            current_pre, 0.0, 5.0, 2)
        if not ok:
            return

        bin_size = bin_ms / 1000.0
        pre_s    = float(pre_s)
        reextract = abs(pre_s - current_pre) > 1e-9

        if reextract:
            self.log.log(
                f"Re-extracting movements (pre_window_s={pre_s}) then "
                f"binning (bin_size={bin_size})…", 'info')
        else:
            self.log.log(f"session.bin_movements_and_spikes(bin_size={bin_size})…", 'info')

        def _bin():
            if reextract:
                self.session.extract_movement_features(pre_window_s=pre_s)
            self.session.bin_movements_and_spikes(bin_size)

        self._run_in_thread(_bin, self.log)

    def _open_encoder_dialog(self):
        """Configure and run a GLM encoder on the binned data."""
        from neurokinematics.gui.dialogs import EncoderDialog
        from neurokinematics.models.glm import compare_glm_models

        dlg = EncoderDialog(self.session, parent=self)
        if dlg.exec() != EncoderDialog.Accepted or dlg.result is None:
            return
        pose_path, spike_path, params = dlg.result
        save_path = getattr(self.session, 'dirs', {}).get('models')
        self.log.log(
            f"Fitting GLM encoder: node={params['pose']['node']} "
            f"unit={params['spikes']['unit']} "
            f"features={params['pose']['features']} "
            f"basis={'on' if params['pose'].get('basis') else 'off'}…", 'info')
        self._run_in_thread(
            lambda: compare_glm_models(pose_path, spike_path, params, save_path),
            self.log)

    def _open_decoder_dialog(self):
        """Configure and run a GLM decoder (population of units -> movement feature)."""
        from neurokinematics.gui.dialogs import DecoderDialog
        from neurokinematics.models.glm import create_glm_decoder

        dlg = DecoderDialog(self.session, parent=self)
        if dlg.exec() != DecoderDialog.Accepted or dlg.result is None:
            return
        pose_path, spike_path, params = dlg.result
        save_path = getattr(self.session, 'dirs', {}).get('models')
        self.log.log(
            f"Fitting GLM decoder: {len(params['spikes']['unit'])} unit(s) "
            f"→ {params['pose']['node']} {params['pose']['features'][0]} "
            f"(cv={'on' if params.get('cv') else 'off'})…", 'info')
        self._run_in_thread(
            lambda: create_glm_decoder(pose_path, spike_path, params, save_path),
            self.log)

    def _run_qc(self):
        """Run QC for this session and show the report in a dock."""
        from neurokinematics.qc import run_qc
        main = self.window()
        sess_id = getattr(self.session, 'session_id', 'session')
        if hasattr(main, '_open_qc_dock'):
            self.log.log(f"Running QC for session {sess_id}…", 'info')
            main._open_qc_dock(run_qc(self.session), f"QC — {sess_id}",
                               sources=self.session)

    def _view_past_qc(self):
        """Open a previously exported QC report for this session."""
        main = self.window()
        sess_id = getattr(self.session, 'session_id', 'session')
        if hasattr(main, '_open_past_qc'):
            main._open_past_qc(self.session, sess_id)

    # ── phy2 integration ──────────────────────────────────────────────────────
    @staticmethod
    def _find_phy_params(spikes_dir):
        """Return the params.py inside a sorter's phy_output folder, or None."""
        if not spikes_dir:
            return None
        d = Path(spikes_dir)
        if not d.exists():
            return None
        hits = list(d.glob('*/phy_output/params.py'))
        return hits[0] if hits else None

    def _attach_phy_context(self, row, spikes_dir):
        """Enable a right-click 'Open in phy2' menu on a sorted Spikes row."""
        params = self._find_phy_params(spikes_dir)
        if not params:
            return
        row.setContextMenuPolicy(Qt.CustomContextMenu)
        row.customContextMenuRequested.connect(
            lambda pos, r=row, p=params: self._phy_menu(r, pos, p)
        )

    def _phy_menu(self, row, pos, params):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        act  = menu.addAction("Open in phy2")
        if menu.exec(row.mapToGlobal(pos)) == act:
            self._open_phy(params)

    def _open_phy(self, params):
        from neurokinematics.gui.settings import load_settings, launch_phy
        s = load_settings()
        env = (s.get('phy_env') or '').strip()
        if not env:
            self.log.log("No phy2 environment set. Configure it in File → Settings.", 'warning')
            return
        phy_dir = Path(params).parent
        try:
            launch_phy(phy_dir, env=env, gui=s.get('phy_gui', 'template-gui'),
                       conda_exe=s.get('conda_exe', 'conda'))
            self.log.log(f"Launching phy2 ({s.get('phy_gui','template-gui')}) "
                         f"in env '{env}' for {phy_dir} …", 'info')
        except Exception as e:
            self.log.log(f"Could not launch phy2: {e}", 'error')

    def _build_outputs_section(self, expanded: bool = True):
        """Section listing the session's recorded outputs (session_outputs.yaml).

        This is the authoritative record of what has already been computed for
        the session — name, file type, creation date, and whether the file is
        currently present on disk.
        """
        from neurokinematics.gui.widgets import CollapsibleSection
        outputs = getattr(self.session, 'session_outputs', {}) or {}
        n_present = 0
        rows      = []
        sess_dir  = getattr(self.session, 'session_path', None)

        for name, info in outputs.items():
            if isinstance(info, dict):
                path, ftype = self._resolve_output_path(sess_dir, info)
                ftype   = ftype or info.get('file_type', '') or ''
                created = (info.get('created', '') or '')[:10]   # YYYY-MM-DD
            else:
                path, ftype, created = None, '', ''
            exists = path is not None
            n_present += int(exists)
            inspectable = exists and Path(path).suffix in self.INSPECTABLE_TYPES
            rows.append((name, ftype, created, exists, path, inspectable))

        title   = f"Available Data ({n_present}/{len(outputs)})" if outputs else "Available Data"
        section = CollapsibleSection(title, expanded=expanded)

        if not rows:
            empty = QLabel("No outputs recorded yet — run a processing step to populate this.")
            empty.setObjectName("subheading")
            empty.setWordWrap(True)
            section.add_widget(empty)
        else:
            for name, ftype, created, exists, path, inspectable in rows:
                section.add_widget(
                    self._make_output_row(name, ftype, created, exists, path, inspectable)
                )
        return section

    def _make_output_row(self, name, ftype, created, exists, path, inspectable) -> QWidget:
        """One row in the Available Data section."""
        row    = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)

        dot = QLabel("●")
        dot.setFixedWidth(14)
        colour = DataItem.STATUS_COLOURS['available' if exists else 'missing']
        dot.setStyleSheet(f"color: {colour}; font-size: 10px;")
        if not exists:
            row.setToolTip("Recorded in session_outputs.yaml but file not found on disk.")

        lbl = QLabel(str(name))
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;" if exists else f"color: {TEXT_DIM};")

        layout.addWidget(dot)
        layout.addWidget(lbl)
        if ftype:
            type_lbl = QLabel(str(ftype))
            type_lbl.setObjectName("tag")
            layout.addWidget(type_lbl)
        layout.addStretch()
        if created:
            date_lbl = QLabel(created)
            date_lbl.setObjectName("subheading")
            layout.addWidget(date_lbl)

        if inspectable:
            btn = QPushButton("Inspect")
            btn.setObjectName("secondary")
            btn.setFixedWidth(75)
            btn.clicked.connect(lambda checked=False, p=path, n=name: self._inspect(p, str(n)))
            layout.addWidget(btn)
        return row

    def _find_file(self, folder, candidates: list):
        """Find first existing candidate file in folder. Returns Path or None."""
        if not folder:
            return None
        folder = Path(folder)
        if not folder.exists():
            # log to help diagnose
            self.log.log(f"Dir not found: {folder}", 'warning')
            return None
        for name in candidates:
            p = folder / name
            if p.exists():
                return p
        # log what IS there so we can see what files actually exist
        found = [f.name for f in folder.iterdir()] if folder.exists() else []
        self.log.log(f"In {folder.name}/: {found}", 'info')
        return None

    def _show_config(self, cfg: dict, title: str = "Config"):
        """Show a config dict in a dock viewer."""
        import yaml as _yaml
        text = _yaml.dump(cfg, default_flow_style=False, sort_keys=False)
        main = self.window()
        if hasattr(main, '_open_text_dock'):
            main._open_text_dock(text, title)

    def _resolve_output_path(self, sess_dir, out_info: dict):
        """
        Resolve a session output entry to an absolute Path that exists, or None.
        Tries the stored path as-is first (may be absolute), then relative to session_path.
        Returns (path, file_type) tuple.
        """
        if not isinstance(out_info, dict):
            return None, None
        stored   = out_info.get('path', None)
        ftype    = out_info.get('file_type', None)
        if not stored:
            return None, None
        # try absolute first
        p = Path(stored)
        if p.exists():
            return p, ftype
        # try relative to session_path
        if sess_dir:
            p2 = Path(sess_dir) / stored
            if p2.exists():
                return p2, ftype
        return None, None

    def _inspect(self, path, label: str):
        """Open a dock inspector for a CSV or parquet file."""
        if not path or not Path(path).exists():
            self.log.log(f"No inspectable file found for {label}", 'warning')
            return
        main = self.window()
        if hasattr(main, '_open_inspect_dock'):
            main._open_inspect_dock(path, str(Path(path).name))
        else:
            # fallback: modal dialog
            from neurokinematics.gui.dialogs import DataFrameDialog
            dlg = DataFrameDialog(source=path, title=label, parent=self)
            dlg.exec()

    def _inspect_pose(self):
        """Open the pose-quality inspector (raw vs processed + what-if)."""
        from neurokinematics.pose.inspect import find_pose_files
        from neurokinematics.gui.pose_inspector import PoseInspectDialog
        if not find_pose_files(getattr(self.session, 'pose_data_path', None)):
            QMessageBox.warning(
                self, "Pose Quality",
                "No raw pose (.h5) data found for this session.\n\n"
                "Link the pose data folder to use the inspector.")
            return
        dlg = PoseInspectDialog(self.session, log=self.log,
                                on_rerun=self._rerun_pose, parent=self)
        dlg.exec()

    def _rerun_pose(self, thresh, max_gap, remove_velocity, vel_thresh):
        """Apply the previewed settings to the pose cfg and re-process (overwrite)."""
        cfg = getattr(self.session, 'pose_cfg', None)
        if not cfg:
            self.log.log("No pose config loaded; cannot re-run pose processing.", 'error')
            return
        pp = cfg.setdefault('pose_preprocessing', {})
        conf = pp.setdefault('confidence', {})
        conf['enabled'] = True
        conf['thresh'] = float(thresh)
        pp['max_gap'] = int(max_gap) if max_gap else None
        vel = pp.setdefault('velocity', {})
        vel['enabled'] = bool(remove_velocity)
        vel['thresh'] = float(vel_thresh)
        # best-effort persist so the new settings survive a reload
        try:
            if hasattr(self.session, '_save_session_config'):
                self.session._save_session_config()
        except Exception as e:
            self.log.log(f"(pose settings not persisted: {e})", 'warning')
        self.log.log(
            f"Re-running pose processing (confidence ≥ {thresh:.2f}, "
            f"max_gap={pp['max_gap']})…", 'info')
        self._run("session.process('pose')",
                  lambda mode: self.session.process('pose', mode))('overwrite')

    # actions that don't need a mode selector
    _NO_MODE_ACTIONS = {"Inspect", "View", "Quality", "Bin", "Fit"}

    def _make_op_row(self, label: str, done: bool, actions: list, fn_map: dict) -> QWidget:
        """
        Build a single labelled row with status dot and action buttons.
        Buttons in _NO_MODE_ACTIONS are called with no arguments.
        All other buttons receive the current mode string.
        Mode selector is hidden if all actions are mode-independent.
        """
        row    = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)

        dot = QLabel("●")
        dot.setFixedWidth(14)
        dot.setStyleSheet(f"color: {'#4caf82' if done else '#555577'}; font-size: 10px;")

        lbl = QLabel(label)
        lbl.setFixedWidth(80)

        status = QLabel("done" if done else "—")
        status.setObjectName("subheading")

        layout.addWidget(dot)
        layout.addWidget(lbl)
        layout.addWidget(status)
        layout.addStretch()

        # mode selector — only show if at least one action needs it
        needs_mode = any(a not in self._NO_MODE_ACTIONS for a in actions)
        mode_combo = QComboBox()
        mode_combo.addItems(["skip", "overwrite", "error"])
        mode_combo.setFixedWidth(85)
        mode_combo.setVisible(needs_mode)
        layout.addWidget(mode_combo)

        for action in actions:
            btn = QPushButton(action)
            btn.setObjectName("run" if action not in self._NO_MODE_ACTIONS else "secondary")
            btn.setFixedWidth(75)
            fn  = fn_map.get(action)
            if fn:
                if action in self._NO_MODE_ACTIONS:
                    # call with no arguments
                    btn.clicked.connect(lambda checked=False, f=fn: f())
                else:
                    # call with mode string
                    btn.clicked.connect(lambda checked=False, f=fn, mc=mode_combo: f(mc.currentText()))
            layout.addWidget(btn)

        return row

    def _run(self, label: str, fn_with_mode):
        """Helper used by _make_op_row lambdas — captures mode and runs in thread."""
        def _execute(mode: str):
            self.log.log(f"{label}, mode='{mode}'...", 'info')
            self._run_in_thread(lambda: fn_with_mode(mode), self.log)
        return _execute


# ── Relationships panel ───────────────────────────────────────────────────────

class RelationshipsPanel(QWidget):
    """
    Shows the hierarchy of loaded groups, subjects, and sessions
    as a collapsible QTreeWidget.
    """

    def __init__(self, loaded_objects: dict, log: LogWidget, parent=None):
        super().__init__(parent)
        self.loaded_objects = loaded_objects
        self.log            = log
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        heading = QLabel("Project Relationships")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        layout.addWidget(HDivider())

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(16)
        self._tree.setRootIsDecorated(False)   # suppress Qt's own branch indicators
        self._tree.itemExpanded.connect(self._on_item_expanded)
        self._tree.itemCollapsed.connect(self._on_item_collapsed)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

        if not self.loaded_objects:
            item = QTreeWidgetItem(self._tree)
            item.setText(0, "No groups or subjects loaded yet.")
            return

        for spec_path, obj in self.loaded_objects.items():
            try:
                from neurokinematics.data.group import ExperimentGroup
                from neurokinematics.data.subject import ExperimentSubject
                if isinstance(obj, ExperimentGroup):
                    self._add_group(obj)
                elif isinstance(obj, ExperimentSubject):
                    self._add_subject(None, obj)
            except Exception:
                pass

        self._tree.expandAll()

    def _add_group(self, group):
        root = QTreeWidgetItem(self._tree)
        root.setText(0, f"▼  👥  {group.group_id}")
        root.setForeground(0, self._colour(PRIMARY))
        root.setFont(0, self._bold_font())

        # summaries
        summaries_dir = group.dirs.get('summaries', None)
        if summaries_dir:
            for modality in ['pose', 'spikes', 'lfp']:
                p = Path(summaries_dir) / f"{modality}_metrics.parquet"
                if p.exists():
                    s = QTreeWidgetItem(root)
                    s.setText(0, f"    📊  {modality}_metrics.parquet")
                    s.setForeground(0, self._colour(SUCCESS))

        for subj in group.subjects:
            self._add_subject(root, subj)

    def _add_subject(self, parent, subj):
        item = QTreeWidgetItem(parent if parent else self._tree)
        item.setText(0, f"▼  🐭  {subj.subject_id}")
        item.setForeground(0, self._colour('#c084fc'))
        item.setFont(0, self._bold_font())

        sessions = getattr(subj, 'sessions', None) or []
        if not sessions:
            empty = QTreeWidgetItem(item)
            empty.setText(0, "    no sessions")
            empty.setForeground(0, self._colour(TEXT_DIM))
            return

        for sess in sessions:
            sess_id = getattr(sess, 'session_id', str(sess))
            outputs = getattr(sess, 'session_outputs', {})
            tags    = []
            if 'pose_data'     in outputs: tags.append("pose")
            if 'spike_sorting' in outputs: tags.append("spikes")
            if 'lfp_data'      in outputs: tags.append("lfp")
            tag_str = f"  [{', '.join(tags)}]" if tags else "  [unprocessed]"
            colour  = SUCCESS if tags else TEXT_DIM

            s = QTreeWidgetItem(item)
            s.setText(0, f"    📅  {sess_id}{tag_str}")
            s.setForeground(0, self._colour(colour))

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int):
        """Toggle expand/collapse when clicking a parent item."""
        if item.childCount() > 0:
            if item.isExpanded():
                self._tree.collapseItem(item)
            else:
                self._tree.expandItem(item)

    def _on_item_expanded(self, item: QTreeWidgetItem):
        """Swap ▶ → ▼ when a node is expanded."""
        text = item.text(0)
        if text.startswith("▶"):
            item.setText(0, "▼" + text[1:])

    def _on_item_collapsed(self, item: QTreeWidgetItem):
        """Swap ▼ → ▶ when a node is collapsed."""
        text = item.text(0)
        if text.startswith("▼"):
            item.setText(0, "▶" + text[1:])

    @staticmethod
    def _colour(hex_colour: str):
        from PySide6.QtGui import QColor, QBrush
        return QBrush(QColor(hex_colour))

    @staticmethod
    def _bold_font():
        from PySide6.QtGui import QFont
        f = QFont()
        f.setBold(True)
        return f


# ── Project tree ────────────────────────────────────────────────────────────

class StructureTree(QTreeWidget):
    """Project tree that draws a purple expand/collapse caret for parent rows.

    The app stylesheet hides Qt's native branch arrows, so parent items would
    otherwise have no visible expander. We paint our own ▶/▼ caret (matching the
    purple used in the Relationships view) while leaving the item labels — and
    their own colours — untouched.
    """

    CARET_COLOUR = "#c084fc"

    def drawBranches(self, painter, rect, index):
        # only parents get a caret; leaves (sessions) get nothing
        if self.model().hasChildren(index):
            from PySide6.QtGui import QColor, QPolygon, QPainter
            from PySide6.QtCore import QPoint

            expanded = self.isExpanded(index)
            cx = rect.right() - 7
            cy = rect.center().y()
            s  = 4

            if expanded:   # ▼ pointing down
                pts = QPolygon([QPoint(cx - s, cy - 2),
                                QPoint(cx + s, cy - 2),
                                QPoint(cx, cy + s - 1)])
            else:          # ▶ pointing right
                pts = QPolygon([QPoint(cx - 2, cy - s),
                                QPoint(cx - 2, cy + s),
                                QPoint(cx + s - 1, cy)])

            painter.save()
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self.CARET_COLOUR))
            painter.drawPolygon(pts)
            painter.restore()
        # deliberately skip super(): we don't want the default arrows/lines


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("neurokinematics")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        self._loaded_objects = {}   # spec_path -> instantiated Group/Subject
        self._build_ui()
        self._build_menu()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Root splitter: tree (left) | content (right)
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left: project tree
        tree_container = QWidget()
        tree_layout    = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(0)

        self.tree = StructureTree()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.setRootIsDecorated(True)   # allocate branch area for the caret
        # transparent branch bg so our purple caret sits cleanly on the panel
        self.tree.setStyleSheet("QTreeWidget::branch { background: transparent; }")
        self.tree.itemClicked.connect(self._on_tree_item_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        tree_layout.addWidget(self.tree)

        # Right: stacked panels
        right_container = QWidget()
        right_layout    = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.welcome_panel = WelcomePanel(on_open=self._open_recent)
        self.stack.addWidget(self.welcome_panel)
        self._refresh_recents()

        right_layout.addWidget(self.stack, stretch=1)

        # Log at the bottom of the right panel
        self.log = LogWidget()
        self.log.setMaximumHeight(180)
        right_layout.addWidget(HDivider())
        right_layout.addWidget(self.log)

        splitter.addWidget(tree_container)
        splitter.addWidget(right_container)
        splitter.setSizes([220, 780])

        # install stdout/stderr redirectors so tqdm appears in the log
        self.log.install_redirectors()

    def _build_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        new_project  = QAction("New Project",  self)
        load_project = QAction("Load Project", self)

        
        

        load_sample = QAction("Load Sample Project", self)
        load_sample.triggered.connect(self._load_sample_project)



        new_group   = QAction("New Group",   self)
        load_group  = QAction("Load Group",  self)

        
        


        new_subject  = QAction("New Subject",  self)
        load_subject = QAction("Load Subject", self)

        # new
        file_menu.addAction(new_subject)
        file_menu.addAction(new_group)
        file_menu.addAction(new_project)
        file_menu.addSeparator()

        # loads
        file_menu.addAction(load_subject)
        file_menu.addAction(load_group)
        file_menu.addAction(load_project)
        file_menu.addAction(load_sample)
        

        file_menu.addSeparator()

        clear_workspace = QAction("Clear Workspace", self)
        file_menu.addAction(clear_workspace)
        file_menu.addSeparator()

        settings_action = QAction("Settings…", self)
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)

        # Connect
        new_group.triggered.connect(self._new_group)
        load_group.triggered.connect(self._load_group)
        new_subject.triggered.connect(self._new_subject)
        load_subject.triggered.connect(self._load_subject)
        load_project.triggered.connect(self._load_project)
        new_project.triggered.connect(self._new_project)
        clear_workspace.triggered.connect(self._clear_workspace)

        # View menu
        view_menu = menubar.addMenu("View")
        show_relationships = QAction("Show Relationships", self)
        show_relationships.triggered.connect(self._show_relationships)
        view_menu.addAction(show_relationships)
        view_menu.addSeparator()

        clear_log = QAction("Clear Log", self)
        clear_log.triggered.connect(self.log.clear_log)
        view_menu.addAction(clear_log)

        view_menu.addSeparator()

        # Screenshot export — crisp, consistently framed PNGs for slides.
        shot_menu = view_menu.addMenu("Save Screenshot")
        shot_window = QAction("Whole Window…", self)
        shot_window.triggered.connect(lambda: self._save_screenshot("window"))
        shot_menu.addAction(shot_window)
        shot_panel = QAction("Content Panel…", self)
        shot_panel.triggered.connect(lambda: self._save_screenshot("panel"))
        shot_menu.addAction(shot_panel)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        plot_action = QAction("Plot Viewer", self)
        plot_action.triggered.connect(self._open_plot_viewer)
        tools_menu.addAction(plot_action)

        qc_action = QAction("Data QC Report", self)
        qc_action.triggered.connect(self._run_qc_all)
        tools_menu.addAction(qc_action)

        tools_menu.addSeparator()

        # Analysis submenu — statistical modelling of group summaries.
        # Kept as a submenu (like Config Creator) so it can grow to hold
        # model comparison, posterior checks, decoding, etc.
        analysis_menu = tools_menu.addMenu("Analysis")
        fit_model_action = QAction("Fit Model…", self)
        fit_model_action.triggered.connect(self._open_analysis_dialog)
        analysis_menu.addAction(fit_model_action)

        tools_menu.addSeparator()

        new_subject_spec = QAction("Subject Spec Builder", self)
        new_subject_spec.triggered.connect(self._open_subject_spec_builder)
        tools_menu.addAction(new_subject_spec)

        new_group_spec = QAction("Group Spec Builder", self)
        new_group_spec.triggered.connect(self._open_group_spec_builder)
        tools_menu.addAction(new_group_spec)

        tools_menu.addSeparator()

        cfg_menu = tools_menu.addMenu("Config Creator")
        for key, label in [
            ('session',    'Session Config'),
            ('pose',       'Pose Config'),
            ('spikes',     'Spike Sorting Config'),
            ('lfp',        'LFP Config'),
            ('multimodal', 'Multimodal Config'),
        ]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, k=key: self._open_config_creator(k))
            cfg_menu.addAction(act)

        # Help menu
        help_menu = menubar.addMenu("Help")

        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(lambda: self._open_url("https://github.com/cjblack/neurokinematics"))
        help_menu.addAction(docs_action)

        osf_action = QAction("Sample Data (OSF)", self)
        osf_action.triggered.connect(lambda: self._open_url("https://doi.org/10.17605/OSF.IO/3SR67"))
        help_menu.addAction(osf_action)

        paper_action = QAction("Publication", self)
        paper_action.triggered.connect(lambda: self._open_url("https://doi.org/10.1016/j.isci.2026.115901"))
        help_menu.addAction(paper_action)

        help_menu.addSeparator()

        about_action = QAction("About neurokinematics", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ── Tree helpers ──────────────────────────────────────────────────────────

    def _make_subject_item(self, parent, subj):
        """Build a subject tree node (with its session children) under *parent*."""
        from PySide6.QtGui import QBrush, QColor
        subj_item = QTreeWidgetItem(parent)
        subj_item.setText(0, f"🐭  {subj.subject_id}")
        subj_item.setData(0, Qt.UserRole, (ITEM_SUBJECT, subj))
        subj_item.setForeground(0, QBrush(QColor("#e8e6f0")))

        for sess in (getattr(subj, 'sessions', None) or []):
            sess_id   = sess if isinstance(sess, str) else getattr(sess, 'session_id', str(sess))
            sess_item = QTreeWidgetItem(subj_item)
            sess_item.setText(0, f"📅  {sess_id}")
            sess_item.setData(0, Qt.UserRole, (ITEM_SESSION, sess))
            sess_item.setForeground(0, QBrush(QColor("#8e8aaa")))

        subj_item.setExpanded(True)
        return subj_item

    def _add_group_to_tree(self, group_obj, spec_path: str):
        from PySide6.QtGui import QBrush, QColor, QFont
        root = QTreeWidgetItem(self.tree)
        root.setText(0, f"👥  {group_obj.group_id}")
        root.setData(0, Qt.UserRole, (ITEM_GROUP, spec_path))
        root.setForeground(0, QBrush(QColor("#e8e6f0")))   # white (matches TEXT)
        f = QFont(); f.setBold(True)
        root.setFont(0, f)

        for subj in group_obj.subjects:
            self._make_subject_item(root, subj)

        root.setExpanded(True)
        self._loaded_objects[spec_path] = group_obj

    def _add_subject_to_tree(self, subject_obj, spec_path: str):
        item = self._make_subject_item(self.tree, subject_obj)
        self._loaded_objects[spec_path] = subject_obj

    # ── Tree lookup helpers ───────────────────────────────────────────────────
    def _iter_top_items(self):
        for i in range(self.tree.topLevelItemCount()):
            yield self.tree.topLevelItem(i)

    def _top_item_id(self, item):
        """Return ('group'|'subject', id) for a top-level tree item, else None."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return None
        kind, payload = data
        if kind == ITEM_GROUP:
            obj = self._loaded_objects.get(payload)
            return ('group', getattr(obj, 'group_id', None)) if obj else None
        if kind == ITEM_SUBJECT:
            return ('subject', getattr(payload, 'subject_id', None))
        return None

    def _find_loaded_item(self, kind: str, obj_id):
        """Find the top-level tree item already representing this id, if any."""
        for item in self._iter_top_items():
            if self._top_item_id(item) == (kind, obj_id):
                return item
        return None

    def _panel_shows(self, obj) -> bool:
        """True if the currently displayed detail panel belongs to *obj*."""
        w = self.stack.currentWidget()
        return (getattr(w, 'group', None) is obj or
                getattr(w, 'subject', None) is obj)

    def _show_panel(self, widget: QWidget):
        # Replace any previously shown detail panel (index >= 1)
        while self.stack.count() > 1:
            old = self.stack.widget(1)
            self.stack.removeWidget(old)
            old.deleteLater()
        self.stack.addWidget(widget)
        self.stack.setCurrentWidget(widget)

    def _refresh_current_panel(self):
        """Rebuild the active detail panel so status dots/outputs reflect disk.

        Called when a background process/alignment finishes, so the user sees
        the green dots update without navigating away and back. Recreates a fresh
        panel for the same object (same path navigation already takes).
        """
        w = self.stack.currentWidget()
        if w is None or w is self.welcome_panel:
            return
        group = getattr(w, 'group', None)
        subject = getattr(w, 'subject', None)
        session = getattr(w, 'session', None)
        if group is not None:
            self._show_panel(GroupPanel(group, self.log))
        elif subject is not None:
            self._show_panel(SubjectPanel(subject, self.log))
        elif session is not None:
            self._show_panel(SessionPanel(session, self.log))

    def _save_screenshot(self, target: str = "window"):
        """Grab the window or just the content panel to a PNG.

        Uses QWidget.grab(), which renders at the device pixel ratio, so the
        capture is crisp and identically framed every time — handy for slides.
        """
        if target == "panel":
            widget = self.stack.currentWidget()
            default = "nk_panel.png"
        else:
            widget = self
            default = "nk_window.png"

        start = str(Path.home() / default)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", start, "PNG Image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        pixmap = widget.grab()
        if pixmap.save(path, "PNG"):
            self.log.log(f"Screenshot saved: {path}", 'success')
        else:
            self.log.log(f"Could not save screenshot to {path}", 'error')

    def _open_settings(self):
        from neurokinematics.gui.dialogs import SettingsDialog
        SettingsDialog(self).exec()

    def _clear_workspace(self):
        """Remove all loaded groups/subjects from the viewer (data on disk is untouched)."""
        self.tree.clear()
        self._loaded_objects.clear()
        # drop any open detail panel and return to the welcome screen
        while self.stack.count() > 1:
            old = self.stack.widget(1)
            self.stack.removeWidget(old)
            old.deleteLater()
        self.stack.setCurrentWidget(self.welcome_panel)
        self.log.log("Workspace cleared.", 'info')

    # ── Recents ───────────────────────────────────────────────────────────────
    def _refresh_recents(self):
        """Reload the recent list onto the welcome panel."""
        from neurokinematics.gui.settings import load_recents
        try:
            self.welcome_panel.set_recents(load_recents())
        except Exception:
            pass

    def _record_recent(self, kind: str, path: str, label: str,
                       project_name: str | None = None):
        from neurokinematics.gui.settings import add_recent
        try:
            add_recent(kind, path, label, project_name)
            self._refresh_recents()
        except Exception:
            pass

    def _open_recent(self, entry: dict):
        """Reopen a project/group/subject clicked on the welcome screen."""
        kind = entry.get("kind")
        path = entry.get("path")
        if not path or not Path(path).exists():
            self.log.log(f"Recent item no longer exists: {path}", 'warning')
            return
        if kind == "project":
            self._load_project_path(path)
        elif kind == "group":
            self._instantiate_and_add_group(path)
        elif kind == "subject":
            self._instantiate_and_add_subject(path)

    def closeEvent(self, event):
        self.log.restore_streams()
        super().closeEvent(event)

    def _show_relationships(self):
        panel = RelationshipsPanel(self._loaded_objects, self.log)
        self._show_panel(panel)

    def _dock_titlebar(self, dock, title: str) -> QWidget:
        """Custom dock title bar with an always-visible float + close button.

        The themed stylesheet blanks the native dock titlebar icons, so a docked
        report/inspector had no usable close button until it was floated. This
        replaces the titlebar with text-glyph buttons that render everywhere and
        work whether docked or floating.
        """
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 2, 4, 2)
        lay.setSpacing(2)

        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {TEXT}; font-weight: bold;")
        lay.addWidget(lbl)
        lay.addStretch()

        btn_css = (
            "QPushButton { border: none; background: transparent; "
            f"color: {TEXT}; font-size: 13px; border-radius: 3px; }} "
            f"QPushButton:hover {{ background: {PRIMARY}; color: white; }}"
        )
        float_btn = QPushButton("❐")
        float_btn.setFixedSize(22, 20)
        float_btn.setToolTip("Float / dock")
        float_btn.setCursor(Qt.PointingHandCursor)
        float_btn.setStyleSheet(btn_css)
        float_btn.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(22, 20)
        close_btn.setToolTip("Close")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(btn_css)
        close_btn.clicked.connect(dock.close)

        lay.addWidget(float_btn)
        lay.addWidget(close_btn)
        bar.setStyleSheet(
            f"background-color: {SECONDARY}; border-bottom: 1px solid {BORDER};")
        return bar

    def _open_text_dock(self, text: str, title: str = "Config"):
        """Open a read-only text viewer as a dock widget."""
        from PySide6.QtWidgets import QTextEdit
        viewer = QTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText(text)
        viewer.setStyleSheet(
            f"font-family: 'Consolas', monospace; font-size: 12px;"
        )
        dock = QDockWidget(title, self)
        dock.setWidget(viewer)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        dock.setTitleBarWidget(self._dock_titlebar(dock, title))
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setMinimumWidth(360)

    def _open_plot_viewer(self):
        """Open an interactive plot viewer dock."""
        from neurokinematics.gui.plot_viewer import PlotViewerPanel
        dock = QDockWidget("Plot Viewer", self)
        panel = PlotViewerPanel(self._loaded_objects, self.log)
        dock.setWidget(panel)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        dock.setTitleBarWidget(self._dock_titlebar(dock, "Plot Viewer"))
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setMinimumWidth(500)

    def _open_analysis_dialog(self):
        """Open the statistical-modelling dialog from the Tools ▸ Analysis menu.

        Unlike the per-modality 'Analyse' action on a group panel, this entry
        point carries no modality context, so the user picks the summary file
        in the dialog. Resolves the target group from the loaded objects.
        """
        from neurokinematics.data.group import ExperimentGroup
        from neurokinematics.gui.dialogs import AnalysisDialog

        groups = [obj for obj in self._loaded_objects.values()
                  if isinstance(obj, ExperimentGroup)]
        if not groups:
            QMessageBox.information(self, "Fit Model",
                                    "Load a group first — analysis runs over "
                                    "a group's summary metrics.")
            return

        if len(groups) == 1:
            group = groups[0]
        else:
            labels = [g.group_id for g in groups]
            choice, ok = QInputDialog.getItem(
                self, "Fit Model", "Group:", labels, 0, False)
            if not ok:
                return
            group = groups[labels.index(choice)]

        dlg = AnalysisDialog(group, parent=self)
        if dlg.exec() != AnalysisDialog.Accepted or dlg.result is None:
            return

        framework, model, data_file, params = dlg.result

        def _run():
            result = group.analyse(
                framework = framework,
                model     = model,
                data      = data_file,
                params    = params,
            )
            if result is not None:
                _, trace = result if isinstance(result, tuple) else (None, result)
                group._last_trace           = trace
                group._last_trace_params    = params
                group._last_trace_model     = model
                group._last_trace_framework = framework

        self._run_bg(
            f"group.analyse('{framework}', '{model}', '{data_file}')", _run)
        self._bg_worker.finished.connect(
            lambda: self.log.log("Trace stored — use Plot Viewer to visualise.",
                                 'success'))

    def _open_config_creator(self, config_type: str = 'session'):
        from neurokinematics.gui.config_creator import ConfigCreatorDialog
        dlg = ConfigCreatorDialog(config_type=config_type, parent=self)
        dlg.exec()

    def _open_subject_spec_builder(self):
        from neurokinematics.gui.dialogs import SubjectDialog
        dlg = SubjectDialog(self)
        if dlg.exec() == SubjectDialog.Accepted and dlg.spec_path:
            self._instantiate_and_add_subject(
                dlg.spec_path,
                project_path=dlg.project_path,
                project_name=dlg.project_name,
                force_create=dlg.created_new,
            )

    def _open_group_spec_builder(self):
        from neurokinematics.gui.dialogs import GroupDialog
        dlg = GroupDialog(self)
        if dlg.exec() == GroupDialog.Accepted and dlg.spec_path:
            self._instantiate_and_add_group(
                dlg.spec_path,
                project_path=dlg.project_path,
                project_name=dlg.project_name,
                force_create=dlg.created_new,
            )

    def _open_url(self, url: str):
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl(url))

    def _show_about(self):
        QMessageBox.about(
            self,
            "About neurokinematics",
            "<b>neurokinematics</b> v0.1.0<br><br>"
            "Building reusable workflows for multimodal neuroscience.<br><br>"
            "Christopher Black, PhD<br>"
            "Miriam Marks Postdoctoral Fellow<br>"
            "Neural Circuits for Movement Laboratory<br>"
            "University College London<br><br>"
            '<a href="https://github.com/cjblack/neurokinematics">GitHub</a>'
        )

    def _run_qc_all(self):
        """Run QC over every loaded group/subject and show a combined report."""
        if not self._loaded_objects:
            QMessageBox.information(self, "Data QC",
                                    "Load a group or subject first.")
            return
        from neurokinematics.qc import run_qc
        self.log.log("Running data QC over loaded objects…", 'info')
        reports, sources = [], []
        for obj in self._loaded_objects.values():
            try:
                reports.append(run_qc(obj))
                sources.append(obj)   # keep parallel so Export targets each folder
            except Exception as exc:
                self.log.log(f"QC failed for an object: {exc}", 'error')
        if reports:
            self._open_qc_dock(reports, "Data QC Report", sources=sources)
            self.log.log("Data QC complete.", 'success')

    def _open_past_qc(self, obj, label: str):
        """Load and display a previously exported QC report for *obj*.

        Reads the timestamped JSONs from the object's ``qc/`` subfolder (written
        by the QC Export button). Lets the user pick which run to view; shown
        read-only (Export falls back to a Save-As so history isn't overwritten).
        """
        from datetime import datetime
        import json
        from neurokinematics.qc.session_qc import QCReport

        folder = None
        for attr in ('session_path', 'subject_path', 'group_path'):
            p = getattr(obj, attr, None)
            if p:
                folder = Path(p)
                break
        if folder is None:
            QMessageBox.information(self, "Past QC",
                                    "Could not locate this item's folder.")
            return

        qc_dir = folder / "qc"
        files = sorted(qc_dir.glob("qc_report_*.json"), reverse=True) if qc_dir.is_dir() else []
        if not files:
            QMessageBox.information(
                self, "Past QC",
                f"No saved QC reports for {label}.\n\n"
                "Run QC, then use Export in the report to save one here.")
            return

        def _stamp(p):
            s = p.stem.replace("qc_report_", "")
            try:
                return datetime.strptime(s, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                return s

        labels = [_stamp(p) for p in files]
        if len(files) == 1:
            chosen = files[0]
        else:
            choice, ok = QInputDialog.getItem(
                self, "Past QC", f"Saved QC reports for {label}:",
                labels, 0, False)
            if not ok:
                return
            chosen = files[labels.index(choice)]

        try:
            report = QCReport.from_dict(json.loads(chosen.read_text()))
        except Exception as e:
            self.log.log(f"Could not load QC report {chosen.name}: {e}", 'error')
            return
        # no sources -> Export uses Save-As, so viewing history can't overwrite it
        self._open_qc_dock(report, f"QC — {label} ({_stamp(chosen)})")

    def _open_qc_dock(self, reports, title: str = "Data QC", sources=None):
        """Open a dockable QC report viewer for one or more QCReports.

        *sources* (the group/subject/session object(s) QC was run on, parallel to
        *reports*) lets Export save each report into that object's own folder.
        """
        from neurokinematics.gui.qc_panel import QCReportWidget
        widget = QCReportWidget(reports, title=title, sources=sources)

        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        dock.setTitleBarWidget(self._dock_titlebar(dock, title))
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setMinimumWidth(440)

    def _open_inspect_dock(self, parquet_path, title: str = "Inspect"):
        """Open a dockable DataFrame inspector for any CSV or parquet file."""
        """Open a dockable DataFrame inspector — floatable and moveable."""
        from neurokinematics.gui.dialogs import DataFrameDialog

        # build the content widget
        content = QWidget()
        layout  = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)

        # reuse DataFrameDialog internals inline
        dlg = DataFrameDialog(source=parquet_path, title=title)
        # extract the table widget from the dialog and embed it
        table   = dlg._table
        shape   = dlg._shape_lbl
        spinner = dlg._n_spin

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Rows:"))
        ctrl.addWidget(spinner)
        ctrl.addStretch()
        ctrl.addWidget(shape)
        layout.addLayout(ctrl)
        layout.addWidget(table)

        dock = QDockWidget(title, self)
        dock.setWidget(content)
        dock.setAllowedAreas(
            Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        dock.setTitleBarWidget(self._dock_titlebar(dock, title))
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setMinimumWidth(400)

    # ── Tree click handler ────────────────────────────────────────────────────

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _col: int):
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        item_type, obj = data

        if item_type == ITEM_GROUP:
            group_obj = self._loaded_objects.get(obj)
            if group_obj:
                panel = GroupPanel(group_obj, self.log)
                self._show_panel(panel)

        elif item_type == ITEM_SUBJECT:
            panel = SubjectPanel(obj, self.log)
            self._show_panel(panel)

        elif item_type == ITEM_SESSION:
            panel = SessionPanel(obj, self.log)
            self._show_panel(panel)

    # ── Tree right-click actions ───────────────────────────────────────────────

    def _on_tree_context_menu(self, pos):
        """Right-click a subject/group for add-session / process / align actions."""
        from PySide6.QtWidgets import QMenu
        item = self.tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        item_type, payload = data

        menu = QMenu(self)

        if item_type == ITEM_SUBJECT:
            subj = payload
            menu.addAction("Add Session…",
                           lambda checked=False, s=subj: self._add_session_to_subject(s))
            menu.addSeparator()
            self._add_proc_align_actions(menu, subj, subj.subject_id)
            # only a standalone (top-level) subject can be cleared from here
            if item.parent() is None:
                menu.addSeparator()
                menu.addAction("Clear Subject",
                               lambda checked=False, it=item: self._clear_loaded(it))

        elif item_type == ITEM_GROUP:
            group = self._loaded_objects.get(payload)
            if group is None:
                return
            menu.addAction("Add Subject…",
                           lambda checked=False, g=group, it=item:
                               self._add_subject_to_group(g, it))
            menu.addSeparator()
            self._add_proc_align_actions(menu, group, group.group_id)
            menu.addSeparator()
            menu.addAction("Clear Group",
                           lambda checked=False, it=item: self._clear_loaded(it))

        elif item_type == ITEM_SESSION:
            session = payload
            menu.setToolTipsVisible(True)
            # grey out actions whose required raw data isn't linked to the session
            has_pose  = bool(getattr(session, 'pose_data_path', None))
            has_ephys = bool(getattr(session, 'ephys_data_path', None))
            enabled = {
                'pose':     has_pose,    # pose preprocessing needs pose data
                'spikes':   has_ephys,   # spike sorting needs ephys
                'lfp':      has_ephys,   # lfp processing needs ephys
                'video':    has_ephys,   # video alignment reads ephys camera events
                'movement': has_pose,    # movement alignment needs pose data
            }
            # Process ▸ Spikes routes through the bad-channel policy/review flow
            # (so the 'ask' policy shows the channel dialog before sorting).
            self._add_proc_align_actions(
                menu, session, getattr(session, 'session_id', 'session'),
                enabled=enabled,
                spikes_handler=lambda s=session: self._run_spike_sorting_qc(s))
            menu.addSeparator()
            add_menu = menu.addMenu("Add data")
            add_menu.addAction(
                "Pose…",
                lambda checked=False, s=session: self._add_data_to_session(s, 'pose'))
            add_menu.addAction(
                "Ephys…",
                lambda checked=False, s=session: self._add_data_to_session(s, 'ephys'))

        else:
            return

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _add_proc_align_actions(self, menu, target, label, enabled=None,
                                spikes_handler=None):
        """Add Process (pose/spikes/lfp) and Align (video/movement) submenus.

        *enabled* (optional) maps a key ('pose','spikes','lfp','video','movement')
        to a bool; falsey entries are greyed out with an explanatory tooltip.
        Passed for sessions so unavailable modalities are disabled; omitted for
        groups/subjects, where every action stays enabled.

        *spikes_handler* (optional) overrides the Spikes action — used for a
        single session so it routes through the bad-channel policy/review flow
        rather than a plain ``process('spikes')``.
        """
        proc = menu.addMenu("Process")
        for text, modality in [("Pose", "pose"), ("Spikes", "spikes"), ("LFP", "lfp")]:
            if modality == "spikes" and spikes_handler is not None:
                act = proc.addAction(text,
                                     lambda checked=False, h=spikes_handler: h())
            else:
                act = proc.addAction(
                    text,
                    lambda checked=False, m=modality, t=target, n=label:
                        self._run_bg(f"{n}.process('{m}')", lambda: t.process(m, 'skip'))
                )
            self._apply_enabled(act, enabled, modality)
        align = menu.addMenu("Align")
        for text, kind in [("Video", "video"), ("Movement", "movement")]:
            act = align.addAction(
                text,
                lambda checked=False, a=kind, t=target, n=label:
                    self._run_bg(f"{n}.align('{a}')", lambda: t.align(a, 'skip'))
            )
            self._apply_enabled(act, enabled, kind)

    @staticmethod
    def _apply_enabled(action, enabled, key):
        """Disable *action* (with a tooltip) when *enabled[key]* is falsey."""
        if enabled is None:
            return
        if not enabled.get(key, True):
            action.setEnabled(False)
            action.setToolTip("Required data not linked to this session")

    def _add_data_to_session(self, session, kind):
        """Link a pose or ephys data folder to an existing session and persist it.

        Useful when data arrives after the session was created. Updates the
        session's data path and rewrites session_config.yaml so it survives a
        reload.
        """
        from neurokinematics.gui.settings import get_data_root
        title = "Select Pose Data Folder" if kind == 'pose' else "Select Ephys Data Folder"
        folder = QFileDialog.getExistingDirectory(self, title, get_data_root(kind))
        if not folder:
            return
        sid = getattr(session, 'session_id', 'session')
        try:
            if kind == 'pose':
                session.pose_data_path = Path(folder)
            else:
                session.ephys_data_path = Path(folder)
            # persist so it survives a reload (session_config.yaml is authoritative)
            if hasattr(session, '_save_session_config'):
                session._save_session_config()
            self.log.log(f"Linked {kind} data to session '{sid}': {folder}", 'success')
        except Exception as e:
            self.log.log(f"Could not link {kind} data to '{sid}': {e}", 'error')

    def _run_spike_sorting_qc(self, session):
        """Detect bad channels, then (per policy) review or auto-sort.

        Step 1 runs detection in the background. On completion, the configured
        policy decides: 'ask' opens the review dialog; 'remove'/'keep' proceed
        straight to sorting with/without the detected channels removed.
        """
        from neurokinematics.gui.settings import load_settings
        self._spike_qc_policy  = load_settings().get('spike_bad_channel_policy', 'ask')
        self._spike_qc_session = session
        sid = getattr(session, 'session_id', 'session')
        self._run_bg(f"{sid}: detecting bad channels",
                     session.preprocess_spikes)
        # Connect to a *bound method* (a QObject in the GUI thread) with an
        # explicit queued connection, so the review dialog is created on the main
        # thread. A bare lambda would run in the worker thread and crash with
        # "Cannot set parent, new parent is in a different thread".
        self._bg_worker.finished.connect(self._after_spike_detect,
                                         Qt.QueuedConnection)

    def _after_spike_detect(self):
        session = getattr(self, '_spike_qc_session', None)
        policy  = getattr(self, '_spike_qc_policy', 'ask')
        if session is None:
            return
        detection = getattr(session, '_spike_detection', None)
        sid = getattr(session, 'session_id', 'session')
        if detection is None:
            self.log.log(f"{sid}: bad-channel detection produced no result.",
                         'warning')
            return
        bad = [str(b) for b in detection.get('bad_ids', [])]

        if policy == 'ask':
            from neurokinematics.gui.dialogs import SpikeQCDialog
            dlg = SpikeQCDialog(detection, session_label=sid, parent=self)
            if dlg.exec() != SpikeQCDialog.Accepted:
                self.log.log(f"{sid}: spike sorting cancelled.", 'info')
                return
            chosen = dlg.selected_bad
        elif policy == 'remove':
            chosen = bad
        else:  # keep
            chosen = []

        self.log.log(
            f"{sid}: {len(bad)} bad channel(s) detected, removing "
            f"{len(chosen)} (policy: {policy}).", 'info')
        self._run_bg(
            f"{sid}.run_spike_sorting(remove={len(chosen)} ch)",
            lambda: session.run_spike_sorting(mode='skip',
                                              bad_channels=chosen or None))

    def _add_session_to_subject(self, subj):
        from neurokinematics.gui.dialogs import SessionDialog
        dlg = SessionDialog(self)
        if dlg.exec() == SessionDialog.Accepted and dlg.result_data:
            try:
                subj.add_sessions([dlg.result_data])
                self.log.log(
                    f"Added session '{dlg.result_data['session_id']}' to {subj.subject_id}.",
                    'success')
                self._rebuild_tree()
            except Exception as e:
                self.log.log(f"Could not add session: {e}", 'error')

    def _add_subject_to_group(self, group, group_item):
        """Create/select a subject spec and attach it to *group*, live in the tree."""
        dlg = SubjectDialog(self)
        if dlg.exec() != SubjectDialog.Accepted or not dlg.spec_path:
            return
        try:
            before = {getattr(s, 'subject_id', None) for s in group.subjects}
            group.add_subjects([{'spec': str(dlg.spec_path)}])
            new = [getattr(s, 'subject_id', '?') for s in group.subjects
                   if getattr(s, 'subject_id', None) not in before]
            self._rebuild_tree()
            self.log.log(
                f"Added {', '.join(new) or 'subject'} to group "
                f"'{group.group_id}'.", 'success')
            # refresh the group panel if it's showing so the subject count updates
            if self._panel_shows(group):
                self._show_panel(GroupPanel(group, self.log))
        except Exception as e:
            self.log.log(f"Could not add subject to group: {e}", 'error')

    def _clear_loaded(self, item):
        """Remove a top-level group/subject from the workspace (disk untouched)."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, payload = data

        if kind == ITEM_GROUP:
            key   = payload
            obj   = self._loaded_objects.get(key)
            label = f"group '{getattr(obj, 'group_id', '?')}'"
        elif kind == ITEM_SUBJECT and item.parent() is None:
            obj   = payload
            key   = next((k for k, v in self._loaded_objects.items() if v is obj), None)
            label = f"subject '{getattr(obj, 'subject_id', '?')}'"
        else:
            return

        # Was the open detail panel showing this item OR one of its children
        # (a subject/session under a cleared group)? If so it's now stale.
        panel_stale = self._panel_shows_any(self._descendants_of(obj))

        if key is not None:
            self._loaded_objects.pop(key, None)
        idx = self.tree.indexOfTopLevelItem(item)
        if idx >= 0:
            self.tree.takeTopLevelItem(idx)
        self.log.log(f"Cleared {label} from workspace.", 'info')

        if panel_stale:
            self._show_fallback_panel()

    def _descendants_of(self, obj):
        """Set of objects 'contained' by obj (itself + subjects + sessions)."""
        out = {obj} if obj is not None else set()
        for subj in (getattr(obj, 'subjects', None) or []):
            out.add(subj)
            for sess in (getattr(subj, 'sessions', None) or []):
                out.add(sess)
        for sess in (getattr(obj, 'sessions', None) or []):   # obj is a subject
            out.add(sess)
        return out

    def _panel_shows_any(self, objs) -> bool:
        """True if the active panel displays any object in *objs*."""
        w = self.stack.currentWidget()
        shown = {getattr(w, 'group', None), getattr(w, 'subject', None),
                 getattr(w, 'session', None)}
        shown.discard(None)
        return bool(shown & objs)

    def _show_fallback_panel(self):
        """After clearing the active item, show another open group, else welcome."""
        from neurokinematics.data.group import ExperimentGroup
        for obj in self._loaded_objects.values():
            if isinstance(obj, ExperimentGroup):
                self._show_panel(GroupPanel(obj, self.log))
                return
        # nothing else to show — return to the opening screen
        while self.stack.count() > 1:
            old = self.stack.widget(1)
            self.stack.removeWidget(old)
            old.deleteLater()
        self.stack.setCurrentWidget(self.welcome_panel)

    def _run_bg(self, label, fn):
        """Run a blocking data operation in a background thread, logging start/finish."""
        from neurokinematics.gui.widgets import Worker
        self.log.log(f"{label} …", 'info')
        # Keep the previous thread/worker referenced until it has truly exited,
        # so a rapid follow-up task (e.g. detect → sort) can't garbage-collect a
        # still-winding-down QThread ("Destroyed while thread is still running").
        prev_t = getattr(self, '_bg_thread', None)
        prev_w = getattr(self, '_bg_worker', None)
        if prev_t is not None:
            keep = getattr(self, '_bg_keepalive', None) or []
            keep.append((prev_t, prev_w))
            self._bg_keepalive = [(t, w) for (t, w) in keep if t.isRunning()]
        self._bg_thread = QThread()
        self._bg_worker = Worker(fn)
        self._bg_worker.moveToThread(self._bg_thread)
        self._bg_thread.started.connect(self._bg_worker.run)
        self._bg_worker.finished.connect(self._bg_thread.quit)
        self._bg_worker.finished.connect(lambda: self.log.log(f"{label} — done.", 'success'))
        # refresh the open detail panel so dots/'done'/outputs update after ops
        # run from the tree menu, spike QC, or the Tools ▸ Analysis dialog
        self._bg_worker.finished.connect(self._refresh_current_panel)
        self._bg_worker.error.connect(self._bg_thread.quit)
        self._bg_worker.error.connect(lambda e: self.log.log(e, 'error'))
        self._bg_thread.start()

    def _rebuild_tree(self):
        """Clear and re-add all loaded objects (e.g. after adding a session)."""
        from neurokinematics.data.group import ExperimentGroup
        from neurokinematics.data.subject import ExperimentSubject
        self.tree.clear()
        for spec_path, obj in list(self._loaded_objects.items()):
            try:
                if isinstance(obj, ExperimentGroup):
                    self._add_group_to_tree(obj, spec_path)
                elif isinstance(obj, ExperimentSubject):
                    self._add_subject_to_tree(obj, spec_path)
            except Exception:
                pass

    # ── Menu actions ──────────────────────────────────────────────────────────

    def _new_group(self):
        dlg = GroupDialog(self)
        if dlg.exec() == GroupDialog.Accepted and dlg.spec_path:
            self._instantiate_and_add_group(
                dlg.spec_path,
                project_path=dlg.project_path,
                project_name=dlg.project_name,
                force_create=dlg.created_new,
            )

    def _load_group(self):
        path = pick_spec_folder(self, 'group')
        if path:
            self._instantiate_and_add_group(path)

    def _new_subject(self):
        dlg = SubjectDialog(self)
        if dlg.exec() == SubjectDialog.Accepted and dlg.spec_path:
            self._instantiate_and_add_subject(
                dlg.spec_path,
                project_path=dlg.project_path,
                project_name=dlg.project_name,
                force_create=dlg.created_new,
            )

    def _load_subject(self):
        path = pick_spec_folder(self, 'subject')
        if path:
            self._instantiate_and_add_subject(path)

    def _load_project(self):
        """Load a project's groups, plus any subjects not already in a group.

        Groups already display their subjects underneath them, so we only add
        *standalone* subjects (those not belonging to a loaded group) to avoid
        showing the same subject twice.
        """
        from neurokinematics.gui.settings import get_default_root
        project_dir = QFileDialog.getExistingDirectory(
            self, "Select Project Directory", get_default_root())
        if not project_dir:
            return
        self._load_project_path(project_dir)

    def _load_project_path(self, project_dir: str):
        from neurokinematics.data.group import ExperimentGroup

        project_path = Path(project_dir)
        n_groups = n_subjects = 0

        # 1) groups (each one also instantiates and shows its own subjects)
        for spec in (project_path / "Groups").rglob("group_spec.yaml"):
            before = len(self._loaded_objects)
            self._instantiate_and_add_group(str(spec))
            n_groups += int(len(self._loaded_objects) > before)

        # 2) collect subject ids already shown under a loaded group
        grouped_ids = set()
        for obj in self._loaded_objects.values():
            if isinstance(obj, ExperimentGroup):
                for subj in obj.subjects:
                    grouped_ids.add(getattr(subj, 'subject_id', None))

        # 3) only add subjects that aren't part of any loaded group
        for spec in (project_path / "Subjects").rglob("subject_spec.yaml"):
            try:
                sid = (yaml.safe_load(spec.read_text()) or {}).get('subject_id')
            except Exception:
                sid = None
            if sid is not None and sid in grouped_ids:
                continue
            before = len(self._loaded_objects)
            self._instantiate_and_add_subject(str(spec))
            n_subjects += int(len(self._loaded_objects) > before)

        self.log.log(
            f"Loaded project from {project_dir}: {n_groups} group(s), "
            f"{n_subjects} standalone subject(s).", 'success'
        )
        if n_groups or n_subjects:
            self._record_recent("project", str(project_path), project_path.name)

    def _new_project(self):
        from neurokinematics.gui.settings import get_default_root
        project_dir = QFileDialog.getExistingDirectory(
            self, "Select Project Root Directory", get_default_root())
        if not project_dir:
            return
        from neurokinematics.data.project import NKProject
        try:
            NKProject(project_dir)
            self.log.log(f"Project created at {project_dir}", 'success')
        except Exception as e:
            self.log.log(f"Could not create project: {e}", 'error')

    def _load_sample_project(self):
        """Generate a small self-contained demo project and load it.

        Builds a group of two subjects with a couple of sessions each under
        ``~/.neurokinematics/SampleProject`` so every panel is populated — handy
        for screenshots and first-run exploration. The session data folders are
        empty placeholders: the constructors only require the paths to exist, so
        no real recordings are needed.
        """
        from neurokinematics.gui.settings import SETTINGS_DIR

        base      = SETTINGS_DIR / "SampleProject"
        pose_dir  = base / "_demo_data" / "pose"
        ephys_dir = base / "_demo_data" / "ephys"
        specs_dir = base / "_demo_specs"
        for d in (pose_dir, ephys_dir, specs_dir):
            d.mkdir(parents=True, exist_ok=True)

        def _subject(sid, sessions):
            return {'spec': {
                'subject_id':  sid,
                'output_root': str(base),
                'process':     {'pose': True, 'spike': True, 'lfp': True},
                'sessions': [
                    {'session_id':      name,
                     'session_config':  None,
                     'ephys_data_path': str(ephys_dir),
                     'pose_data_path':  str(pose_dir)}
                    for name in sessions
                ],
            }}

        group_spec = {
            'group_id':    'DemoGroup',
            'output_root': str(base),
            'subjects': [
                _subject('DEMO-M01', ['Baseline', 'Stim_Day1']),
                _subject('DEMO-M02', ['Baseline', 'Stim_Day1']),
            ],
        }

        spec_path = specs_dir / "demo_group_spec.yaml"
        spec_path.write_text(yaml.safe_dump(group_spec, sort_keys=False))

        if str(spec_path) in self._loaded_objects:
            self.log.log("Sample project already loaded.", 'info')
            return

        self.log.log("Generating sample project…", 'info')
        self._instantiate_and_add_group(
            str(spec_path),
            project_path=str(SETTINGS_DIR),
            project_name="SampleProject",
            force_create=True,
        )

    # ── Instantiation helpers ─────────────────────────────────────────────────

    def _instantiate_and_add_group(self, spec_path: str,
                                   project_path: str | None = None,
                                   project_name: str | None = None,
                                   force_create: bool = False):
        """
        Instantiate a group from a spec and add it to the tree.

        If *force_create* (a brand-new spec) or *project_path* is given, the
        group is created via the full constructor (project root =
        <project_path or home>/<name>). Otherwise we try from_existing() first
        (no stray folders), falling back to inferred instantiation.
        """
        try:
            from neurokinematics.data.group import ExperimentGroup
            spec_path = Path(spec_path)

            if project_path or force_create:
                group = ExperimentGroup(
                    group_specs=str(spec_path),
                    project_path=project_path,
                    name=project_name or 'NK',
                )
            else:
                # from_existing() expects the group folder (parent of the spec file)
                group_folder = spec_path.parent
                try:
                    group = ExperimentGroup.from_existing(group_folder)
                except Exception:
                    # fall back to full init — spec file passed directly
                    with open(spec_path) as f:
                        spec = yaml.safe_load(f)
                    group = ExperimentGroup(
                        group_specs=str(spec_path),
                        project_path=spec_path.parent.parent.parent,  # Groups/ -> project root
                        name='NK'   # default project name — never the group id
                    )

            existing = self._find_loaded_item('group', group.group_id)
            if existing is not None:
                self.log.log(
                    f"Group '{group.group_id}' is already loaded.", 'warning')
                self.tree.setCurrentItem(existing)
                self.tree.scrollToItem(existing)
                return

            self._add_group_to_tree(group, str(spec_path))
            loc = f" in project '{group.project_name}'" if project_path else ""
            self.log.log(f"Loaded group '{group.group_id}'{loc}", 'success')
            self._record_recent("group", str(spec_path), group.group_id,
                                getattr(group, 'project_name', None))
        except Exception as e:
            self.log.log(f"Failed to load group: {e}", 'error')

    def _instantiate_and_add_subject(self, spec_path: str,
                                     project_path: str | None = None,
                                     project_name: str | None = None,
                                     force_create: bool = False):
        """
        Instantiate a subject from a spec and add it to the tree.

        If *force_create* (a brand-new spec) or *project_path* is given, the
        subject is created via the full constructor (root =
        <project_path or home>/<name>). Otherwise from_existing() is tried first
        to avoid stray folder creation, falling back to inferred instantiation.
        """
        try:
            from neurokinematics.data.subject import ExperimentSubject
            spec_path = Path(spec_path)

            if project_path or force_create:
                subject = ExperimentSubject(
                    subject_specs=str(spec_path),
                    project_path=project_path,
                    name=project_name or 'NK',
                )
            else:
                subject_folder = spec_path.parent
                try:
                    subject = ExperimentSubject.from_existing(subject_folder)
                except Exception:
                    with open(spec_path) as f:
                        spec = yaml.safe_load(f)
                    subject = ExperimentSubject(
                        subject_specs=str(spec_path),
                        project_path=spec_path.parent.parent.parent,  # Subjects/ -> project root
                        name='NK'   # default project name — never the subject id
                    )

            existing = self._find_loaded_item('subject', subject.subject_id)
            if existing is not None:
                self.log.log(
                    f"Subject '{subject.subject_id}' is already loaded.", 'warning')
                self.tree.setCurrentItem(existing)
                self.tree.scrollToItem(existing)
                return

            self._add_subject_to_tree(subject, str(spec_path))
            self.log.log(f"Loaded subject '{subject.subject_id}'", 'success')
            self._record_recent("subject", str(spec_path), subject.subject_id,
                                getattr(subject, 'project_name', None))
        except Exception as e:
            self.log.log(f"Failed to load subject: {e}", 'error')


# ── Entry point ───────────────────────────────────────────────────────────────

def launch():
    # On Windows, give the process its own AppUserModelID so the taskbar shows
    # our icon instead of the generic python.exe one.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "neurokinematics.gui"
            )
        except Exception:
            pass

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
