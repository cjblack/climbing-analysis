"""
Dialogs for creating Subject and Group spec files, and viewing dataframes.
"""

import os
import yaml
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QListWidget, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox, QDoubleSpinBox,
    QDialogButtonBox, QFileDialog, QMessageBox, QSizePolicy, QAbstractItemView,
    QComboBox, QTextEdit, QTabWidget, QWidget, QRadioButton
)
from PySide6.QtCore import Qt

from neurokinematics.gui.widgets import PathField, HDivider


class ProjectSelector(QGroupBox):
    """Project chooser used by the subject/group builders.

    Two modes: *create new* (name + parent location) or *use existing* (point at
    an existing project folder). ``resolve()`` returns ``(project_name,
    project_path)`` suitable for the data-layer constructors — for an existing
    project the name/parent are derived from the chosen folder.
    """

    def __init__(self, parent=None):
        super().__init__("Project", parent)
        from neurokinematics.gui.settings import get_default_root
        default_root = get_default_root()

        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        self._new_radio      = QRadioButton("Create new project")
        self._existing_radio = QRadioButton("Use existing project")
        self._new_radio.setChecked(True)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self._new_radio)
        mode_row.addWidget(self._existing_radio)
        mode_row.addStretch()
        outer.addLayout(mode_row)

        # create-new container (Name + Location)
        self._new_box = QWidget()
        nf = QFormLayout(self._new_box)
        nf.setContentsMargins(0, 0, 0, 0)
        nf.setLabelAlignment(Qt.AlignRight)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("NK")
        self._location = PathField(mode='folder')
        self._location.setText(default_root)
        nf.addRow("Project Name:",     self._name_edit)
        nf.addRow("Project Location:", self._location)
        outer.addWidget(self._new_box)

        # use-existing container (single folder picker)
        self._existing_box = QWidget()
        ef = QFormLayout(self._existing_box)
        ef.setContentsMargins(0, 0, 0, 0)
        ef.setLabelAlignment(Qt.AlignRight)
        self._existing = PathField(mode='folder')
        self._existing.setText(default_root)
        ef.addRow("Existing project:", self._existing)
        outer.addWidget(self._existing_box)

        self._new_radio.toggled.connect(self._sync)
        self._sync()

    def _sync(self):
        new = self._new_radio.isChecked()
        self._new_box.setVisible(new)
        self._existing_box.setVisible(not new)

    def resolve(self):
        """Return (project_name, project_path); blanks fall back to None."""
        if self._existing_radio.isChecked():
            folder = self._existing.text().strip()
            if folder:
                p = Path(folder)
                return p.name, str(p.parent)
            return None, None
        name = self._name_edit.text().strip() or None
        loc  = self._location.text().strip() or None
        return name, loc


# spec filename expected inside a group / subject folder
SPEC_FILENAMES = {'group': 'group_spec.yaml', 'subject': 'subject_spec.yaml'}


def pick_spec_folder(parent, kind: str):
    """Prompt for a *kind* folder and return the spec yaml inside it, or None.

    *kind* is 'group' or 'subject'. The user selects the folder (named after the
    group/subject) that contains its spec file; we resolve to
    ``<folder>/<spec>.yaml``. If no spec file is present, an error is shown and
    None is returned.
    """
    spec_name = SPEC_FILENAMES[kind]
    from neurokinematics.gui.settings import get_default_root
    folder = QFileDialog.getExistingDirectory(
        parent, f"Select {kind.capitalize()} Folder", get_default_root())
    if not folder:
        return None
    spec = Path(folder) / spec_name
    if not spec.exists():
        QMessageBox.critical(
            parent, "No Spec File",
            f"No {spec_name} found in:\n{folder}\n\n"
            f"Select the {kind}'s folder that contains its {spec_name}."
        )
        return None
    return str(spec)


def _write_temp_spec(spec: dict, kind: str) -> str:
    """Write a freshly-built spec dict to a temp file and return its path.

    The canonical spec is (re)written inside the group/subject folder on
    instantiation, so this throwaway file just feeds the constructor — no need
    to prompt the user for a save location.
    """
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.yaml', prefix=f'nk_{kind}_spec_')
    os.close(fd)
    with open(path, 'w') as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
    return path


# ── Session dialog (used inside Subject dialog) ───────────────────────────────

class SessionDialog(QDialog):
    """Modal dialog for adding a single session to a subject spec.

    Besides the session config, the associated sub-configs (spikes / pose / lfp
    / multimodal / models) can be overridden via dropdowns. If any are changed
    from what the chosen session config references, a new session config is
    written with a ``_custom_V<n>`` suffix (auto-incremented) so existing
    configs are never overwritten.
    """

    DEFAULT_SESSION_CFG = "demo_session.yaml"

    # (label, session-config key, CFG_PATHS category)
    SUBCFG_FIELDS = [
        ("Spikes",     "spikes",      "spksorting"),
        ("Pose",       "pose",        "pose"),
        ("LFP",        "lfp",         "lfp"),
        ("Multimodal", "multi_modal", "multimodal"),
        ("Models",     "models",      "models"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Session")
        self.setMinimumWidth(520)
        self.result_data = None
        self._build_ui()

    def _build_ui(self):
        from neurokinematics.io import CFG_PATHS
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(8)

        self.session_id     = QLineEdit()
        self.session_config = PathField(
            mode='file',
            placeholder=self.DEFAULT_SESSION_CFG,
            start_dir=str(CFG_PATHS.get('session', ''))
        )
        from neurokinematics.gui.settings import get_data_root
        self.ephys_path     = PathField(mode='folder', placeholder="Path to Open Ephys data",
                                        start_dir=get_data_root("ephys"))
        self.pose_path      = PathField(mode='folder', placeholder="Path to SLEAP .h5 files",
                                        start_dir=get_data_root("pose"))

        self.session_id.setPlaceholderText("e.g. session_001")

        form.addRow("Session ID:",      self.session_id)
        form.addRow("Session Config:",  self.session_config)
        form.addRow("Ephys Data Path:", self.ephys_path)
        form.addRow("Pose Data Path:",  self.pose_path)
        layout.addLayout(form)

        # ── associated sub-config overrides ──
        cfg_box  = QGroupBox("Associated Configs (override)")
        cfg_form = QFormLayout(cfg_box)
        cfg_form.setLabelAlignment(Qt.AlignRight)
        self._subcfg_combos = {}
        for label, key, category in self.SUBCFG_FIELDS:
            combo = QComboBox()
            combo.addItems(self._list_cfgs(category))
            cfg_form.addRow(f"{label}:", combo)
            self._subcfg_combos[key] = (combo, category)
        layout.addWidget(cfg_box)

        layout.addWidget(HDivider())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # keep the dropdowns in sync with the chosen session config
        self.session_config.edit.textChanged.connect(lambda *_: self._sync_subconfigs())
        self._sync_subconfigs()

    # ── helpers ──
    @staticmethod
    def _list_cfgs(category):
        from neurokinematics.io import CFG_PATHS
        d = CFG_PATHS.get(category)
        if d and Path(d).exists():
            return sorted(f.name for f in Path(d).glob("*.yaml"))
        return []

    def _base_session_cfg_name(self):
        return self.session_config.text().strip() or self.DEFAULT_SESSION_CFG

    def _load_base_session_cfg(self):
        from neurokinematics.io import load_config
        try:
            cfg = load_config(self._base_session_cfg_name(), config_type='session')
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def _sync_subconfigs(self):
        """Pre-select each dropdown from the chosen session config's references."""
        refs = self._load_base_session_cfg().get('configs', {})
        for key, (combo, _cat) in self._subcfg_combos.items():
            ref = refs.get(key)
            if not ref:
                continue
            idx = combo.findText(ref)
            if idx < 0:
                combo.addItem(ref)
                idx = combo.findText(ref)
            combo.setCurrentIndex(idx)

    @staticmethod
    def _next_custom_name(base_stem):
        import re
        from neurokinematics.io import CFG_PATHS
        base_stem = re.sub(r'_custom_V\d+$', '', base_stem)   # don't stack suffixes
        i = 1
        while (CFG_PATHS['session'] / f"{base_stem}_custom_V{i}.yaml").exists():
            i += 1
        return f"{base_stem}_custom_V{i}.yaml"

    def _resolve_session_config(self):
        """Return the session-config name to store, writing a custom version if
        any sub-config dropdown differs from the base session config."""
        import copy
        from neurokinematics.io import CFG_PATHS, save_yaml

        base_name = self._base_session_cfg_name()
        base = self._load_base_session_cfg()
        if 'configs' not in base:
            # can't introspect (odd/missing config) — just pass the name through
            return base_name

        selected = {key: combo.currentText()
                    for key, (combo, _c) in self._subcfg_combos.items()
                    if combo.currentText()}
        current = base.get('configs', {})
        if all(current.get(k) == v for k, v in selected.items()):
            return base_name                       # nothing changed

        new_cfg = copy.deepcopy(base)
        new_cfg.setdefault('configs', {}).update(selected)
        new_name = self._next_custom_name(Path(base_name).stem)
        save_yaml(new_cfg, CFG_PATHS['session'] / new_name)
        return new_name

    def _confirm(self):
        session_id = self.session_id.text().strip()
        if not session_id:
            QMessageBox.warning(self, "Missing Field", "Session ID is required.")
            return

        self.result_data = {
            'session_id':      session_id,
            'session_config':  self._resolve_session_config(),
            'ephys_data_path': self.ephys_path.text() or None,
            'pose_data_path':  self.pose_path.text() or None,
        }
        self.accept()


# ── Subject dialog ────────────────────────────────────────────────────────────

class SubjectDialog(QDialog):
    """
    Dialog for creating a new Subject spec file, or loading an existing one.
    On accept, writes the spec YAML and returns the path via self.spec_path.
    """

    def __init__(self, parent=None, load_path: str = None):
        super().__init__(parent)
        self.setWindowTitle("Create Subject" if not load_path else "Load Subject")
        self.setMinimumWidth(560)
        self.setMinimumHeight(480)
        self.sessions = []
        self.spec_path = None
        self.project_name = None     # set on accept from the Project fields
        self.project_path = None
        self.created_new = False     # True when a brand-new spec was built here
        self._build_ui()

        if load_path:
            self._load_existing(load_path)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Instantiate directly from an existing subject spec ──
        existing_row = QHBoxLayout()
        existing_hint = QLabel("Already have a subject spec file?")
        existing_hint.setObjectName("subheading")
        existing_btn = QPushButton("Create from Subject Spec File…")
        existing_btn.clicked.connect(self._use_existing_spec)
        existing_row.addWidget(existing_hint)
        existing_row.addStretch()
        existing_row.addWidget(existing_btn)
        layout.addLayout(existing_row)
        layout.addWidget(HDivider())

        # Project: create a new one or point at an existing project folder
        self.project_selector = ProjectSelector()
        layout.addWidget(self.project_selector)

        # Subject info
        subj_group = QGroupBox("Subject")
        subj_form = QFormLayout(subj_group)
        subj_form.setLabelAlignment(Qt.AlignRight)
        subj_form.setSpacing(8)

        self.subject_id  = QLineEdit()
        self.subject_id.setPlaceholderText("e.g. SK01")

        subj_form.addRow("Subject ID:",   self.subject_id)
        layout.addWidget(subj_group)

        # Process flags
        proc_group = QGroupBox("Process")
        proc_layout = QHBoxLayout(proc_group)

        self.proc_pose  = QCheckBox("Pose")
        self.proc_spike = QCheckBox("Spikes")
        self.proc_lfp   = QCheckBox("LFP")

        for cb in [self.proc_pose, self.proc_spike, self.proc_lfp]:
            cb.setChecked(True)
            proc_layout.addWidget(cb)
        proc_layout.addStretch()
        layout.addWidget(proc_group)

        # Sessions
        sess_group = QGroupBox("Sessions")
        sess_layout = QVBoxLayout(sess_group)

        self.session_list = QListWidget()
        self.session_list.setMinimumHeight(100)
        sess_layout.addWidget(self.session_list)

        btn_row = QHBoxLayout()
        add_btn    = QPushButton("+ Add Session")
        remove_btn = QPushButton("− Remove Selected")
        remove_btn.setObjectName("secondary")
        add_btn.clicked.connect(self._add_session)
        remove_btn.clicked.connect(self._remove_session)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        sess_layout.addLayout(btn_row)
        layout.addWidget(sess_group)

        layout.addWidget(HDivider())

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _capture_project(self):
        """Resolve the Project selector into project_name / project_path."""
        self.project_name, self.project_path = self.project_selector.resolve()

    def _use_existing_spec(self):
        """Create a subject from an existing subject_spec .yaml file.

        Pick the spec file directly (vs. the folder-based 'load existing' flow);
        the subject is then created from it in the chosen project.
        """
        from neurokinematics.gui.settings import get_default_root
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Subject Spec File", get_default_root(),
            "YAML Files (*.yaml *.yml)")
        if path:
            self.spec_path = path
            self.created_new = True
            self._capture_project()
            self.accept()

    def _add_session(self):
        dlg = SessionDialog(self)
        if dlg.exec() == QDialog.Accepted and dlg.result_data:
            self.sessions.append(dlg.result_data)
            self.session_list.addItem(dlg.result_data['session_id'])

    def _remove_session(self):
        idx = self.session_list.currentRow()
        if idx >= 0:
            self.session_list.takeItem(idx)
            self.sessions.pop(idx)

    def _load_existing(self, path: str):
        with open(path) as f:
            spec = yaml.safe_load(f)
        self.subject_id.setText(spec.get('subject_id', ''))
        proc = spec.get('process', {})
        self.proc_pose.setChecked(proc.get('pose', True))
        self.proc_spike.setChecked(proc.get('spike', True))
        self.proc_lfp.setChecked(proc.get('lfp', True))
        for sess in spec.get('sessions') or []:
            self.sessions.append(sess)
            self.session_list.addItem(sess.get('session_id', ''))

    def _save(self):
        subject_id = self.subject_id.text().strip()
        if not subject_id:
            QMessageBox.warning(self, "Missing Field", "Subject ID is required.")
            return

        self._capture_project()
        spec = {
            'subject_id':  subject_id,
            # output_root is deprecated in the data classes; keep the key for
            # spec validity and default it to the project location behind the scenes
            'output_root': self.project_path or '',
            'process': {
                'pose':  self.proc_pose.isChecked(),
                'spike': self.proc_spike.isChecked(),
                'lfp':   self.proc_lfp.isChecked(),
            },
            'sessions': self.sessions or None,
        }
        # the canonical subject_spec.yaml is written inside the subject's folder
        # on instantiation, so no need to prompt for a save location here
        self.spec_path = _write_temp_spec(spec, 'subject')
        self.created_new = True
        self.accept()


# ── Group dialog ──────────────────────────────────────────────────────────────

class GroupDialog(QDialog):
    """
    Dialog for creating a new Group spec file, or loading an existing one.
    Subjects are added by browsing for existing subject spec YAML files.
    On accept, writes the spec YAML and returns the path via self.spec_path.
    """

    def __init__(self, parent=None, load_path: str = None):
        super().__init__(parent)
        self.setWindowTitle("Create Group" if not load_path else "Load Group")
        self.setMinimumWidth(560)
        self.setMinimumHeight(420)
        self.subject_specs = []
        self.spec_path = None
        self.project_name = None     # set on accept from the Project fields
        self.project_path = None
        self.created_new = False     # True when a brand-new spec was built here
        self._build_ui()

        if load_path:
            self._load_existing(load_path)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Instantiate directly from an existing group spec ──
        existing_row = QHBoxLayout()
        existing_hint = QLabel("Already have a group spec file?")
        existing_hint.setObjectName("subheading")
        existing_btn = QPushButton("Create from Group Spec File…")
        existing_btn.clicked.connect(self._use_existing_spec)
        existing_row.addWidget(existing_hint)
        existing_row.addStretch()
        existing_row.addWidget(existing_btn)
        layout.addLayout(existing_row)
        layout.addWidget(HDivider())

        # Project: create a new one or point at an existing project folder
        self.project_selector = ProjectSelector()
        layout.addWidget(self.project_selector)

        # Group info
        grp_group = QGroupBox("Group")
        grp_form  = QFormLayout(grp_group)
        grp_form.setLabelAlignment(Qt.AlignRight)
        grp_form.setSpacing(8)

        self.group_id    = QLineEdit()
        self.group_id.setPlaceholderText("e.g. climbing_group")

        grp_form.addRow("Group ID:",    self.group_id)
        layout.addWidget(grp_group)

        # Subjects
        subj_group  = QGroupBox("Subjects")
        subj_layout = QVBoxLayout(subj_group)

        self.subject_list = QListWidget()
        self.subject_list.setMinimumHeight(120)
        subj_layout.addWidget(self.subject_list)

        btn_row    = QHBoxLayout()
        add_btn    = QPushButton("+ Add Existing Spec")
        create_btn = QPushButton("+ Create Subject Spec")
        remove_btn = QPushButton("− Remove Selected")
        remove_btn.setObjectName("secondary")
        add_btn.clicked.connect(self._add_subject)
        create_btn.clicked.connect(self._create_subject)
        remove_btn.clicked.connect(self._remove_subject)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(create_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        subj_layout.addLayout(btn_row)
        layout.addWidget(subj_group)

        layout.addWidget(HDivider())

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _capture_project(self):
        """Resolve the Project selector into project_name / project_path."""
        self.project_name, self.project_path = self.project_selector.resolve()

    def _use_existing_spec(self):
        """Create a group from an existing group_spec .yaml file.

        Pick the spec file directly (vs. the folder-based 'load existing' flow);
        the group is then created from it in the chosen project.
        """
        from neurokinematics.gui.settings import get_default_root
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Group Spec File", get_default_root(),
            "YAML Files (*.yaml *.yml)")
        if path:
            self.spec_path = path
            self.created_new = True
            self._capture_project()
            self.accept()

    def _add_subject(self):
        path = pick_spec_folder(self, 'subject')
        if path:
            self.subject_specs.append({'spec': path})
            # label with the subject's folder name
            self.subject_list.addItem(Path(path).parent.name)

    def _create_subject(self):
        """Create a new subject spec inline and add it to the group."""
        dlg = SubjectDialog(parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.spec_path:
            self.subject_specs.append({'spec': dlg.spec_path})
            self.subject_list.addItem(Path(dlg.spec_path).stem)

    def _remove_subject(self):
        idx = self.subject_list.currentRow()
        if idx >= 0:
            self.subject_list.takeItem(idx)
            self.subject_specs.pop(idx)

    def _load_existing(self, path: str):
        with open(path) as f:
            spec = yaml.safe_load(f)
        self.group_id.setText(spec.get('group_id', ''))
        for subj in spec.get('subjects') or []:
            self.subject_specs.append(subj)
            self.subject_list.addItem(Path(subj.get('spec', '')).stem)

    def _save(self):
        group_id = self.group_id.text().strip()
        if not group_id:
            QMessageBox.warning(self, "Missing Field", "Group ID is required.")
            return
        if not self.subject_specs:
            QMessageBox.warning(self, "No Subjects", "Add at least one subject spec.")
            return

        self._capture_project()
        spec = {
            'group_id':    group_id,
            # output_root is deprecated in the data classes; keep the key and
            # default it to the project location behind the scenes
            'output_root': self.project_path or '',
            'subjects':    self.subject_specs,
        }
        # canonical group_spec.yaml is written in the group's folder on
        # instantiation, so no save prompt is needed here
        self.spec_path = _write_temp_spec(spec, 'group')
        self.created_new = True
        self.accept()


# ── Settings dialog ───────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """Edit persistent GUI settings (currently the phy2 setup)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)
        from neurokinematics.gui.settings import load_settings
        self._settings = load_settings()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        phy_box  = QGroupBox("phy2")
        phy_form = QFormLayout(phy_box)
        phy_form.setLabelAlignment(Qt.AlignRight)

        self.phy_env  = QLineEdit(self._settings.get("phy_env", ""))
        self.phy_env.setPlaceholderText("conda env where phy2 is installed, e.g. phy2")
        self.phy_gui  = QLineEdit(self._settings.get("phy_gui", "template-gui"))
        self.phy_gui.setPlaceholderText("template-gui")
        self.conda_exe = QLineEdit(self._settings.get("conda_exe", "conda"))
        self.conda_exe.setPlaceholderText("conda  (or full path to conda)")

        phy_form.addRow("phy2 environment:", self.phy_env)
        phy_form.addRow("phy GUI:",          self.phy_gui)
        phy_form.addRow("conda command:",    self.conda_exe)
        layout.addWidget(phy_box)

        hint = QLabel("Right-click a session's processed <b>Spikes</b> row to open phy2.")
        hint.setObjectName("subheading")
        layout.addWidget(hint)

        # Paths
        paths_box  = QGroupBox("Paths")
        paths_form = QFormLayout(paths_box)
        paths_form.setLabelAlignment(Qt.AlignRight)
        self.default_root = PathField(mode='folder')
        self.default_root.setText(self._settings.get("default_root", ""))
        paths_form.addRow("Default project root:", self.default_root)

        self.default_ephys_root = PathField(mode='folder')
        self.default_ephys_root.setText(self._settings.get("default_ephys_root", ""))
        paths_form.addRow("Default ephys data root:", self.default_ephys_root)

        self.default_pose_root = PathField(mode='folder')
        self.default_pose_root.setText(self._settings.get("default_pose_root", ""))
        paths_form.addRow("Default pose data root:", self.default_pose_root)
        layout.addWidget(paths_box)

        root_hint = QLabel(
            "New projects, the spec builders, and Open/Load dialogs start at the "
            "project root. Ephys/pose roots set where the 'add data' file dialogs "
            "open. Leave blank to use your home folder.")
        root_hint.setObjectName("subheading")
        root_hint.setWordWrap(True)
        layout.addWidget(root_hint)

        # Spike sorting
        spk_box  = QGroupBox("Spike sorting")
        spk_form = QFormLayout(spk_box)
        spk_form.setLabelAlignment(Qt.AlignRight)
        self.spike_policy = QComboBox()
        self.spike_policy.addItems(["ask", "remove", "keep"])
        cur = self._settings.get("spike_bad_channel_policy", "ask")
        i = self.spike_policy.findText(cur)
        self.spike_policy.setCurrentIndex(i if i >= 0 else 0)
        spk_form.addRow("Bad-channel handling:", self.spike_policy)
        layout.addWidget(spk_box)

        spk_hint = QLabel(
            "<b>ask</b>: review detected bad channels before sorting.&nbsp; "
            "<b>remove</b>/<b>keep</b>: skip the review and sort automatically.")
        spk_hint.setObjectName("subheading")
        spk_hint.setWordWrap(True)
        layout.addWidget(spk_hint)

        layout.addWidget(HDivider())
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self):
        from neurokinematics.gui.settings import save_settings
        save_settings({
            "phy_env":   self.phy_env.text().strip(),
            "phy_gui":   self.phy_gui.text().strip() or "template-gui",
            "conda_exe": self.conda_exe.text().strip() or "conda",
            "default_root": self.default_root.text().strip(),
            "default_ephys_root": self.default_ephys_root.text().strip(),
            "default_pose_root": self.default_pose_root.text().strip(),
            "spike_bad_channel_policy": self.spike_policy.currentText(),
        })
        self.accept()


# ── DataFrame viewer ──────────────────────────────────────────────────────────

class DataFrameDialog(QDialog):
    """
    Shows the head of a pandas DataFrame (or parquet file) in a table.
    Lets the user choose how many rows to display.
    """

    def __init__(self, source, title: str = "DataFrame", parent=None):
        """
        Parameters
        ----------
        source : str | Path | pd.DataFrame
            Either a path to a parquet file, or an already-loaded DataFrame.
        title : str
            Window title.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 480)
        self._source = source
        self._df     = None
        self._build_ui()
        self._load()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # controls row
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Show rows:"))

        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 1000)
        self._n_spin.setValue(20)
        self._n_spin.valueChanged.connect(self._refresh_table)
        ctrl.addWidget(self._n_spin)

        self._shape_lbl = QLabel()
        self._shape_lbl.setObjectName("subheading")
        ctrl.addStretch()
        ctrl.addWidget(self._shape_lbl)

        layout.addLayout(ctrl)

        # table
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(True)
        layout.addWidget(self._table)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _load(self):
        import pandas as pd
        try:
            if isinstance(self._source, (str, Path)):
                from neurokinematics.io import load_file
                result = load_file(self._source, method='pandas')
                # load_file may return a DataFrame or something else (e.g. zarr)
                if isinstance(result, pd.DataFrame):
                    self._df = result
                elif hasattr(result, 'to_dataframe'):
                    # xarray Dataset
                    self._df = result.to_dataframe().reset_index()
                else:
                    self._shape_lbl.setText(f"Cannot display type: {type(result).__name__}")
                    return
            else:
                self._df = self._source
            self._shape_lbl.setText(
                f"{self._df.shape[0]:,} rows × {self._df.shape[1]} columns"
            )
            self._refresh_table()
        except Exception as e:
            self._shape_lbl.setText(f"Error loading data: {e}")

    def _refresh_table(self):
        if self._df is None:
            return

        n  = self._n_spin.value()
        df = self._df.head(n)

        self._table.clear()
        self._table.setRowCount(len(df))
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for i, row in enumerate(df.itertuples(index=False)):
            for j, val in enumerate(row):
                text = f"{val:.4f}" if isinstance(val, float) else str(val)
                self._table.setItem(i, j, QTableWidgetItem(text))


# ── Analysis dialog ───────────────────────────────────────────────────────────

# Default parameters for each model
_MODEL_DEFAULTS = {
    'bayesian': {
        'hierarchical_linear': {
            'node':      'r_forepaw',
            'feature':   'v_mag_max',
            'predictor': ['session_number'],
            'samples':   2000,
            'tune':      1500,
            'chains':    4,
            'seed':      42,
            'target_accept': 0.9,
            'likelihood': 'Normal',
            'priors': {
                # NOTE: group_baseline mu/sigma should match your data scale.
                # Check df[feature].mean() and set mu accordingly.
                # With a single predictor, data is NOT standardised — use raw units.
                'group_baseline':   {'dist': 'Normal',     'mu': 0,   'sigma': 50},
                'group_slope':      {'dist': 'Normal',     'mu': 0,   'sigma': 10},
                'sigma_baseline':   {'dist': 'HalfNormal', 'sigma': 20},
                'sigma_slope':      {'dist': 'HalfNormal', 'sigma': 5},
                'sigma_obs':        {'dist': 'HalfNormal', 'sigma': 20},
                'subject_baseline': {'dist': 'Normal'},
                'subject_slope':    {'dist': 'Normal'},
            }
        }
    },
    'frequentist': {}
}

_FRAMEWORK_MODELS = {
    'bayesian':    ['hierarchical_linear'],
    'frequentist': [],
}


class AnalysisDialog(QDialog):
    """
    Dialog for configuring and launching a statistical analysis via group.analyse().

    Shows framework / model selectors and an editable YAML parameter editor.
    On accept, returns (framework, model, params) via self.result.
    """

    def __init__(self, group, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Analysis")
        self.setMinimumWidth(560)
        self.setMinimumHeight(520)
        self.group    = group
        self.result   = None  # (framework, model, params) on accept
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Data source ──
        data_group = QGroupBox("Data")
        data_form  = QFormLayout(data_group)
        data_form.setSpacing(8)

        summaries_dir = self.group.dirs.get('summaries', None)
        available = []
        if summaries_dir:
            from pathlib import Path as _Path
            for p in _Path(summaries_dir).glob("*.parquet"):
                available.append(p.name)

        self._data_combo = QComboBox()
        self._data_combo.addItems(available if available else ["(no summaries found)"])
        data_form.addRow("Summary file:", self._data_combo)
        layout.addWidget(data_group)

        # ── Model selection ──
        model_group = QGroupBox("Model")
        model_form  = QFormLayout(model_group)
        model_form.setSpacing(8)

        self._framework_combo = QComboBox()
        self._framework_combo.addItems(list(_FRAMEWORK_MODELS.keys()))
        self._framework_combo.currentTextChanged.connect(self._on_framework_changed)
        model_form.addRow("Framework:", self._framework_combo)

        self._model_combo = QComboBox()
        model_form.addRow("Model:", self._model_combo)

        layout.addWidget(model_group)

        # ── Parameters (YAML editor) ──
        param_group = QGroupBox("Parameters  (YAML)")
        param_layout = QVBoxLayout(param_group)

        self._param_edit = QTextEdit()
        self._param_edit.setMinimumHeight(180)
        self._param_edit.setPlaceholderText("Parameters as YAML...")
        param_layout.addWidget(self._param_edit)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.setObjectName("secondary")
        reset_btn.clicked.connect(self._reset_params)
        param_layout.addWidget(reset_btn)

        layout.addWidget(param_group)

        # ── Buttons ──
        from neurokinematics.gui.widgets import HDivider
        layout.addWidget(HDivider())
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Run Analysis")
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # initialise model combo and params
        self._on_framework_changed(self._framework_combo.currentText())

    def _on_framework_changed(self, framework: str):
        self._model_combo.clear()
        models = _FRAMEWORK_MODELS.get(framework, [])
        if models:
            self._model_combo.addItems(models)
        else:
            self._model_combo.addItem("(none available)")
        self._reset_params()

    def _reset_params(self):
        framework = self._framework_combo.currentText()
        model     = self._model_combo.currentText()
        defaults  = _MODEL_DEFAULTS.get(framework, {}).get(model, {})
        self._param_edit.setPlainText(
            yaml.dump(defaults, default_flow_style=False, sort_keys=False)
        )

    def _confirm(self):
        framework = self._framework_combo.currentText()
        model     = self._model_combo.currentText()
        data_file = self._data_combo.currentText()

        if data_file.startswith("("):
            QMessageBox.warning(self, "No Data", "No summary parquet files available.\nRun group.summarize() first.")
            return

        try:
            params = yaml.safe_load(self._param_edit.toPlainText()) or {}
        except yaml.YAMLError as e:
            QMessageBox.critical(self, "YAML Error", f"Invalid YAML in parameters:\n{e}")
            return

        self.result = (framework, model, data_file, params)
        self.accept()


# ── GLM encoder configuration ─────────────────────────────────────────────────

# pose features the encoder can use, in the 'feature_coord' form create_glm_encoder parses
_ENCODER_FEATURES = [
    "velocity_x", "velocity_y", "speed",
    "position_x", "position_y",
    "acceleration_x", "acceleration_y",
]


def _discover_binned_stores(pose_dir, spikes_dir):
    """Pair resampled-pose and spike-count zarr stores by their bin-size suffix.

    Returns a dict ``{bin_ms: (pose_path, spike_path)}`` for bin sizes that have
    *both* stores present, so the encoder always has matching X and y.
    """
    import re
    pose_dir, spikes_dir = Path(pose_dir), Path(spikes_dir)

    def _by_ms(folder, pattern):
        out = {}
        if folder.exists():
            for p in folder.glob(pattern):
                m = re.search(r"_(\d+)ms\.zarr$", p.name)
                if m:
                    out[int(m.group(1))] = p
        return out

    pose = _by_ms(pose_dir, "resampled_movements_*ms.zarr")
    spk = _by_ms(spikes_dir, "movement_spike_counts_*ms.zarr")
    return {ms: (pose[ms], spk[ms]) for ms in sorted(set(pose) & set(spk))}


class EncoderDialog(QDialog):
    """Configure and launch a GLM encoder (pose -> single-unit spiking).

    Discovers the binned pose/spike zarr stores produced by
    ``session.bin_movements_and_spikes`` and lets the user pick the reference
    node, unit, pose features, an optional raised-cosine temporal basis (to
    recover lead/lag), and a model-comparison mode. On accept,
    ``self.result`` holds ``(pose_path, spike_path, params)``.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit GLM Encoder")
        self.setMinimumWidth(520)
        self.session = session
        self.result = None
        self._stores = _discover_binned_stores(
            session.dirs.get("pose"), session.dirs.get("spikes"))
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Binned data source ──
        data_group = QGroupBox("Binned data")
        data_form = QFormLayout(data_group)
        self._bin_combo = QComboBox()
        if self._stores:
            self._bin_combo.addItems([f"{ms} ms" for ms in self._stores])
        else:
            self._bin_combo.addItem("(none — run Bin first)")
            self._bin_combo.setEnabled(False)
        self._bin_combo.currentIndexChanged.connect(self._on_bin_changed)
        data_form.addRow("Bin size:", self._bin_combo)
        self._pre_label = QLabel("—")
        self._pre_label.setObjectName("subheading")
        data_form.addRow("Pre-movement bins:", self._pre_label)
        layout.addWidget(data_group)

        # ── Target ──
        target_group = QGroupBox("Target")
        target_form = QFormLayout(target_group)
        self._node_combo = QComboBox()
        self._unit_combo = QComboBox()
        target_form.addRow("Reference node:", self._node_combo)
        target_form.addRow("Unit:", self._unit_combo)
        layout.addWidget(target_group)

        # ── Pose features ──
        feat_group = QGroupBox("Pose features (predictors)")
        feat_layout = QHBoxLayout(feat_group)
        self._feat_checks = {}
        col = QVBoxLayout()
        for i, feat in enumerate(_ENCODER_FEATURES):
            cb = QCheckBox(feat)
            cb.setChecked(feat in ("velocity_x", "velocity_y"))
            self._feat_checks[feat] = cb
            col.addWidget(cb)
            if (i + 1) % 4 == 0:
                feat_layout.addLayout(col)
                col = QVBoxLayout()
        feat_layout.addLayout(col)
        layout.addWidget(feat_group)

        # ── Temporal basis (lags) ──
        self._basis_group = QGroupBox("Temporal basis (recover lead/lag)")
        self._basis_group.setCheckable(True)
        self._basis_group.setChecked(True)
        basis_form = QFormLayout(self._basis_group)

        self._win_min = QDoubleSpinBox()
        self._win_min.setRange(-2.0, 2.0)
        self._win_min.setSingleStep(0.02)
        self._win_min.setDecimals(3)
        self._win_min.setValue(-0.10)
        self._win_max = QDoubleSpinBox()
        self._win_max.setRange(-2.0, 2.0)
        self._win_max.setSingleStep(0.02)
        self._win_max.setDecimals(3)
        self._win_max.setValue(0.20)
        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("from"))
        win_row.addWidget(self._win_min)
        win_row.addWidget(QLabel("to"))
        win_row.addWidget(self._win_max)
        win_row.addWidget(QLabel("s"))
        win_widget = QWidget()
        win_widget.setLayout(win_row)
        basis_form.addRow("Window (rel. spike):", win_widget)

        self._n_basis = QSpinBox()
        self._n_basis.setRange(1, 15)
        self._n_basis.setValue(5)
        basis_form.addRow("# basis functions:", self._n_basis)

        self._spacing_combo = QComboBox()
        self._spacing_combo.addItems(["linear", "log"])
        basis_form.addRow("Spacing:", self._spacing_combo)

        hint = QLabel("Window > 0 = unit leads movement; < 0 = unit lags.")
        hint.setObjectName("subheading")
        basis_form.addRow(hint)
        layout.addWidget(self._basis_group)

        # ── Model ──
        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)
        self._family_combo = QComboBox()
        self._family_combo.addItems(["Poisson", "Gaussian"])
        model_form.addRow("Family:", self._family_combo)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["full", "single", "single_and_full", "drop_one"])
        model_form.addRow("Comparison:", self._mode_combo)
        self._cv_spin = QSpinBox()
        self._cv_spin.setRange(0, 20)
        self._cv_spin.setValue(5)
        self._cv_spin.setToolTip("Event-grouped cross-validation folds. 0 = in-sample fit.")
        model_form.addRow("CV folds (0 = in-sample):", self._cv_spin)
        layout.addWidget(model_group)

        # ── Buttons ──
        layout.addWidget(HDivider())
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Fit Encoder")
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if self._stores:
            self._on_bin_changed(0)

    def _current_paths(self):
        if not self._stores:
            return None, None
        ms = list(self._stores.keys())[max(0, self._bin_combo.currentIndex())]
        return self._stores[ms]

    def _on_bin_changed(self, _idx):
        """Populate node/unit combos from the selected stores' coordinates."""
        pose_path, spike_path = self._current_paths()
        if pose_path is None:
            return
        from neurokinematics.io import load_zarr
        self._node_combo.clear()
        self._unit_combo.clear()
        try:
            pose_ds = load_zarr(pose_path, method="xarray")
            nodes = [str(n) for n in pose_ds.node.values]
            self._node_combo.addItems(nodes)
        except Exception:
            self._node_combo.setEditable(True)
        try:
            import numpy as np
            spike_ds = load_zarr(spike_path, method="xarray")
            self._unit_combo.addItems([str(int(u)) for u in spike_ds.unit.values])
            # surface whether this binned data carries pre-movement labelling
            if 'pre_movement' in spike_ds:
                pm = spike_ds['pre_movement'].values
                val = spike_ds['valid'].values if 'valid' in spike_ds else np.ones_like(pm, bool)
                if val.any() and pm[val].any():
                    self._pre_label.setText(f"present ({100 * pm[val].mean():.0f}% of valid bins)")
                else:
                    self._pre_label.setText("none — re-Bin with a pre-movement window to add")
            else:
                self._pre_label.setText("none — re-Bin with a pre-movement window to add")
        except Exception:
            self._unit_combo.setEditable(True)
            self._pre_label.setText("—")

    def _confirm(self):
        from neurokinematics.models.glm import build_encoder_params

        pose_path, spike_path = self._current_paths()
        if pose_path is None:
            QMessageBox.warning(
                self, "No binned data",
                "No matching pose/spike zarr stores were found.\n"
                "Run 'Bin' to generate them first.")
            return

        features = [f for f, cb in self._feat_checks.items() if cb.isChecked()]
        if not features:
            QMessageBox.warning(self, "No features",
                                "Select at least one pose feature.")
            return

        if self._win_max.value() < self._win_min.value():
            QMessageBox.warning(self, "Invalid window",
                                "Window end must be ≥ window start.")
            return

        node = self._node_combo.currentText()
        try:
            unit = int(self._unit_combo.currentText())
        except (ValueError, TypeError):
            QMessageBox.warning(self, "No unit", "Select a valid unit.")
            return

        basis = None
        if self._basis_group.isChecked():
            basis = {
                "window": (self._win_min.value(), self._win_max.value()),
                "n_basis": self._n_basis.value(),
                "spacing": self._spacing_combo.currentText(),
            }

        params = build_encoder_params(
            node=node, unit=unit, features=features,
            family=self._family_combo.currentText(),
            mode=self._mode_combo.currentText(),
            basis=basis,
            n_splits=self._cv_spin.value(),
        )
        self.result = (pose_path, spike_path, params)
        self.accept()


class DecoderDialog(QDialog):
    """Configure and launch a GLM decoder (population of units -> a movement feature).

    Asks whether a set of units can reconstruct a kinematic target (speed,
    position, velocity, ...) for a node. On accept, ``self.result`` holds
    ``(pose_path, spike_path, params)``.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit GLM Decoder")
        self.setMinimumWidth(520)
        self.session = session
        self.result = None
        self._stores = _discover_binned_stores(
            session.dirs.get("pose"), session.dirs.get("spikes"))
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Binned data source ──
        data_group = QGroupBox("Binned data")
        data_form = QFormLayout(data_group)
        self._bin_combo = QComboBox()
        if self._stores:
            self._bin_combo.addItems([f"{ms} ms" for ms in self._stores])
        else:
            self._bin_combo.addItem("(none — run Bin first)")
            self._bin_combo.setEnabled(False)
        self._bin_combo.currentIndexChanged.connect(self._on_bin_changed)
        data_form.addRow("Bin size:", self._bin_combo)
        layout.addWidget(data_group)

        # ── Target ──
        target_group = QGroupBox("Decode target")
        target_form = QFormLayout(target_group)
        self._node_combo = QComboBox()
        target_form.addRow("Reference node:", self._node_combo)
        self._target_combo = QComboBox()
        self._target_combo.addItems(_ENCODER_FEATURES)   # speed / position_* / velocity_* / ...
        target_form.addRow("Movement feature:", self._target_combo)
        self._all_events = QCheckBox("Use all movement events (not just this node's)")
        self._all_events.setChecked(True)
        self._all_events.setToolTip("Decode this node's kinematics from every bout, "
                                    "not only the ones it initiated — more training data.")
        target_form.addRow("", self._all_events)
        layout.addWidget(target_group)

        # ── Population (units) ──
        units_group = QGroupBox("Population (units)")
        units_layout = QVBoxLayout(units_group)
        self._units_list = QListWidget()
        self._units_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._units_list.setMaximumHeight(140)
        hint = QLabel("Select the units to decode from (Ctrl/Shift for multiple; none = all).")
        hint.setObjectName("subheading")
        units_layout.addWidget(self._units_list)
        units_layout.addWidget(hint)
        layout.addWidget(units_group)

        # ── Spike history (lags) — the key to decoding well ──
        self._lag_group = QGroupBox("Spike history (population lags)")
        self._lag_group.setCheckable(True)
        self._lag_group.setChecked(True)
        lag_form = QFormLayout(self._lag_group)
        self._lag_min = QDoubleSpinBox()
        self._lag_min.setRange(-1.0, 1.0); self._lag_min.setSingleStep(0.02)
        self._lag_min.setDecimals(3); self._lag_min.setValue(-0.15)
        self._lag_max = QDoubleSpinBox()
        self._lag_max.setRange(-1.0, 1.0); self._lag_max.setSingleStep(0.02)
        self._lag_max.setDecimals(3); self._lag_max.setValue(0.15)
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("from")); lrow.addWidget(self._lag_min)
        lrow.addWidget(QLabel("to")); lrow.addWidget(self._lag_max); lrow.addWidget(QLabel("s"))
        lw = QWidget(); lw.setLayout(lrow)
        lag_form.addRow("Window (rel. kinematic):", lw)
        self._lag_nbasis = QSpinBox()
        self._lag_nbasis.setRange(1, 15); self._lag_nbasis.setValue(5)
        lag_form.addRow("# basis functions:", self._lag_nbasis)
        lhint = QLabel("Spans spikes before & after each moment; off = same-bin only (usually decodes poorly).")
        lhint.setObjectName("subheading"); lhint.setWordWrap(True)
        lag_form.addRow(lhint)
        layout.addWidget(self._lag_group)

        # ── Model ──
        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)
        self._family_combo = QComboBox()
        self._family_combo.addItems(["Gaussian", "Poisson"])
        model_form.addRow("Family:", self._family_combo)
        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.0, 1.0); self._smooth_spin.setDecimals(3)
        self._smooth_spin.setSingleStep(0.01); self._smooth_spin.setValue(0.05)
        self._smooth_spin.setToolTip("Gaussian σ (s) smoothing spikes into firing rates "
                                     "before lagging. 0 = raw counts. Denoising usually helps.")
        model_form.addRow("Rate smoothing σ (s):", self._smooth_spin)
        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0, 10000.0); self._alpha_spin.setDecimals(2)
        self._alpha_spin.setSingleStep(1.0); self._alpha_spin.setValue(1.0)
        self._alpha_spin.setToolTip("L2 (ridge) penalty. 0 = ordinary least squares. "
                                    "Higher = more shrinkage; helps with many units/lags.")
        model_form.addRow("Ridge α (0 = none):", self._alpha_spin)
        self._cv_spin = QSpinBox()
        self._cv_spin.setRange(0, 20)
        self._cv_spin.setValue(5)
        self._cv_spin.setToolTip("Event-grouped cross-validation folds. 0 = in-sample fit.")
        model_form.addRow("CV folds (0 = in-sample):", self._cv_spin)
        self._shuffle_spin = QSpinBox()
        self._shuffle_spin.setRange(0, 1000); self._shuffle_spin.setValue(0)
        self._shuffle_spin.setToolTip("Trial-shuffle permutation null → a p-value for the "
                                      "CV R². 0 = off; 100–200 typical. Slower (re-runs CV).")
        model_form.addRow("Shuffle null (0 = off):", self._shuffle_spin)
        layout.addWidget(model_group)

        layout.addWidget(HDivider())
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Fit Decoder")
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if self._stores:
            self._on_bin_changed(0)

    def _current_paths(self):
        if not self._stores:
            return None, None
        ms = list(self._stores.keys())[max(0, self._bin_combo.currentIndex())]
        return self._stores[ms]

    def _on_bin_changed(self, _idx):
        pose_path, spike_path = self._current_paths()
        if pose_path is None:
            return
        from neurokinematics.io import load_zarr
        self._node_combo.clear()
        self._units_list.clear()
        try:
            pose_ds = load_zarr(pose_path, method="xarray")
            self._node_combo.addItems([str(n) for n in pose_ds.node.values])
        except Exception:
            self._node_combo.setEditable(True)
        try:
            spike_ds = load_zarr(spike_path, method="xarray")
            for u in spike_ds.unit.values:
                self._units_list.addItem(str(int(u)))
        except Exception:
            pass

    def _selected_units(self):
        sel = [int(i.text()) for i in self._units_list.selectedItems()]
        if sel:
            return sel
        # none selected -> use the whole population
        return [int(self._units_list.item(i).text()) for i in range(self._units_list.count())]

    def _confirm(self):
        from neurokinematics.models.glm import build_decoder_params

        pose_path, spike_path = self._current_paths()
        if pose_path is None:
            QMessageBox.warning(self, "No binned data",
                                "No matching pose/spike zarr stores were found.\n"
                                "Run 'Bin' to generate them first.")
            return
        units = self._selected_units()
        if not units:
            QMessageBox.warning(self, "No units", "No units available to decode from.")
            return

        if self._lag_group.isChecked() and self._lag_max.value() < self._lag_min.value():
            QMessageBox.warning(self, "Invalid window",
                                "Spike-history window end must be ≥ start.")
            return

        lag = None
        if self._lag_group.isChecked():
            lag = {
                "window": (self._lag_min.value(), self._lag_max.value()),
                "n_basis": self._lag_nbasis.value(),
            }

        params = build_decoder_params(
            node=self._node_combo.currentText(),
            units=units,
            target=self._target_combo.currentText(),
            family=self._family_combo.currentText(),
            n_splits=self._cv_spin.value(),
            lag=lag,
            alpha=self._alpha_spin.value(),
            smoothing_s=self._smooth_spin.value(),
            all_events=self._all_events.isChecked(),
            n_shuffle=self._shuffle_spin.value(),
        )
        self.result = (pose_path, spike_path, params)
        self.accept()


# ── Spike bad-channel review ──────────────────────────────────────────────────

class SpikeQCDialog(QDialog):
    """Review SpikeInterface bad-channel detection before sorting.

    Shows a bandpassed snippet of the flagged channels and a checklist (checked
    = remove). On accept, ``self.selected_bad`` holds the channel ids to drop;
    if no channels are flagged, the dialog still lets the user proceed straight
    to sorting. *detection* is the dict returned by
    ``session.preprocess_spikes()``.
    """

    def __init__(self, detection: dict, session_label: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spike Sorting — Bad Channel Review")
        self.setMinimumSize(720, 560)
        self.detection   = detection or {}
        self.selected_bad = []
        self._checks      = {}
        self._build_ui(session_label)

    def _build_ui(self, session_label):
        from PySide6.QtWidgets import QScrollArea
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        bad      = [str(b) for b in self.detection.get("bad_ids", [])]
        channels = [str(c) for c in self.detection.get("channel_ids", [])]
        labels   = list(self.detection.get("labels", []))
        label_of = dict(zip(channels, labels))

        title = QLabel(f"<b>{session_label or 'Session'}</b> — "
                       f"{len(bad)}/{len(channels)} channel(s) flagged bad")
        title.setObjectName("heading")
        layout.addWidget(title)

        # tally of labels (dead/noise/out)
        counts = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        if counts:
            summary = "  ·  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            sub = QLabel(summary)
            sub.setObjectName("subheading")
            layout.addWidget(sub)

        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        # ── snippet plot (left) ──
        fig = Figure(figsize=(5, 4), tight_layout=True)
        canvas = FigureCanvas(fig)
        self._plot_snippet(fig, bad, channels)
        body.addWidget(canvas, stretch=3)

        # ── checklist (right) ──
        right = QVBoxLayout()
        right.addWidget(QLabel("Remove from sorting:"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        vbox   = QVBoxLayout(holder)
        vbox.setSpacing(2)
        if bad:
            for cid in bad:
                lab = label_of.get(cid, "")
                cb  = QCheckBox(f"ch {cid}" + (f"  ({lab})" if lab else ""))
                cb.setChecked(True)          # default: remove flagged channels
                self._checks[cid] = cb
                vbox.addWidget(cb)
        else:
            vbox.addWidget(QLabel("No bad channels detected."))
        vbox.addStretch()
        scroll.setWidget(holder)
        right.addWidget(scroll, stretch=1)

        if bad:
            btn_row = QHBoxLayout()
            all_btn  = QPushButton("All")
            none_btn = QPushButton("None")
            all_btn.clicked.connect(lambda: self._set_all(True))
            none_btn.clicked.connect(lambda: self._set_all(False))
            btn_row.addWidget(all_btn)
            btn_row.addWidget(none_btn)
            btn_row.addStretch()
            right.addLayout(btn_row)

        body.addLayout(right, stretch=1)

        layout.addWidget(HDivider())
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Run Sorting")
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _plot_snippet(self, fig, bad, channels):
        import numpy as np
        snippet = self.detection.get("snippet", {}) or {}
        traces  = snippet.get("traces")
        t       = snippet.get("t")
        ax = fig.add_subplot(111)
        if traces is None or len(channels) == 0:
            ax.text(0.5, 0.5, "No snippet available", ha="center", va="center")
            ax.axis("off")
            return

        traces = np.asarray(traces)
        t      = np.asarray(t) if t is not None else np.arange(traces.shape[0])
        # decimate for responsiveness
        if traces.shape[0] > 5000:
            step   = traces.shape[0] // 5000
            traces = traces[::step]
            t      = t[::step]

        # plot bad channels (red); if none, show first few good channels (grey)
        bad_set = set(bad)
        show = [c for c in channels if c in bad_set] or channels[:6]
        # vertical offset so stacked traces are legible
        spread = np.nanmedian(np.nanstd(traces, axis=0)) or 1.0
        offset = 6.0 * spread
        for i, cid in enumerate(show):
            col = channels.index(cid)
            colour = "#e05050" if cid in bad_set else "#8e8aaa"
            ax.plot(t, traces[:, col] + i * offset, color=colour, linewidth=0.6)
            ax.text(t[0], i * offset, f"ch {cid}", color=colour,
                    fontsize=7, va="bottom")
        ax.set_xlabel("time (s)")
        ax.set_yticks([])
        ax.set_title("Bandpassed snippet" +
                     ("" if bad else "  (no bad channels — showing samples)"),
                     fontsize=9)

    def _set_all(self, state: bool):
        for cb in self._checks.values():
            cb.setChecked(state)

    def _confirm(self):
        self.selected_bad = [cid for cid, cb in self._checks.items()
                             if cb.isChecked()]
        self.accept()
