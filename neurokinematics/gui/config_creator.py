"""
Config creator dialog for the neurokinematics GUI.

Lets the user create or edit session, pose, spike sorting, LFP,
and multimodal config YAML files, starting from built-in templates.
"""

import yaml
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QPushButton, QTextEdit,
    QDialogButtonBox, QFileDialog, QMessageBox,
    QGroupBox, QTabWidget, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


# ── Templates (mirrors the actual demo configs) ───────────────────────────────

TEMPLATES = {
    'session': {
        'session': {
            'output_root': 'set_location_for_data',
            'behaviour': 'my_experiment',
            'ephys': {
                'acquisition': 'openephys',
                'lfp':    {'node_idx': 0, 'rec_idx': 0},
                'spikes': {'node_idx': 1, 'rec_idx': 0},
            }
        },
        'configs': {
            'spikes':     'demo_spike_sorting_cfg.yaml',
            'lfp':        'demo_lfp_cfg.yaml',
            'pose':       'demo_pose_cfg.yaml',
            'multi_modal':'demo_camera_alignment_cfg.yaml',
            'models':     'demo_model_cfg.yaml',
        },
        'pipeline': {
            'run_pose':   True,
            'run_spikes': True,
            'run_lfp':    True,
        }
    },

    'pose': {
        'pose_format': {
            'tracker':   'sleap',
            'file_format': 'h5',
            'frame_rate':  200.0,
        },
        'pose_preprocessing': {
            'fill_missing': True,
            'confidence':   {'enabled': False, 'thresh': 0.7},
            'velocity':     {'enabled': False, 'thresh': 20},
        },
        'post_processing': {
            'storage_format': 'csv',
        },
        'movement_detection': {
            'enabled': True,
            'sort_cols':  ['Trial', 'Date', 'frame_id'],
            'group_cols': ['Trial', 'Date'],
            'node_list':  ['r_forepaw', 'l_forepaw', 'r_hindpaw', 'l_hindpaw'],
        }
    },

    'spikes': {
        'rec_type': 'openephys',
        'sorter':   'kilosort4',
        'to_compute': {
            'random_spikes': {},
            'waveforms':     {'ms_before': 1.0, 'ms_after': 2.0},
            'templates':     {},
            'noise_levels':  {},
            'spike_locations': {},
        },
        'quality_metrics': [
            'firing_rate', 'snr', 'amplitude_cutoff',
            'drift', 'isi_violation', 'd_prime',
        ],
        'probe_manufacturer': 'cambridge_neurotech',
        'probe_id':           'ASSY-236-H5',
        'group_mode':         'auto',
        'channel_map':        'h5_open_ephys_acquisition_channel_map.npy',
        'stream_name':        'Record Node 109#Acquisition_Board-100.acquisition_board-B',
        'sample_rate':        30000.0,
    },

    'lfp': {
        'dtype':         'float32',
        'chunking': {
            'chunk_duration_s': 10.0,
            'pad_duration_s':   1.0,
        },
        'filters': {
            'notch':     50.0,
            'bandpass':  [0.1, 100.0],
            'quality':   30.0,
        },
        'downsample_rate': 1000.0,
        'storage_format':  'zarr',
    },

    'multimodal': {
        'camera': {
            'fps':       200.0,
            'bandwidth': 10.0,
        },
        'strobe_detection': {
            'threshold': 0.5,
            'min_duration_s': 0.001,
        }
    },
}

CONFIG_SAVE_DIRS = {
    'session':   'session_cfg',
    'pose':      'pose_cfg',
    'spikes':    'spk_sorting_cfg',
    'lfp':       'lfp_cfg',
    'multimodal':'multimodal_cfg',
}

CONFIG_LABELS = {
    'session':    'Session Config',
    'pose':       'Pose Config',
    'spikes':     'Spike Sorting Config',
    'lfp':        'LFP Config',
    'multimodal': 'Multimodal Config',
}


class ConfigCreatorDialog(QDialog):
    """
    Dialog for creating and editing neurokinematics config YAML files.

    - Select config type from a tab or dropdown
    - Edit YAML in a syntax-highlighted text editor
    - Load an existing config to edit it
    - Save to the package configs folder or a custom location
    """

    def __init__(self, config_type: str = 'session', parent=None):
        super().__init__(parent)
        self.setWindowTitle("Config Creator")
        self.setMinimumSize(660, 580)
        self._config_type = config_type
        self._build_ui()
        self._load_template(config_type)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Type selector ──
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Config type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(list(CONFIG_LABELS.values()))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo)
        type_row.addStretch()

        load_btn = QPushButton("Load existing...")
        load_btn.setObjectName("secondary")
        load_btn.clicked.connect(self._load_existing)
        type_row.addWidget(load_btn)

        reset_btn = QPushButton("Reset to template")
        reset_btn.setObjectName("secondary")
        reset_btn.clicked.connect(lambda: self._load_template(self._config_type))
        type_row.addWidget(reset_btn)

        layout.addLayout(type_row)

        # ── YAML editor ──
        editor_group = QGroupBox("YAML Editor")
        editor_layout = QVBoxLayout(editor_group)

        self._editor = QTextEdit()
        font = QFont("Consolas", 11)
        self._editor.setFont(font)
        self._editor.setAcceptRichText(False)
        editor_layout.addWidget(self._editor)

        layout.addWidget(editor_group, stretch=1)

        # ── Filename ──
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Filename:"))
        self._name_edit = __import__('PySide6.QtWidgets', fromlist=['QLineEdit']).QLineEdit()
        self._name_edit.setPlaceholderText("my_config.yaml")
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        # ── Buttons ──
        btn_layout = QHBoxLayout()

        save_pkg_btn = QPushButton("Save to package configs")
        save_pkg_btn.clicked.connect(self._save_to_package)
        btn_layout.addWidget(save_pkg_btn)

        save_custom_btn = QPushButton("Save to custom location...")
        save_custom_btn.setObjectName("secondary")
        save_custom_btn.clicked.connect(self._save_custom)
        btn_layout.addWidget(save_custom_btn)

        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setObjectName("secondary")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        # initialise type combo to match constructor arg
        keys = list(CONFIG_LABELS.keys())
        if self._config_type in keys:
            self._type_combo.setCurrentIndex(keys.index(self._config_type))

    def _on_type_changed(self, idx: int):
        key = list(CONFIG_LABELS.keys())[idx]
        self._config_type = key
        self._load_template(key)

    def _load_template(self, config_type: str):
        template = TEMPLATES.get(config_type, {})
        self._editor.setPlainText(
            yaml.dump(template, default_flow_style=False, sort_keys=False)
        )
        self._name_edit.setPlaceholderText(f"my_{config_type}_cfg.yaml")

    def _load_existing(self):
        from neurokinematics.io import CFG_PATHS
        start = str(CFG_PATHS.get(
            {'spikes': 'spksorting', 'multimodal': 'multimodal'}.get(
                self._config_type, self._config_type
            ), ''
        ))
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", start, "YAML Files (*.yaml *.yml)"
        )
        if path:
            with open(path) as f:
                content = f.read()
            self._editor.setPlainText(content)
            self._name_edit.setText(Path(path).name)

    def _get_yaml(self):
        try:
            text = self._editor.toPlainText()
            yaml.safe_load(text)   # validate
            return text
        except yaml.YAMLError as e:
            QMessageBox.critical(self, "YAML Error", f"Invalid YAML:\n{e}")
            return None

    def _save_to_package(self):
        text = self._get_yaml()
        if text is None:
            return

        from neurokinematics.io import CFG_PATHS
        cfg_key = {'spikes': 'spksorting', 'multimodal': 'multimodal'}.get(
            self._config_type, self._config_type
        )
        save_dir = CFG_PATHS.get(cfg_key, None)
        if save_dir is None:
            QMessageBox.warning(self, "Unknown path", "Cannot resolve package config directory.")
            return

        filename = self._name_edit.text().strip() or f"my_{self._config_type}_cfg.yaml"
        if not filename.endswith(('.yaml', '.yml')):
            filename += '.yaml'

        dest = Path(save_dir) / filename
        if dest.exists():
            ans = QMessageBox.question(
                self, "Overwrite?", f"{filename} already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if ans != QMessageBox.Yes:
                return

        with open(dest, 'w') as f:
            f.write(text)
        QMessageBox.information(self, "Saved", f"Config saved to:\n{dest}")

    def _save_custom(self):
        text = self._get_yaml()
        if text is None:
            return

        filename = self._name_edit.text().strip() or f"my_{self._config_type}_cfg.yaml"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", filename, "YAML Files (*.yaml *.yml)"
        )
        if path:
            with open(path, 'w') as f:
                f.write(text)
            QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
