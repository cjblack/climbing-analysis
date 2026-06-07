import sys
import yaml
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QListWidget, QCheckBox, QFileDialog, QDialog, QDialogButtonBox,
    QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt


class PathField(QWidget):
    """A line edit with a browse button for folder or file selection."""
    def __init__(self, mode='folder', parent=None):
        super().__init__(parent)
        self.mode = mode
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Browse or type path...")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(70)
        self.browse_btn.clicked.connect(self._browse)

        layout.addWidget(self.edit)
        layout.addWidget(self.browse_btn)

    def _browse(self):
        if self.mode == 'folder':
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Config File", filter="YAML Files (*.yaml *.yml)"
            )
        if path:
            self.edit.setText(path)

    def text(self):
        return self.edit.text().strip()

    def setText(self, text):
        self.edit.setText(text)


class SessionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Session")
        self.setMinimumWidth(500)
        self.result_data = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.session_id = QLineEdit()
        self.session_id.setPlaceholderText("e.g. session_001")

        self.session_config = PathField(mode='file')
        self.ephys_path = PathField(mode='folder')
        self.pose_path = PathField(mode='folder')

        form.addRow("Session ID:", self.session_id)
        form.addRow("Session Config:", self.session_config)
        form.addRow("Ephys Data Path:", self.ephys_path)
        form.addRow("Pose Data Path:", self.pose_path)

        layout.addLayout(form)

        # buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _confirm(self):
        session_id = self.session_id.text().strip()
        if not session_id:
            QMessageBox.warning(self, "Missing Field", "Session ID is required.")
            return

        self.result_data = {
            'session_id': session_id,
            'session_config': self.session_config.text() or None,
            'ephys_data_path': self.ephys_path.text() or None,
            'pose_data_path': self.pose_path.text() or None,
        }
        self.accept()


class SubjectSpecBuilder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("neurokinematics — Subject Spec Builder")
        self.setMinimumWidth(560)
        self.sessions = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # --- Subject info ---
        subj_group = QGroupBox("Subject")
        subj_form = QFormLayout(subj_group)
        subj_form.setLabelAlignment(Qt.AlignRight)

        self.subject_id = QLineEdit()
        self.subject_id.setPlaceholderText("e.g. SK01")

        self.output_root = PathField(mode='folder')

        subj_form.addRow("Subject ID:", self.subject_id)
        subj_form.addRow("Output Root:", self.output_root)
        main_layout.addWidget(subj_group)

        # --- Process flags ---
        proc_group = QGroupBox("Process")
        proc_layout = QHBoxLayout(proc_group)

        self.proc_pose = QCheckBox("Pose")
        self.proc_spike = QCheckBox("Spikes")
        self.proc_lfp = QCheckBox("LFP")

        for cb in [self.proc_pose, self.proc_spike, self.proc_lfp]:
            cb.setChecked(True)
            proc_layout.addWidget(cb)

        proc_layout.addStretch()
        main_layout.addWidget(proc_group)

        # --- Sessions ---
        sess_group = QGroupBox("Sessions")
        sess_layout = QVBoxLayout(sess_group)

        self.session_list = QListWidget()
        self.session_list.setMinimumHeight(120)
        sess_layout.addWidget(self.session_list)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Session")
        remove_btn = QPushButton("− Remove Selected")
        add_btn.clicked.connect(self._add_session)
        remove_btn.clicked.connect(self._remove_session)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        sess_layout.addLayout(btn_layout)

        main_layout.addWidget(sess_group)

        # --- Save ---
        save_btn = QPushButton("Save Spec")
        save_btn.setFixedHeight(36)
        save_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        save_btn.clicked.connect(self._save_spec)
        main_layout.addWidget(save_btn)

    def _add_session(self):
        dialog = SessionDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.result_data:
            self.sessions.append(dialog.result_data)
            self.session_list.addItem(dialog.result_data['session_id'])

    def _remove_session(self):
        selected = self.session_list.currentRow()
        if selected >= 0:
            self.session_list.takeItem(selected)
            self.sessions.pop(selected)

    def _save_spec(self):
        subject_id = self.subject_id.text().strip()
        output_root = self.output_root.text()

        if not subject_id:
            QMessageBox.warning(self, "Missing Field", "Subject ID is required.")
            return
        if not output_root:
            QMessageBox.warning(self, "Missing Field", "Output Root is required.")
            return

        spec = {
            'subject_id': subject_id,
            'output_root': output_root,
            'process': {
                'pose': self.proc_pose.isChecked(),
                'spike': self.proc_spike.isChecked(),
                'lfp': self.proc_lfp.isChecked(),
            },
            'sessions': self.sessions if self.sessions else None
        }

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Spec", f"{subject_id}.yaml",
            "YAML Files (*.yaml *.yml)"
        )

        if save_path:
            with open(save_path, 'w') as f:
                yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
            QMessageBox.information(self, "Saved", f"Spec saved to:\n{save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SubjectSpecBuilder()
    window.show()
    sys.exit(app.exec())