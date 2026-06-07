"""GUI widget that renders a neurokinematics QC report.

Pure presentation: it takes one or more :class:`~neurokinematics.qc.QCReport`
objects (already computed) and shows them as a colour-coded tree with a summary
header and a JSON export button.
"""

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QHeaderView, QMessageBox
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtCore import Qt

from neurokinematics.qc import QCStatus
from neurokinematics.gui.style import SUCCESS, WARNING, ERROR, TEXT_DIM, TEXT


_STATUS_COLOUR = {
    QCStatus.PASS: SUCCESS,
    QCStatus.WARN: WARNING,
    QCStatus.FAIL: ERROR,
    QCStatus.NA:   TEXT_DIM,
}
_STATUS_LABEL = {
    QCStatus.PASS: "PASS",
    QCStatus.WARN: "WARN",
    QCStatus.FAIL: "FAIL",
    QCStatus.NA:   "n/a",
}


class QCReportWidget(QWidget):
    def __init__(self, reports, title: str = "Data QC", sources=None, parent=None):
        super().__init__(parent)
        # accept a single report or a list
        if not isinstance(reports, (list, tuple)):
            reports = [reports]
        self._reports = list(reports)
        # optional source objects (group/subject/session), parallel to reports,
        # used to export each report into the folder of the level QC was run at
        if sources is None:
            sources = []
        elif not isinstance(sources, (list, tuple)):
            sources = [sources]
        self._sources = list(sources)
        self._build(title)

    # ── UI ──
    def _build(self, title: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # summary header
        tally = {s.value: 0 for s in QCStatus}
        for rep in self._reports:
            for k, v in rep.counts().items():
                tally[k] += v
        worst = self._overall_status()

        header = QHBoxLayout()
        badge = QLabel(_STATUS_LABEL[worst])
        badge.setStyleSheet(
            f"background-color: {_STATUS_COLOUR[worst]}; color: white; "
            f"font-weight: bold; border-radius: 4px; padding: 2px 10px;"
        )
        summary = QLabel(
            f"  {tally['pass']} pass · {tally['warn']} warn · "
            f"{tally['fail']} fail · {tally['na']} n/a"
        )
        summary.setObjectName("subheading")
        header.addWidget(badge)
        header.addWidget(summary)
        header.addStretch()

        export_btn = QPushButton("Export JSON")
        export_btn.setObjectName("secondary")
        export_btn.clicked.connect(self._export)
        header.addWidget(export_btn)
        layout.addLayout(header)

        # tree
        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Check / Item", "Status", "Details"])
        self._tree.setAlternatingRowColors(False)
        self._tree.setRootIsDecorated(True)
        hdr = self._tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self._tree)

        for rep in self._reports:
            self._add_report(self._tree, rep)
        self._tree.expandAll()

    def _overall_status(self) -> QCStatus:
        statuses = [r.status for r in self._reports]
        non_na = [s for s in statuses if s is not QCStatus.NA]
        if not non_na:
            return QCStatus.NA
        return max(non_na, key=lambda s: s.severity)

    def _add_report(self, parent, report):
        """Recursively add a QCReport (group/subject/session) to the tree."""
        item = (QTreeWidgetItem(parent) if isinstance(parent, QTreeWidgetItem)
                else QTreeWidgetItem(self._tree))
        item.setText(0, f"{report.level}: {report.target}")
        self._set_status_cell(item, report.status)
        c = report.counts()
        item.setText(2, f"{c['pass']} pass · {c['warn']} warn · "
                        f"{c['fail']} fail · {c['na']} n/a")
        # bold the report node label
        f = item.font(0); f.setBold(True); item.setFont(0, f)
        item.setForeground(0, QBrush(QColor(TEXT)))

        for res in report.results:
            leaf = QTreeWidgetItem(item)
            leaf.setText(0, res.name)
            self._set_status_cell(leaf, res.status)
            leaf.setText(2, res.message)

        for child in report.children:
            self._add_report(item, child)

    @staticmethod
    def _set_status_cell(item: QTreeWidgetItem, status: QCStatus):
        item.setText(1, _STATUS_LABEL[status])
        item.setForeground(1, QBrush(QColor(_STATUS_COLOUR[status])))
        f = item.font(1); f.setBold(True); item.setFont(1, f)

    @staticmethod
    def _target_dir(obj):
        """Folder for the level QC was run at: session, then subject, then group."""
        for attr in ("session_path", "subject_path", "group_path"):
            p = getattr(obj, attr, None)
            if p:
                p = Path(p)
                if p.exists():
                    return p
        return None

    def _export(self):
        # When we know the source objects, write each report into a 'qc/'
        # subfolder of its own folder (session/subject/group) — the level QC was
        # performed at — with a timestamped filename so history accumulates.
        if self._sources:
            from datetime import datetime
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved, failed = [], []
            for rep, src in zip(self._reports, self._sources):
                folder = self._target_dir(src)
                if folder is None:
                    failed.append(f"{rep.level} '{rep.target}' — no folder on disk")
                    continue
                try:
                    qc_dir = folder / "qc"
                    qc_dir.mkdir(parents=True, exist_ok=True)
                    out = qc_dir / f"qc_report_{stamp}.json"
                    out.write_text(json.dumps(rep.to_dict(), indent=2))
                    saved.append(str(out))
                except Exception as e:
                    failed.append(f"{rep.level} '{rep.target}' — {e}")

            lines = []
            if saved:
                lines.append("Saved QC report to:\n" + "\n".join(saved))
            if failed:
                lines.append("Could not save:\n" + "\n".join(failed))
            QMessageBox.information(self, "Export QC Report",
                                    "\n\n".join(lines) or "Nothing to export.")
            return

        # Fallback (no source objects known): ask the user where to save.
        path, _ = QFileDialog.getSaveFileName(
            self, "Export QC Report", "qc_report.json", "JSON Files (*.json)"
        )
        if not path:
            return
        payload = [r.to_dict() for r in self._reports]
        data = payload[0] if len(payload) == 1 else payload
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
