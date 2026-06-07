"""
Shared widgets for the neurokinematics GUI.
"""

import sys
import io

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton,
    QTextEdit, QLabel, QComboBox, QSizePolicy, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QTextCursor
from neurokinematics.gui.style import TEXT_DIM, SUCCESS, WARNING, ERROR, BG_MID, PRIMARY


# ── Path field ────────────────────────────────────────────────────────────────

class PathField(QWidget):
    """Line edit + browse button for selecting a folder or file."""

    def __init__(self, mode: str = 'folder', placeholder: str = "",
                 start_dir: str = "", parent=None):
        super().__init__(parent)
        self.mode      = mode
        self.start_dir = start_dir

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder or ("Select folder..." if mode == 'folder' else "Select file..."))

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setObjectName("secondary")
        self.browse_btn.setFixedWidth(70)
        self.browse_btn.clicked.connect(self._browse)

        layout.addWidget(self.edit)
        layout.addWidget(self.browse_btn)

    def _browse(self):
        from PySide6.QtWidgets import QFileDialog
        if self.mode == 'folder':
            path = QFileDialog.getExistingDirectory(
                self, "Select Folder", self.start_dir
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select File", self.start_dir,
                "YAML Files (*.yaml *.yml)"
            )
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, text: str):
        self.edit.setText(text)

    def clear(self):
        self.edit.clear()


# ── Divider ───────────────────────────────────────────────────────────────────

class HDivider(QFrame):
    """A simple horizontal divider line."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


# ── Stream redirector ─────────────────────────────────────────────────────────

class StreamRedirector(QObject):
    """
    Captures writes to stdout/stderr and emits lines as Qt signals.
    Strips carriage returns — tqdm line-detection is handled by LogWidget
    using content matching rather than \r parsing, which is more reliable
    across platforms and tqdm versions.
    """
    line_written = Signal(str)   # one complete line (stripped of \r\n)

    def __init__(self, original_stream=None):
        super().__init__()
        self._original = original_stream
        self._buffer   = ""

    def write(self, text: str):
        if self._original:
            self._original.write(text)
        # strip \r so we only split on \n
        self._buffer += text.replace('\r', '')
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            line = line.strip()
            if line:
                self.line_written.emit(line)

    def flush(self):
        if self._buffer.strip():
            self.line_written.emit(self._buffer.strip())
            self._buffer = ""
        if self._original:
            try:
                self._original.flush()
            except (OSError, IOError):
                # Windows console handles can become invalid (errno 22)
                # when running inside a GUI — safe to ignore
                pass

    def fileno(self):
        if self._original:
            try:
                return self._original.fileno()
            except Exception:
                pass
        return -1

    def isatty(self):
        return False


# ── Python logging → LogWidget bridge ────────────────────────────────────────

class _QtLogHandler(QObject):
    """
    Forwards Python log records to the LogWidget via Qt signals.
    This captures PyMC's INFO/WARNING messages which go through
    logging rather than stdout/stderr.
    """
    import logging as _logging

    def __init__(self, append_signal, format_fn):
        super().__init__()
        import logging
        self._append_signal = append_signal
        self._format_fn     = format_fn
        # Plain logging.Handler (not StreamHandler): its flush() is a safe no-op,
        # so interpreter shutdown won't choke on a missing .stream attribute.
        self._handler       = logging.Handler()
        self._handler.emit  = self._on_record

    def _on_record(self, record):
        import logging
        level_map = {
            logging.DEBUG:    'info',
            logging.INFO:     'info',
            logging.WARNING:  'warning',
            logging.ERROR:    'error',
            logging.CRITICAL: 'error',
        }
        level = level_map.get(record.levelno, 'info')
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record)
        # filter out very noisy pytensor compilation messages
        if any(skip in msg for skip in ['compiling', 'reusing', 'Compiling', 'WARNING (pytensor']):
            return
        html = self._format_fn(level, msg)
        self._append_signal.emit(html)

    def __getattr__(self, name):
        """Delegate logging.Handler interface to inner handler."""
        return getattr(self._handler, name)


# ── Log widget ────────────────────────────────────────────────────────────────

class LogWidget(QTextEdit):
    """
    Read-only log output area with coloured message levels.
    Can optionally capture stdout/stderr (including tqdm) via install_redirectors().
    """

    _append_signal  = Signal(str)   # thread-safe append new line
    _replace_signal = Signal(str)   # thread-safe replace last line (tqdm)

    # tqdm progress lines contain these markers
    _TQDM_MARKERS = ('%|', 'it/s]', 's/it]', '?/s]', '/s]')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._append_signal.connect(self._do_append)
        self._replace_signal.connect(self._do_replace_last)

        self._stdout_redirector = None
        self._stderr_redirector = None
        self._last_was_tqdm     = False   # track whether previous line was a tqdm bar
        self._last_tqdm_key     = None    # identity of that bar (its description)

    def install_redirectors(self):
        """
        Redirect stdout, stderr, and Python logging (including PyMC/rich output)
        into this widget.
        """
        self._stdout_redirector = StreamRedirector(sys.stdout)
        self._stderr_redirector = StreamRedirector(sys.stderr)

        self._stdout_redirector.line_written.connect(
            lambda t: self._append_signal.emit(self._format('info', t))
        )
        self._stderr_redirector.line_written.connect(self._on_stderr_line)

        sys.stdout = self._stdout_redirector
        sys.stderr = self._stderr_redirector

        # capture Python logging (PyMC uses this for INFO messages)
        self._log_handler = _QtLogHandler(self._append_signal, self._format)
        import logging
        logging.getLogger().addHandler(self._log_handler)
        logging.getLogger('pymc').setLevel(logging.INFO)
        logging.getLogger('pytensor').setLevel(logging.WARNING)

        # Note: PyMC rich progress bars are disabled via progressbar=False in
        # the default params to avoid Console/cursor conflicts with the GUI.
        # PyMC's logging output (NUTS init, divergences, ESS, R-hat) still
        # flows through the Python logging handler installed above.

    def _is_tqdm(self, text: str) -> bool:
        return any(m in text for m in self._TQDM_MARKERS)

    @staticmethod
    def _tqdm_key(text: str) -> str:
        """Identity of a progress bar = its description (text before the %).

        tqdm re-emits the same bar many times as it advances; every emission
        shares this prefix. A *different* bar (e.g. the next subject/session)
        has a different description, so we can tell when one bar finished and a
        new one began — and append a fresh line instead of overwriting.
        """
        head = text.split('%|', 1)[0]
        return head.rstrip('0123456789 \t:%')

    def _on_stderr_line(self, text: str):
        html = self._format('tqdm', text)
        if self._is_tqdm(text):
            key = self._tqdm_key(text)
            if self._last_was_tqdm and key == self._last_tqdm_key:
                # same bar advancing — update it in place on its single line
                self._replace_signal.emit(html)
            else:
                # first bar, or a new bar — append so prior output is preserved
                self._append_signal.emit(html)
            # _do_append resets the flag, so set state *after* emitting
            self._last_was_tqdm = True
            self._last_tqdm_key = key
        else:
            self._append_signal.emit(html)
            self._last_was_tqdm = False
            self._last_tqdm_key = None

    def restore_streams(self):
        """Restore original stdout/stderr and logging — call on app exit."""
        if self._stdout_redirector and self._stdout_redirector._original:
            sys.stdout = self._stdout_redirector._original
        if self._stderr_redirector and self._stderr_redirector._original:
            sys.stderr = self._stderr_redirector._original
        if hasattr(self, '_log_handler'):
            import logging
            logging.getLogger().removeHandler(self._log_handler)
            # close() deregisters the handler so logging.shutdown() won't touch it
            try:
                self._log_handler.close()
            except Exception:
                pass

    def log(self, message: str, level: str = 'info'):
        html = self._format(level, message)
        self._append_signal.emit(html)

    def _format(self, level: str, message: str) -> str:
        colours = {
            'info':    TEXT_DIM,
            'success': SUCCESS,
            'warning': WARNING,
            'error':   ERROR,
            'tqdm':    '#7a9cc4',   # muted blue for progress output
        }
        prefixes = {
            'info': '›', 'success': '✓', 'warning': '⚠',
            'error': '✗', 'tqdm': ' ',
        }
        colour = colours.get(level, TEXT_DIM)
        prefix = prefixes.get(level, '›')
        # escape HTML special chars
        message = message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return f'<span style="color:{colour};">{prefix} {message}</span>'

    def _do_append(self, html: str):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        if not self.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(html)
        self.moveCursor(QTextCursor.End)
        self._last_was_tqdm = False   # normal append resets tqdm tracking
        self._last_tqdm_key = None

    def _do_replace_last(self, html: str):
        """
        Replace the last non-empty block — used for tqdm progress lines.
        Walks backwards through document blocks to skip any empty trailing ones.
        """
        doc   = self.document()
        block = doc.lastBlock()
        while block.isValid() and not block.text().strip():
            block = block.previous()

        if block.isValid() and block.text().strip():
            # Select only this block's text (start→end), NOT BlockUnderCursor —
            # the latter includes the leading paragraph separator, so removing it
            # merges into the previous line and eats earlier output.
            cursor = QTextCursor(block)
            cursor.movePosition(QTextCursor.StartOfBlock)
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertHtml(html)
        else:
            self._do_append(html)

        self.moveCursor(QTextCursor.End)

    def clear_log(self):
        self.clear()
        self._last_was_tqdm = False
        self._last_tqdm_key = None


# ── Worker (background thread) ────────────────────────────────────────────────

class Worker(QObject):
    """
    Runs a callable in a background QThread.
    stdout/stderr are already redirected globally by LogWidget.install_redirectors(),
    so tqdm and print output appears in the log automatically.
    """
    finished = Signal()
    error    = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn     = fn
        self.args   = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.fn(*self.args, **self.kwargs)
            self.finished.emit()
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# ── Collapsible section ───────────────────────────────────────────────────────

class CollapsibleSection(QWidget):
    """
    A labelled section with a clickable header (▼/▶) that shows/hides its content.

    Usage:
        section = CollapsibleSection("Subjects")
        section.add_widget(some_widget)
        layout.addWidget(section)
    """

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # ── Header button ──
        self._header = QPushButton()
        self._header.setFlat(True)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.clicked.connect(self._toggle)
        self._header.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                font-size: 14px;
                font-weight: bold;
                color: #e8e6f0;
                background: transparent;
                border: none;
                border-bottom: 1px solid #3d3860;
                padding: 6px 4px;
            }}
            QPushButton:hover {{
                color: #9e1e62;
            }}
        """)
        outer.addWidget(self._header)

        # ── Content container ──
        self._container = QWidget()
        self._content   = QVBoxLayout(self._container)
        self._content.setContentsMargins(8, 4, 0, 8)
        self._content.setSpacing(4)
        outer.addWidget(self._container)

        self._set_title(title)
        self._container.setVisible(expanded)

    def _set_title(self, title: str):
        arrow = "▼" if self._expanded else "▶"
        self._header.setText(f"  {arrow}  {title}")
        self._title = title

    def _toggle(self):
        self._expanded = not self._expanded
        self._container.setVisible(self._expanded)
        self._set_title(self._title)

    def add_widget(self, widget: QWidget):
        self._content.addWidget(widget)

    def add_layout(self, layout):
        self._content.addLayout(layout)

    def set_expanded(self, expanded: bool):
        if expanded != self._expanded:
            self._toggle()


# ── Detail panel (right-hand side base class) ─────────────────────────────────

class DetailPanel(QWidget):
    """
    Base class for right-hand panels. Provides scrollable layout,
    heading helpers, and background thread runner.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # outer layout holds the scroll area
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        inner_widget = QWidget()
        self._layout = QVBoxLayout(inner_widget)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(10)

        scroll.setWidget(inner_widget)
        outer.addWidget(scroll)

        self._thread = None
        self._worker = None

    def _add_heading(self, title: str, subtitle: str = ""):
        heading = QLabel(title)
        heading.setObjectName("heading")
        self._layout.addWidget(heading)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("subheading")
            self._layout.addWidget(sub)

    def _run_in_thread(self, fn, log_widget: 'LogWidget', *args, **kwargs):
        """Run fn in a QThread. Clears log first so tqdm output is clean."""
        log_widget.clear_log()

        self._thread = QThread()
        self._worker = Worker(fn, *args, **kwargs)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(lambda: log_widget.log("Done.", 'success'))
        # refresh the detail panel so status dots/outputs update immediately
        # (bound method => queued onto the GUI thread, safe for widget rebuilds)
        self._worker.finished.connect(self._notify_panel_done)
        self._worker.error.connect(self._thread.quit)
        self._worker.error.connect(lambda e: log_widget.log(e, 'error'))
        self._thread.start()

    def _notify_panel_done(self):
        """Ask the main window to rebuild this panel so new outputs show up."""
        main = self.window()
        if main is not None and hasattr(main, '_refresh_current_panel'):
            main._refresh_current_panel()
