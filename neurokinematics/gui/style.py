"""
neurokinematics colour scheme and dark theme stylesheet.

Primary:   #9e1e62  (magenta)
Secondary: #262161  (dark purple)
"""

# Brand colours
PRIMARY   = "#9e1e62"
SECONDARY = "#262161"

# Derived palette
BG_DARK    = "#1a1829"   # near-black with purple tint
BG_MID     = "#242038"   # panel backgrounds
BG_LIGHT   = "#2e2a47"   # hover / selected
BORDER     = "#3d3860"   # subtle borders
TEXT       = "#e8e6f0"   # primary text
TEXT_DIM   = "#8e8aaa"   # secondary / placeholder text
ACCENT     = PRIMARY     # buttons, highlights
ACCENT_HOV = "#b8246f"   # button hover
ACCENT_PRE = "#7a1750"   # button pressed
SUCCESS    = "#4caf82"   # green for completed
WARNING    = "#e0a050"   # amber for warnings
ERROR      = "#e05050"   # red for errors


# ── Brand colour range (for plots) ──────────────────────────────────
# Ordered dark → light across the neurokinematics palette. Use the list
# for categorical series and the colormap for continuous gradients.
BRAND_COLORS = [SECONDARY, "#5a2f78", PRIMARY, ACCENT_HOV, "#e08fb8"]


def brand_cmap():
    """LinearSegmentedColormap spanning the neurokinematics brand range.

    Dark purple (#262161) → magenta (#9e1e62) → light pink (#e08fb8),
    a drop-in replacement for sequential maps like ``cm.cool``.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("neurokinematics", BRAND_COLORS)


STYLESHEET = f"""
/* ── Global ─────────────────────────────────────────── */
QWidget {{
    background-color: {BG_DARK};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background-color: {BG_DARK};
}}

/* ── Menu bar ────────────────────────────────────────── */
QMenuBar {{
    background-color: {BG_MID};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
    padding: 2px;
}}
QMenuBar::item:selected {{
    background-color: {BG_LIGHT};
    border-radius: 4px;
}}
QMenu {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {ACCENT};
    color: white;
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

/* ── Group boxes ─────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding: 8px;
    font-weight: bold;
    color: {TEXT_DIM};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}

/* ── Line edits ──────────────────────────────────────── */
QLineEdit {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {TEXT};
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:placeholder {{
    color: {TEXT_DIM};
}}

/* ── Buttons ─────────────────────────────────────────── */
QPushButton {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {ACCENT_HOV};
}}
QPushButton:pressed {{
    background-color: {ACCENT_PRE};
}}
QPushButton:disabled {{
    background-color: {BORDER};
    color: {TEXT_DIM};
}}
/* secondary button style defined at bottom of sheet */
QPushButton#run {{
    background-color: {SUCCESS};
    color: white;
}}
QPushButton#run:hover {{
    background-color: #3d9e6a;
}}

/* ── Combo boxes ─────────────────────────────────────── */
QComboBox {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {TEXT};
    min-width: 90px;
}}
QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    color: {TEXT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

/* ── Tree widget ─────────────────────────────────────── */
QTreeWidget {{
    background-color: {BG_MID};
    border: none;
    border-right: 1px solid {BORDER};
    outline: none;
}}
QTreeWidget::item {{
    padding: 6px 4px;
    border-radius: 4px;
}}
QTreeWidget::item:selected {{
    background-color: {ACCENT};
    color: white;
}}
QTreeWidget::item:hover:!selected {{
    background-color: {BG_LIGHT};
}}
QTreeWidget::branch {{
    background-color: {BG_MID};
    width: 0px;
}}

/* ── List widget ─────────────────────────────────────── */
QListWidget {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 5px 8px;
    border-radius: 3px;
}}
QListWidget::item:selected {{
    background-color: {ACCENT};
    color: white;
}}

/* ── Check boxes ─────────────────────────────────────── */
QCheckBox {{
    color: {TEXT};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {BG_MID};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* ── Splitter ────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {BORDER};
    width: 1px;
}}

/* ── Scroll bars ─────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG_MID};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── Text edit (log) ─────────────────────────────────── */
QTextEdit {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 4px;
}}

/* ── Dialog buttons ──────────────────────────────────── */
QDialogButtonBox QPushButton {{
    min-width: 80px;
}}

/* ── Table widget ────────────────────────────────────── */
QTableWidget {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    gridline-color: {BORDER};
    color: {TEXT};
}}
QTableWidget::item {{
    padding: 4px 8px;
}}
QTableWidget::item:selected {{
    background-color: {ACCENT};
    color: white;
}}
QHeaderView::section {{
    background-color: {SECONDARY};
    color: {TEXT};
    padding: 6px 8px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: bold;
    font-size: 12px;
}}
QHeaderView::section:horizontal:first {{
    border-left: none;
}}
QHeaderView {{
    background-color: {SECONDARY};
}}

/* ── Labels ──────────────────────────────────────────── */
QLabel#heading {{
    font-size: 16px;
    font-weight: bold;
    color: {TEXT};
}}
QLabel#subheading {{
    font-size: 12px;
    color: {TEXT_DIM};
}}
QLabel#tag {{
    background-color: {SECONDARY};
    color: {TEXT};
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 11px;
}}

/* ── Tab widget ──────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
}}
QTabBar::tab {{
    background-color: {BG_MID};
    color: {TEXT_DIM};
    padding: 6px 16px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT};
    background-color: {BG_LIGHT};
}}

/* ── Secondary / utility buttons (expand, collapse, etc.) ─ */
QPushButton#secondary {{
    background-color: {TEXT};
    color: {BG_DARK};
    border: none;
    font-weight: bold;
}}
QPushButton#secondary:hover {{
    background-color: {TEXT_DIM};
    color: {BG_DARK};
}}
QPushButton#secondary:pressed {{
    background-color: {BORDER};
    color: {TEXT};
}}

/* ── Dock widget ─────────────────────────────────────── */
QDockWidget {{
    color: {TEXT};
    font-weight: bold;
    titlebar-close-icon: url(none);
    titlebar-normal-icon: url(none);
}}
QDockWidget::title {{
    background-color: {SECONDARY};
    color: {TEXT};
    padding: 6px 8px;
    border-bottom: 1px solid {BORDER};
}}
QDockWidget::close-button,
QDockWidget::float-button {{
    background-color: {TEXT};
    color: {BG_DARK};
    border: none;
    border-radius: 3px;
    padding: 2px;
    subcontrol-position: top right;
}}
QDockWidget::close-button:hover,
QDockWidget::float-button:hover {{
    background-color: {ACCENT};
    color: white;
}}

/* tree branch column hidden — ▶/▼ drawn in item text instead */
"""
