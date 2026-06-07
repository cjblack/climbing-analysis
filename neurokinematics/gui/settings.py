"""Persistent GUI settings for neurokinematics (stored in the user's home).

Currently holds the user's phy2 setup so the GUI can launch phy on a sorted
session. Settings live at ``~/.neurokinematics/gui_settings.json``.
"""

import json
import os
import sys
import subprocess
from pathlib import Path


def _phy_subprocess_env() -> dict:
    """Environment for the phy2 subprocess with Qt vars stripped.

    phy uses PyQt5; this GUI uses PySide6. If the host's Qt plugin/platform
    variables leak into the child, PyQt5 can load the wrong plugins and emit
    'QMimeDatabase: ... Premature end of document' and similar warnings. We hand
    phy a copy of the environment with those variables removed so its conda env
    sets up its own Qt cleanly.
    """
    child = dict(os.environ)
    for var in (
        "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_PLATFORM",
        "QML2_IMPORT_PATH", "QML_IMPORT_PATH", "QT_DEBUG_PLUGINS",
        "QT_QPA_FONTDIR", "QT_OPENGL", "QT_QUICK_BACKEND", "QT_SCALE_FACTOR",
    ):
        child.pop(var, None)
    return child

SETTINGS_DIR  = Path.home() / ".neurokinematics"
SETTINGS_PATH = SETTINGS_DIR / "gui_settings.json"

DEFAULTS = {
    "phy_env":      "",            # conda env where phy2 is installed
    "phy_gui":      "template-gui", # phy gui subcommand
    "conda_exe":    "conda",       # how to invoke conda (name on PATH or full path)
    "recents":      [],            # recently loaded projects/groups/subjects
    "default_root": "",            # default base dir for new projects / browse dialogs
    "default_ephys_root": "",      # default start dir when browsing for ephys data
    "default_pose_root": "",       # default start dir when browsing for pose data
    "spike_bad_channel_policy": "ask",  # ask | remove | keep (manual-check skip behaviour)
}

MAX_RECENTS = 8


def get_default_root() -> str:
    """Configured default project/data root, or '' if unset.

    Returned as a string so callers can pass it straight to a file dialog's
    start directory. Empty string means 'no preference' (fall back to home /
    the OS default), matching the data layer's NKProject behaviour.
    """
    root = load_settings().get("default_root", "") or ""
    return root if root and Path(root).is_dir() else ""


def get_data_root(kind: str) -> str:
    """Default browse dir for 'ephys' or 'pose' data, or '' if unset/missing."""
    key = "default_ephys_root" if kind == "ephys" else "default_pose_root"
    root = load_settings().get(key, "") or ""
    return root if root and Path(root).is_dir() else ""


def load_settings() -> dict:
    data = dict(DEFAULTS)
    try:
        if SETTINGS_PATH.exists():
            stored = json.loads(SETTINGS_PATH.read_text())
            if isinstance(stored, dict):
                data.update(stored)
    except Exception:
        pass
    return data


def save_settings(settings: dict) -> dict:
    # Start from any existing on-disk settings so a partial save (e.g. the
    # Settings dialog writing only phy keys) doesn't wipe unrelated keys like
    # "recents". Precedence: DEFAULTS < stored-on-disk < provided.
    merged = load_settings()
    merged.update(settings or {})
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
    return merged


def load_recents() -> list:
    """Return the recent-items list, newest first."""
    recents = load_settings().get("recents", [])
    return recents if isinstance(recents, list) else []


def add_recent(kind: str, path: str, label: str,
               project_name: str | None = None) -> list:
    """Record a recently opened item, de-duplicating by (kind, path).

    *kind* is one of 'project', 'group', 'subject'. *path* is the spec path
    (group/subject) or project directory used to re-open it. Returns the
    updated recents list (newest first, capped at MAX_RECENTS).
    """
    path = str(path)
    entry = {"kind": kind, "path": path, "label": label,
             "project_name": project_name}
    recents = [r for r in load_recents()
               if not (r.get("kind") == kind and r.get("path") == path)]
    recents.insert(0, entry)
    recents = recents[:MAX_RECENTS]
    save_settings({"recents": recents})
    return recents


def clear_recents() -> None:
    save_settings({"recents": []})


def launch_phy(phy_dir, env: str, gui: str = "template-gui", conda_exe: str = "conda"):
    """Open phy2 in a separate window for the given phy_output dir.

    Activates the user's phy conda environment and runs
    ``phy <gui> params.py`` with *phy_dir* as the working directory. Raises if
    no env is configured or params.py is missing.
    """
    phy_dir = Path(phy_dir)
    if not env:
        raise ValueError("No phy2 environment configured. Set it in File → Settings.")
    if not (phy_dir / "params.py").exists():
        raise FileNotFoundError(f"params.py not found in {phy_dir}")

    gui = gui or "template-gui"
    child_env = _phy_subprocess_env()
    if sys.platform == "win32":
        # open a fresh console that activates the env and launches phy
        inner = f'conda activate {env} && phy {gui} params.py'
        subprocess.Popen(
            f'start "phy2 - neurokinematics" cmd /k "{inner}"',
            cwd=str(phy_dir), shell=True, env=child_env,
        )
    else:
        subprocess.Popen(
            [conda_exe or "conda", "run", "-n", env, "phy", gui, "params.py"],
            cwd=str(phy_dir), env=child_env,
        )
