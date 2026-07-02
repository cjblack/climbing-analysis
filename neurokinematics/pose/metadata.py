from pathlib import Path
from functools import lru_cache
import re

import pandas as pd

# Fields every resolver must return for a file. ``Date`` / ``Trial`` are coerced
# (``pd.to_datetime`` / ``int``) by the caller in ``neurokinematics.pose.io``.
REQUIRED_FIELDS = ('Id', 'Type', 'Date', 'Trial')

# Default sidecar manifest written next to the pose files (e.g. by the GUI
# "Metadata" dialog) when the config doesn't point somewhere else.
DEFAULT_MANIFEST_NAME = 'pose_metadata.csv'


def resolve_file_metadata(filename, meta_cfg):
    source = meta_cfg.get('source', 'filename')
    if source not in _METADATA_RESOLVERS:
        valid = ", ".join(_METADATA_RESOLVERS)
        raise ValueError(
            f"Unknown pose metadata source {source!r}. Valid options: {valid}."
        )
    return _METADATA_RESOLVERS[source](filename, meta_cfg)


def _from_filename(filename, meta_cfg):
    name = Path(filename).name
    m = re.match(meta_cfg['pattern'], name)
    if not m:
        raise ValueError(
            f"Filename {name!r} did not match pattern "
            f"{meta_cfg['pattern']!r}. Check pose_format.metadata in config to reformat pattern."
        )
    meta = m.groupdict()
    return meta


def _from_manifest(filename, meta_cfg):
    """Look up a file's metadata in a sidecar CSV manifest.

    The manifest is a table with one row per pose file. By default it lives next
    to the data as ``pose_metadata.csv`` (this is what the GUI metadata dialog
    writes), with columns ``file, Id, Type, Date, Trial``. Set
    ``pose_format.metadata.path`` to point elsewhere and ``key`` to rename the
    filename column. Matching is on the file's basename so the manifest can list
    bare names or full paths.
    """
    manifest_path = _manifest_path(filename, meta_cfg)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Pose metadata manifest not found at {manifest_path}. Create it "
            f"(e.g. via the GUI 'Metadata' dialog) or set "
            f"'pose_format.metadata.path' in the pose config."
        )

    key = meta_cfg.get('key', 'file')
    table = _read_manifest(str(manifest_path), manifest_path.stat().st_mtime, key)

    name = Path(filename).name
    if name not in table:
        raise ValueError(
            f"File {name!r} is not listed in manifest {manifest_path.name!r}. "
            f"Add a row for it (column {key!r}) with {', '.join(REQUIRED_FIELDS)}."
        )
    return table[name]


def _manifest_path(filename, meta_cfg):
    """Resolve the manifest location: an explicit ``path`` (absolute, or relative
    to the data folder) or the default sidecar name in the data folder."""
    data_dir = Path(filename).parent
    raw = meta_cfg.get('path')
    if not raw:
        return data_dir / DEFAULT_MANIFEST_NAME
    p = Path(raw)
    return p if p.is_absolute() else data_dir / p


@lru_cache(maxsize=8)
def _read_manifest(manifest_path, mtime, key):
    """Load + index a manifest CSV by file basename.

    Cached on ``(path, mtime)`` so a batch of files doesn't re-read the CSV once
    per file, while still picking up edits (mtime change busts the cache).
    """
    df = pd.read_csv(manifest_path, dtype=str).fillna('')
    missing = [c for c in (key, *REQUIRED_FIELDS) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manifest {Path(manifest_path).name!r} is missing column(s): "
            f"{', '.join(missing)}. Expected: {key}, {', '.join(REQUIRED_FIELDS)}."
        )
    table = {}
    for row in df.to_dict('records'):
        fname = Path(str(row[key])).name
        table[fname] = {f: row[f] for f in REQUIRED_FIELDS}
    return table


def _from_sidecar(filename, meta_cfg):
    pass


def _from_custom(filename, meta_cfg):
    """Add your own custom layout for parsing your metadata

    Ensure that metadata returns a dictionary containing
    {
        'Id': str, # subject id
        'Type': str, # experiment type/name
        'Date': datetime, # date experiment was performed
        'Trial': int # id of trial - this is used for sorting/querying dataframes later on
    }

    """
    pass


_METADATA_RESOLVERS = {
    'filename': _from_filename,
    'manifest': _from_manifest,
    'sidecar': _from_sidecar,
    'custom': _from_custom
}
