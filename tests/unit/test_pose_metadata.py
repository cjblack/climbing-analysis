"""Tests for pose file metadata resolution (neurokinematics.pose.metadata).

Covers the sidecar-CSV ``manifest`` source: default location next to the data,
basename matching, and clear errors for a missing manifest / unlisted file /
malformed columns. Also covers ``filename`` regex parsing and unknown sources.
"""

import pandas as pd
import pytest

from neurokinematics.pose.metadata import (
    resolve_file_metadata,
    DEFAULT_MANIFEST_NAME,
)


def _write_manifest(folder, rows, name=DEFAULT_MANIFEST_NAME):
    path = folder / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_manifest_default_location(tmp_path):
    f = tmp_path / "reach_T1.h5"
    f.write_text("")
    _write_manifest(tmp_path, [
        {"file": "reach_T1.h5", "Id": "m1", "Type": "reach", "Date": "20240312", "Trial": "1"},
        {"file": "reach_T2.h5", "Id": "m1", "Type": "reach", "Date": "20240312", "Trial": "2"},
    ])

    meta = resolve_file_metadata(str(f), {"source": "manifest"})
    assert meta == {"Id": "m1", "Type": "reach", "Date": "20240312", "Trial": "1"}


def test_manifest_matches_on_basename(tmp_path):
    f = tmp_path / "reach_T1.h5"
    f.write_text("")
    # manifest lists a full path; resolver should still match by basename
    _write_manifest(tmp_path, [
        {"file": str(f), "Id": "m1", "Type": "reach", "Date": "20240312", "Trial": "1"},
    ])
    meta = resolve_file_metadata(str(f), {"source": "manifest"})
    assert meta["Trial"] == "1"


def test_manifest_custom_path_and_key(tmp_path):
    f = tmp_path / "reach_T1.h5"
    f.write_text("")
    _write_manifest(tmp_path, [
        {"fname": "reach_T1.h5", "Id": "m1", "Type": "reach", "Date": "20240312", "Trial": "1"},
    ], name="custom.csv")
    meta = resolve_file_metadata(
        str(f), {"source": "manifest", "path": "custom.csv", "key": "fname"}
    )
    assert meta["Id"] == "m1"


def test_manifest_missing_file_raises(tmp_path):
    f = tmp_path / "reach_T1.h5"
    f.write_text("")
    with pytest.raises(FileNotFoundError):
        resolve_file_metadata(str(f), {"source": "manifest"})


def test_manifest_unlisted_file_raises(tmp_path):
    f = tmp_path / "reach_T9.h5"
    f.write_text("")
    _write_manifest(tmp_path, [
        {"file": "reach_T1.h5", "Id": "m1", "Type": "reach", "Date": "20240312", "Trial": "1"},
    ])
    with pytest.raises(ValueError, match="not listed"):
        resolve_file_metadata(str(f), {"source": "manifest"})


def test_manifest_missing_column_raises(tmp_path):
    f = tmp_path / "reach_T1.h5"
    f.write_text("")
    _write_manifest(tmp_path, [
        {"file": "reach_T1.h5", "Id": "m1", "Type": "reach", "Date": "20240312"},  # no Trial
    ])
    with pytest.raises(ValueError, match="missing column"):
        resolve_file_metadata(str(f), {"source": "manifest"})


def test_filename_source_still_works(tmp_path):
    f = tmp_path / "m1_reach_20240312_T1.h5"
    pattern = r'(?P<Id>[^_]+)_(?P<Type>[^_]+)_(?P<Date>\d{8})_T(?P<Trial>\d+)'
    meta = resolve_file_metadata(str(f), {"source": "filename", "pattern": pattern})
    assert meta == {"Id": "m1", "Type": "reach", "Date": "20240312", "Trial": "1"}


def test_unknown_source_raises(tmp_path):
    f = tmp_path / "reach_T1.h5"
    with pytest.raises(ValueError, match="Unknown pose metadata source"):
        resolve_file_metadata(str(f), {"source": "nonsense"})
