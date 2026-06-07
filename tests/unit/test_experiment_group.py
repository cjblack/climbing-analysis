from pathlib import Path
import pytest
import yaml

from neurokinematics.data.group import ExperimentGroup


def _minimal_group_spec(output_root: str) -> dict:
    """Return the smallest valid group spec dict (no subjects)."""
    return {
        "group_id": "test_group",
        "subjects": [],
        "output_root": output_root,
    }


# ---------------------------------------------------------------------------
# _load_group_specs — validation
# ---------------------------------------------------------------------------

def test_load_group_specs_valid_dict(tmp_path):
    group = ExperimentGroup(group_specs=None, project_path=tmp_path)
    result = group._load_group_specs(_minimal_group_spec(str(tmp_path)))
    assert result["group_id"] == "test_group"


def test_load_group_specs_missing_group_id_raises(tmp_path):
    group = ExperimentGroup(group_specs=None, project_path=tmp_path)
    bad_spec = {"subjects": []}
    with pytest.raises(ValueError, match="group_id"):
        group._load_group_specs(bad_spec)


def test_load_group_specs_missing_subjects_raises(tmp_path):
    group = ExperimentGroup(group_specs=None, project_path=tmp_path)
    bad_spec = {"group_id": "g1"}
    with pytest.raises(ValueError, match="subjects"):
        group._load_group_specs(bad_spec)


def test_load_group_specs_invalid_type_raises(tmp_path):
    group = ExperimentGroup(group_specs=None, project_path=tmp_path)
    with pytest.raises(ValueError):
        group._load_group_specs(["not", "a", "dict"])


def test_load_group_specs_from_yaml_file(tmp_path):
    spec = _minimal_group_spec(str(tmp_path))
    spec_path = tmp_path / "group_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))

    group = ExperimentGroup(group_specs=None, project_path=tmp_path)
    result = group._load_group_specs(spec_path)
    assert result["group_id"] == "test_group"


# ---------------------------------------------------------------------------
# ExperimentGroup construction (no subjects)
# ---------------------------------------------------------------------------

def test_create_group_no_subjects(tmp_path):
    """Group created from a spec with no subjects initialises cleanly."""
    spec = _minimal_group_spec(str(tmp_path))
    group = ExperimentGroup(group_specs=spec, project_path=tmp_path)

    assert group.group_id == "test_group"
    assert group.group_path.exists()
    assert (group.group_path / "group_spec.yaml").exists()


def test_group_directory_structure_created(tmp_path):
    spec = _minimal_group_spec(str(tmp_path))
    group = ExperimentGroup(group_specs=spec, project_path=tmp_path)

    assert group.dirs["summaries"].exists()
    assert group.dirs["stats"].exists()
    assert group.dirs["results"].exists()


def test_group_from_existing_roundtrip(tmp_path):
    """from_existing reloads a group with matching attributes."""
    spec = _minimal_group_spec(str(tmp_path))
    group = ExperimentGroup(group_specs=spec, project_path=tmp_path)
    group_path = group.group_path

    reloaded = ExperimentGroup.from_existing(group_path)
    assert reloaded.group_id == group.group_id
    assert reloaded.group_path == group.group_path
