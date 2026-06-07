from pathlib import Path
import pytest
import yaml

from neurokinematics.data.subject import ExperimentSubject


def _minimal_spec(output_root: str) -> dict:
    """Return the smallest valid subject spec dict (no sessions)."""
    return {
        "subject_id": "test_subject",
        "output_root": output_root,
        "process": {"pose": False, "spike": False, "lfp": False},
    }


# ---------------------------------------------------------------------------
# _load_subject_specs — validation
# ---------------------------------------------------------------------------

def test_load_subject_specs_valid_dict(tmp_path):
    subject = ExperimentSubject(subject_specs=None, project_path=tmp_path)
    result = subject._load_subject_specs(_minimal_spec(str(tmp_path)))
    assert result["subject_id"] == "test_subject"


def test_load_subject_specs_missing_key_raises(tmp_path):
    subject = ExperimentSubject(subject_specs=None, project_path=tmp_path)
    bad_spec = {"subject_id": "s1", "output_root": str(tmp_path)}  # missing 'process'
    with pytest.raises(ValueError, match="Missing"):
        subject._load_subject_specs(bad_spec)


def test_load_subject_specs_invalid_type_raises(tmp_path):
    subject = ExperimentSubject(subject_specs=None, project_path=tmp_path)
    with pytest.raises(ValueError):
        subject._load_subject_specs(42)


def test_load_subject_specs_invalid_file_extension_raises(tmp_path):
    subject = ExperimentSubject(subject_specs=None, project_path=tmp_path)
    fake_json = tmp_path / "spec.json"
    fake_json.write_text("{}")
    with pytest.raises(ValueError, match=".yaml"):
        subject._load_subject_specs(fake_json)


def test_load_subject_specs_from_yaml_file(tmp_path):
    spec = _minimal_spec(str(tmp_path))
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))

    subject = ExperimentSubject(subject_specs=None, project_path=tmp_path)
    result = subject._load_subject_specs(spec_path)
    assert result["subject_id"] == "test_subject"


# ---------------------------------------------------------------------------
# ExperimentSubject construction (no sessions in spec)
# ---------------------------------------------------------------------------

def test_create_subject_no_sessions(tmp_path):
    """Subject created from a spec with no sessions list initialises cleanly."""
    spec = _minimal_spec(str(tmp_path))
    subject = ExperimentSubject(subject_specs=spec, project_path=tmp_path)

    assert subject.subject_id == "test_subject"
    assert subject.subject_path.exists()
    assert (subject.subject_path / "subject_spec.yaml").exists()


def test_create_subject_saves_spec_yaml(tmp_path):
    spec = _minimal_spec(str(tmp_path))
    subject = ExperimentSubject(subject_specs=spec, project_path=tmp_path)

    saved = yaml.safe_load((subject.subject_path / "subject_spec.yaml").read_text())
    assert saved["subject_id"] == "test_subject"


def test_subject_from_existing_roundtrip(tmp_path):
    """from_existing reloads a subject whose spec already contains a runtime block."""
    spec = _minimal_spec(str(tmp_path))
    subject = ExperimentSubject(subject_specs=spec, project_path=tmp_path)

    # Manually inject a runtime block so from_existing can parse the spec.
    # A subject with no sessions never calls create_sessions_from_log, so
    # runtime is not written automatically in that code path.
    spec_path = subject.subject_path / "subject_spec.yaml"
    saved = yaml.safe_load(spec_path.read_text())
    saved["runtime"] = {"sessions": []}
    spec_path.write_text(yaml.safe_dump(saved))

    reloaded = ExperimentSubject.from_existing(subject.subject_path)
    assert reloaded.subject_id == subject.subject_id
    assert reloaded.subject_path == subject.subject_path
