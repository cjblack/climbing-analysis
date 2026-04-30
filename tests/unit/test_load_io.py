import pytest
import pandas as pd
import yaml

from neurokinematics.io import load_csv, load_json, load_config

def test_load_csv_pandas(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("d1,d2\n0,1\n0,1\n")

    df = load_csv(csv_path, method = 'pandas')

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['d1', 'd2']
    assert df.shape == (2, 2)

def test_load_json(tmp_path):
    json_path = tmp_path / "test.json"
    json_path.write_text('{"d1": 0, "d2": 1}')

    data = load_json(json_path)

    assert data == {"d1": 0, "d2": 1}

def test_load_config(tmp_path):
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("d1: 0\nd2: 1\n")

    cfg = load_config(cfg_path, config_type=None)

    assert cfg["d1"] == 0
    assert cfg["d2"] == 1

def test_load_csv_missing(tmp_path):
    missing_file = tmp_path / "missing_file.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing_file)

def test_load_csv_non_method(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("d1,d2\n0,1\n0,1\n")

    with pytest.raises(ValueError):
        load_csv(csv_path, method='non_method')