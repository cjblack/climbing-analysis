from __future__ import annotations

from pathlib import Path
from typing import Any
import glob

import datajoint as dj
import pandas as pd

from dj_config import connect

connect()

schema = dj.schema("climbing_analysis")

# Helpers

def _list_subdirs(path: Path):
    """Returns list of folders in path, use this for quick iteration of open ephys stored data

    Args:
        path (Path): Path variable for parent folder

    Returns:
        list: Sorted list of all children folders
    """
    return sorted([p for p in path.iterdir() if p.is_dir()])

def discover_ephys_record_nodes(session_path: Path) -> list[dict[str, Any]]:
    nodes = []
    
    for p in _list_subdirs(session_path):
        if "Record Node" in p.name:
            node_id = int(p.name.split()[-1])
            nodes.append({
                "record_node_id": node_id,
                "record_node_path": str(p.as_posix())
            })
    
    return nodes

def discover_ephys_experiments(node_path: Path) -> list[dict[str, Any]]:
    experiments = []

    for p in _list_subdirs(node_path):
        if "experiment" in p.name:
            experiment_id = int(p.name.split('experiment')[-1])
            experiments.append({
                "experiment_id": experiment_id,
                "experiment_path": str(p.as_posix())
            })
    
    return experiments

def discover_ephys_recordings(experiemnt_path: Path) -> list[dict[str, Any]]:
    recordings = []

    for p in _list_subdirs(experiemnt_path):
        if "recording" in p.name:
            recording_id = int(p.name.split('recording')[-1])
            recordings.append({
                "recording_id": recording_id,
                "recording_path": str(p.as_posix())
            })
            
    return recordings


# CORE TABLES

@schema
class Subject(dj.Manual):
    definition = """
    subject_id: varchar(32)
    ---
    sex='unknown': varchar(16)
    genotype='': varchar(128)
    dob=null: date
    """

@schema
class Session(dj.Manual):
    definition = """
    -> Subject
    session_id: int
    ---
    session_datetime: datetime
    session_path: varchar(512)
    notes='': varchar(1024)

    """


# IMPORTED
@schema
class EphysRecordNode(dj.Imported):
    definition = """
    -> Session
    record_node_id: int
    ---
    record_node_path: varchar(512)
    """

    def make(self, key):
        session_path = Path((Session & key).fetch1("session_path"))

        rows = discover_ephys_record_nodes(session_path)
        if not rows:
            raise FileNotFoundError(f"No ephys record nodes found in {session_path}")
        
        inserts = []
        for row in rows:
            inserts.append({
                **key,
                "record_node_id": row["record_node_id"],
                "record_node_path": str(Path(row["record_node_path"]).as_posix())
            })
        self.insert(inserts, skip_duplicates=True)


@schema
class EphysExperiment(dj.Imported):
    definition = """
    -> EphysRecordNode
    experiment_id: int
    ---
    experiment_path: varchar(512)
    """
    def make(self, key):
        record_node_path = Path((EphysRecordNode & key).fetch1("record_node_path"))

        rows = discover_ephys_experiments(record_node_path)
        if not rows:
            raise FileNotFoundError(f"No ephys experiments found in {record_node_path}")
        
        inserts = []
        for row in rows:
            inserts.append({
                **key,
                "experiment_id": row["experiment_id"],
                "experiment_path": str(Path(row["experiment_path"]).as_posix())
            })
        self.insert(inserts, skip_duplicates=True)

@schema
class EphysRecording(dj.Imported):
    definition = """
    -> EphysExperiment
    recording_id: int
    ---
    recording_path: varchar(512)
    """

    def make(self, key):
        experiment_path = Path((EphysExperiment & key).fetch1("experiment_path"))

        rows = discover_ephys_recordings(experiment_path)
        if not rows:
            raise FileNotFoundError(f"No ephys recordings found in {experiment_path}")
        
        inserts = []
        for row in rows:
            inserts.append({
                **key,
                "recording_id": row["recording_id"],
                "recording_path": str(Path(row["recording_path"]).as_posix())
            })
        self.insert(inserts, skip_duplicates=True)