from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse

from dj_config import connect
from schema import (
    Subject,
    Session,
    EphysRecordNode,
    EphysExperiment,
    EphysRecording,
)

def get_session_id(subject_id: str, session_path: str) -> int:
    session_path_posix = str(Path(session_path).as_posix())

    existing = Session & {
        "subject_id": subject_id,
        "session_path": session_path_posix
    }

    if len(existing) > 0:
        return int(existing.fetch1("session_id"))
    
    subject_session = Session & {"subject_id": subject_id}
    if len(subject_session) == 0:
        return 1
    
    max_id = max(subject_session.fetch("session_id"))
    return int(max_id)+1

def run(session_path: str):
    connect()
    
    session_path = str(Path(session_path).as_posix())
    folder_name = Path(session_path).name

    # example folder name style:
    # subj-001_yyyy-mm-dd_hh-mm-ss_task
    path_breakdown = folder_name.split('_')
    if len(path_breakdown) < 4:
        raise ValueError(
            f"Session folder name '{folder_name}' does not match expected pattern: "
            f"<subject_id>_<date>_<time>_<task>"
        )
    
    subject_id = path_breakdown[0]
    date_string = path_breakdown[1]
    clock_string = path_breakdown[2]
    task_string = path_breakdown[3]

    session_datetime = datetime.strptime(f"{date_string}_{clock_string}", '%Y-%m-%d_%H-%M-%S')
    session_id = get_session_id(subject_id, session_path)
    subject_key = {"subject_id": subject_id}
    session_key = {"subject_id": subject_id, "session_id": session_id}
  
    Subject.insert1({
        **subject_key,
        "sex": "F",
        "genotype": "sod1-wt"
    }, skip_duplicates=True)

    Session.insert1({
        **session_key,
        "session_datetime": str(session_datetime),
        "session_path": str(Path(session_path).as_posix()),
        "notes": task_string
    }, skip_duplicates=True)

    # populate imports
    # EphysRecording.drop()
    # EphysExperiment.drop()
    # EphysRecordNode.drop()
    
    EphysRecordNode.populate(session_key, display_progress=True)
    EphysExperiment.populate(session_key, display_progress=True)
    EphysRecording.populate(session_key, display_progress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a session folder")
    parser.add_argument(
        "session_path",
        type=str,
        help="Path to session folder"
    )

    args = parser.parse_args()
    run(args.session_path)