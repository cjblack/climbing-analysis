""" 
Example: reload previously created minimal pose neurokinematics session

This script demonstrates how to load a previously created ExperimentSession.

Before running:
    1. Create and activate conda environment
    2. Ensure pose sample data from (https://doi.org/10.17605/OSF.IO/3SR67) exists in examples/sample_data/
    3. Ensure either `run_pose_example.py` or `run_session_pipeline.py` have already been performed

"""

from pathlib import Path

from neurokinematics.data.session import ExperimentSession


def main():

    repo_root = Path(__file__).resolve().parents[1]
    session_path = repo_root / 'examples' / 'output' / 'pose_demo_nk'
    print(session_path)
    session = ExperimentSession.from_existing(session_path=session_path)

    session.run_pose_processing() # should return path to data

    print(session)


if __name__ == "__main__":
    main()