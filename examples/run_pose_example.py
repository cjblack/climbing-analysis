""" 
Example: run a simple version of neurokinematics session for pose data

This script demonstrates how to create an ExperimentSession, and perform processing of pose data.

Before running:
    1. Create and activate conda environment
    2. Ensure pose sample data from (https://doi.org/10.17605/OSF.IO/3SR67) exists in examples/sample_data/

"""

from pathlib import Path

from neurokinematics.data.session import ExperimentSession


def main():

    repo_root = Path(__file__).resolve().parents[1]

    session = ExperimentSession(
        session_id="pose_demo",
        ephys_data_path=repo_root / "examples" / "sample_data" / "dummy_ephys_data",
        pose_data_path=repo_root / "examples" / "sample_data" / "pose_data",
        output_root_path=repo_root / "examples" / "output"
    )

    session.run_pose_processing()

    print(session)


if __name__ == "__main__":
    main()