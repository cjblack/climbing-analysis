""" 
Example: run a neurokinematics session pipeline

This script demonstrates how to create an ExperimentSession, run selected processing steps.

Before running:
    1. Create and activate conda environment
    2. Ensure all sample data from (https://doi.org/10.17605/OSF.IO/3SR67) exists in examples/sample_data/

"""

from pathlib import Path

from neurokinematics.data.session import ExperimentSession

def main():
    repo_root = Path(__file__).resolve().parents[1]

    ephys_data_path = repo_root / 'examples' / 'sample_data' / 'ephys_data'
    pose_data_path = repo_root / 'examples' / 'sample_data' / 'pose_data'
    output_root_path = repo_root / 'examples' / 'output'

    session = ExperimentSession(
        session_id = "demo_session",
        ephys_data_path = ephys_data_path,
        pose_data_path = pose_data_path,
        output_root_path = output_root_path,
        cfg = 'demo_session.yaml'
    )

    print(session)

    session.run_pose_processing()
    session.align_video()
    session.align_movements()

    print(f"Session saved to: {session.session_path}")


if __name__ == "__main__":
    main()