from pathlib import Path
from climbing_analysis.data.session_data import ClimbingSessionData

def test_climbing_session():
    data_path = Path(__file__).resolve().parent.parent / 'climbing_analysis/data/test_sets/example_session_01'
    