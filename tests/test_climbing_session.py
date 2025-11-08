from pathlib import Path
import pandas as pd
from climbing_analysis.data.session_data import ClimbingSessionData

def test_climbing_session():
    data_path = Path(__file__).resolve().parent.parent / 'climbing_analysis/data/test_sets/example_session_01'
    climbing_session = ClimbingSessionData(data_path)

    assert isinstance(climbing_session.stances, list)
    assert isinstance(climbing_session.pose_df_list[0], pd.DataFrame)
    assert list(climbing_session.pose_df_list[0].columns) == ['r_forepaw_X', 'r_forepaw_Y', 
                                                        'l_forepaw_X', 'l_forepaw_Y',
                                                        'r_hindpaw_X', 'r_hindpaw_Y', 
                                                        'l_hindpaw_X', 'l_hindpaw_Y', 
                                                        'snout_X', 'snout_Y', 
                                                        'tail_X', 'tail_Y']
    assert list(climbing_session.stances[0].keys()) == ['r_forepaw','l_forepaw','r_hindpaw','l_hindpaw']
    assert list(climbing_session.stances[0]['r_forepaw'].keys()) == ['start', 'end', 'max']