from pathlib import Path
from climbing_analysis.pose.utils import load_df_list, load_pickle
from climbing_analysis.ephys.spike_sorting import *

class ClimbingSessionData:
    def __init__(self, session_path):
        self.session_path = Path(session_path)
        self.ephys_data = None
        self.pose_data = None
        self.metadata = {}

        # For running/loading sorting
        self.rec_node = 'Record Node 109'
        self.experiment_no = 'experiment1'
        self.recording_no = 'recording1'
        self.sorter_method = 'kilosort4'

        self.check_pose()
        self.check_ephys()

    def check_pose(self):
        print('checking pose data...')
        for child in self.session_path.iterdir():
            if child.name == 'PoseData':
                child_ = Path(child)
                if child_.is_dir():
                    self.pose_path = child_
                    self.get_pose_data(self.pose_path)
                else:
                    print('no pose data available.')

    def check_ephys(self):
        print('checking ephys data...')
        self.sorting_path = self.session_path.joinpath(self.rec_node, self.experiment_no, self.recording_no, self.sorter_method, 'phy_output')
        if self.sorting_path.exists():
            self.get_ephys_data()

    def get_pose_data(self, pose_path):
        print('loading pose data...')
        # load pose data
        self.pose_df_list = load_df_list(pose_path / 'pose.h5')
        self.stances = load_pickle(pose_path / 'stances.pkl')

    def get_ephys_data(self):
        print('loading ephys data...')
        self.sorter = load_phy_sorting(self.sorting_path)
