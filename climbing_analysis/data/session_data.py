from pathlib import Path
from climbing_analysis.decorators import log_call
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

    @log_call(label='pose data check')
    def check_pose(self):
        for child in self.session_path.iterdir():
            if child.name == 'PoseData':
                child_ = Path(child)
                if child_.is_dir():
                    self.pose_path = child_
                    self.get_pose_data(self.pose_path)
                else:
                    print('no pose data available.')

    @log_call(label='ephys data check')
    def check_ephys(self):
        self.sorting_path = self.session_path.joinpath(self.rec_node, self.experiment_no, self.recording_no, self.sorter_method, 'phy_output')
        if self.sorting_path.exists():
            self.get_ephys_data()

    @log_call(label='pose data', type='load')
    def get_pose_data(self, pose_path):
        # load pose data
        self.pose_df_list = load_df_list(pose_path / 'pose.h5')
        self.stances = load_pickle(pose_path / 'stances.pkl')

    @log_call(label='ephys data', type='load')
    def get_ephys_data(self):
        self.sorter = load_phy_sorting(self.sorting_path)
