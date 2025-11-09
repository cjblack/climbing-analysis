from pathlib import Path
from climbing_analysis.decorators import log_call
from climbing_analysis.pose.utils import load_df_list, load_pickle
from climbing_analysis.ephys.spike_sorting import *
from climbing_analysis.ephys.events import get_camera_events

class ClimbingSessionData:
    def __init__(self, session_path):
        self.session_path = Path(session_path)
        self.pose_path = self.session_path / 'PoseData'
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
        self.get_event_data()

    @log_call(label='pose data check')
    def check_pose(self):
        if self.session_path.is_dir():
            self.get_pose_data(self.pose_path)
        else:
            print('no pose data available.')

    @log_call(label='ephys data check')
    def check_ephys(self):
        self.sorting_path = self.session_path.joinpath(self.rec_node, self.experiment_no, self.recording_no, self.sorter_method, 'phy_output')
        if self.sorting_path.exists():
            self.get_spike_data()
            #self.get_lfp_data()

    @log_call(label='pose data', type='load')
    def get_pose_data(self, pose_path):
        # load pose data
        self.pose_df_list = load_df_list(str(pose_path / 'pose.h5')) # convert to string for linux systems
        self.stances = load_pickle(str(pose_path / 'stances.pkl')) # convert to string for linux systems

    @log_call(label='spike data', type='load')
    def get_spike_data(self):
        self.sorter = load_phy_sorting(self.sorting_path)

    @log_call(label='event data', type='load')
    def get_event_data(self):
        prefix = 'Record Node'
        self.has_ephys = any(p.is_dir() and p.name.startswith(prefix) for p in self.session_path.iterdir())
        if self.has_ephys:
            event_data, ts, bouts, frame_captures, continuous = get_camera_events(self.session_path)
            self.frame_captures = frame_captures

    @property
    def unit_ids(self):
        if self.sorter:
            return self.sorter.get_unit_ids()
        else:
            raise AttributeError('sorting object does not exist, please run spike sorting.')
    @property
    def trial_number(self):
        if self.pose_df_list:
            return len(self.pose_df_list)
        else:
            raise AttributeError('pose data does not exist, please add to root folder.')
        