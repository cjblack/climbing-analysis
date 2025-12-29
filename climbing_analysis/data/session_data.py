from pathlib import Path
import xmltodict
import pandas as pd
from climbing_analysis.decorators import log_call
from climbing_analysis.pose.utils import load_df_list, load_pickle
from climbing_analysis.ephys.utils import *
from climbing_analysis.ephys.spike_sorting import *
from climbing_analysis.ephys.lfp import morlet_lfp
from climbing_analysis.ephys.events import get_camera_events

class ClimbingSessionData:
    """
    Class for loading in all relevant pose and ephys data during climbing session.
    Requires processing of video data for pose estimation to be stored within data directory in 'PoseData' folder.
    
    Use:
    from climbing_analysis.data.session_data import ClimbingSessionData
    csession = ClimbingSessionData('path/to/data/directory')
    """
    def __init__(self, session_path, params='climbing_sorting_params.yaml'):
        self.session_path = Path(session_path)
        self.pose_path = self.session_path / 'PoseData'
        self.sorting_params = get_sorting_params(params)
        self.ephys_data = None
        self.pose_data = None
        self.analyzer = None
        self.metadata = {}

        # For running/loading sorting
        self.rec_node = 'Record Node 109'
        self.experiment_no = 'experiment1'
        self.recording_no = 'recording1'
        self.sorter_method = 'kilosort4'

        # For data initialization
        self.lfp_recording_loaded = False

        # Run data checks and get events
        self.check_pose()
        self.check_ephys()
        self.get_event_data()

    @log_call(label='pose data check')
    def check_pose(self):
        """
        Checks that pose data exists, then loads
        """
        if self.session_path.is_dir():
            self.get_pose_data(self.pose_path)
        else:
            print('no pose data available.')

    @log_call(label='ephys data check')
    def check_ephys(self):
        """
        Checks that ephys data exists, then loads
        """
        self.sorting_path = self.session_path.joinpath(self.rec_node, self.experiment_no, self.recording_no, self.sorter_method, 'phy_output')
        if self.sorting_path.exists():
            self.get_spike_data()
            self.get_recording()
            #self.get_lfp_data()

    @log_call(label='pose data', type='load')
    def get_pose_data(self, pose_path):
        """
        Gets pose data
        stores:
            self.pose_df_list: list of data frames containing x,y coordinates of each ROI for each trial
            self.stances: dictionary of start, end, and maximum velocity time points for reaching events of fore and hind paws
        """
        # load pose data
        self.pose_df_list = load_df_list(str(pose_path / 'pose.h5')) # convert to string for linux systems
        self.stances = load_pickle(str(pose_path / 'stances.pkl')) # convert to string for linux systems

    @log_call(label='spike data', type='load')
    def get_spike_data(self):
        """
        Gets sorted spike data
        stores:
            self.sorter: spikeinterface sorting object, containing spike sorted data
        """
        self.sorter = load_phy_sorting(self.sorting_path) # return sorting object
        self.cluster_df = pd.read_csv(self.sorting_path / 'cluster_group.tsv',sep='\t') # this will return the labeled clusters from phy2 GUI

    @log_call(label='lfp data', type='load')
    def get_lfp_data(self):
        """
        Gets LFP data (currently stored in a separate recording node that is hardcoded)
        stores:
            self.lfp: open ephys object, containing lfp data
        """
        self.lfp = get_lfp(self.session_path)
        self.lfp_shape = self.lfp.samples.shape
        self.lfp_recording_loaded = True
    
    @log_call(label='channel map', type='load')
    def get_acquisition_channel_map(self, chan_count=64):
        """
        Gets channel mapping if stored in acquisition settings *CURRENTLY FOR OPEN EPHYS ACQUISTION ONLY*
        """
        
        self.acquisition_channel_map = np.zeros((chan_count,1))
        xml_loc = self.session_path / 'Record Node 9/settings.xml'
        with open(xml_loc, 'r') as f:
            xml_data = xmltodict.parse(f.read())
        processors = xml_data['SETTINGS']['SIGNALCHAIN']['PROCESSOR']
        for proc in processors:
            if proc['@name'] == 'Channel Map':
                chan_map = proc['CUSTOM_PARAMETERS']['STREAM'][0]['CH']
                for i in range(chan_count):
                    self.acquisition_channel_map[i,0]=int(chan_map[i]['@index'])


    @log_call(label='recording', type='load')
    def get_recording(self):
        """
        Gets ephys recording
        stores:
            self.recording: spikeinterface recording object containing raw data traces, with relevant probe information
        """
        self.recording = read_data(data_path=str(self.session_path), rec_type=self.sorting_params['rec_type'])
        self.probe = create_probe(self.sorting_params['probe_manufacturer'],self.sorting_params['probe_id'], self.sorting_params['channel_map'])
        self.recording = self.recording.set_probe(self.probe,group_mode='by_shank') # have to rename to set probe


    @log_call(label='event data', type='load')
    def get_event_data(self):
        """
        Gets camera event data from analog channel - currently hard coded to specific channel
        stores:
            self.frame_captures: list containing lists of sample indices for each captured frame for each trial
        """
        prefix = 'Record Node'
        self.has_ephys = any(p.is_dir() and p.name.startswith(prefix) for p in self.session_path.iterdir())
        if self.has_ephys:
            event_data, ts, bouts, frame_captures, continuous = get_camera_events(self.session_path)
            self.frame_captures = frame_captures

    @log_call(label='analyzer', type='load')
    def get_analyzer(self):
        """
        Gets the analyzer data:
            self.analyzer: stores the spikeinterface analyzer object, use this for plotting spike waveforms
        """
        analyzer_path = self.sorting_path.resolve().parent.parent / 'analyzer_folder'
        if analyzer_path.is_dir():
            self.analyzer = load_analyzer(analyzer_path)
    
    @log_call(label='waveforms', type='load')
    def get_waveforms(self):
        if self.analyzer:
            self.waveforms = self.analyzer.get_extension('waveforms')
        


    @log_call(label='psth', type='plot')
    def plot_psth(self, unit_id, node='r_hindpaw', epoch_loc='start', ylim_=[0,100],save_fig=False):
        """
        Plot peri-stimulus time histograms from the climbing session for different nodes with respect to different movements
        inputs:
            node: str -> dependent on self.pose_df_list nodes, current choices are: 'r_hindpaw', 'r_forepaw', 'l_hindpaw', 'l_forepaw'
            epoch_loc: str -> 'start', 'end', 'max'; to align data to start of movement, end of movement, or maximum velocity of movement
            save_fig: bool | str -> False if no plot, directory string if you want to store plot. Will update in the future
        """
        if (unit_id in self.unit_ids) and (self.sorter) and (self.pose_df_list) and (self.stances) and (self.frame_captures):
            spikes, kinematics, mirror_kinematics = plot_session_psth([unit_id],self.sorter, self.pose_df_list, self.frame_captures, self.stances, node=node, epoch_loc=epoch_loc, ylim_=ylim_,save_fig=save_fig)
        else:
            print('not all data loaded.')
    
    @log_call(label='morlet spectrogram', type='plot')
    def plot_morlet_spectrogram(self, channel, node='r_hindpaw', epoch_loc='start', freqs=np.arange(2,40,1), n_cycles=None, xlim_=[-0.5,0.5],save_fig=False):
        """
        Plots spectrogram for given channel using morlet wavelet from MNE package.
        """
        if self.lfp_recording_loaded == False:
            self.get_lfp_data()
        
        lfp_chan = self.lfp.get_samples(start_sample_index=0,end_sample_index=self.lfp_shape[0],selected_channels=[channel])
        power_z = morlet_lfp(lfp_chan[:,0],self.pose_df_list,self.frame_captures,self.stances,node=node,epoch_loc=epoch_loc,freqs=freqs, n_cycles=n_cycles, xlim_=xlim_, save_fig=save_fig)
        return power_z
    
    @property
    def unit_ids(self):
        """
        Property storing a list of unit ids from the sorting object
        """
        if self.sorter:
            return self.sorter.get_unit_ids()
        else:
            raise AttributeError('sorting object does not exist, please run spike sorting.')
    
    @property
    def good_units(self):
        """
        Property storing the 'good' units from the cluster groups. This will only work if data has been labeled
        """
        if self.cluster_df is not None:
            return self.cluster_df.loc[self.cluster_df['group']=='good']['cluster_id'].to_numpy()
        else:
            raise AttributeError('cluster data frame does not exist, please run spike sorting.')
    
    @property
    def good_unit_idx(self):
        """
        Property storing the 'good' unit indices from the cluster groups, which is what is needed for plotting.
        """
        if self.cluster_df is not None:
            return self.cluster_df[self.cluster_df['group']=='good'].index.to_list()
        else:
            raise AttributeError('cluster data frame does not exist, please run spike sorting.')
        
    @property
    def trial_number(self):
        """
        Property storing number of trials for the session, extracted from the length of the self.pose_df_list
        """
        if self.pose_df_list:
            return len(self.pose_df_list)
        else:
            raise AttributeError('pose data does not exist, please add to root folder.')