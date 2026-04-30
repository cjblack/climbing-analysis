from pathlib import Path
import shutil
import xmltodict
import pandas as pd
import yaml
import dask.dataframe as dd

from neurokinematics.decorators import log_call

# nk io
from neurokinematics.io import create_session_dirs

# pose
from neurokinematics.pose.preprocessing.base import process_sleap
from neurokinematics.pose.io import load_df_list, load_pickle

# ephys
from neurokinematics.ephys.io import *
from neurokinematics.ephys.spikes.sorting import sort
from neurokinematics.ephys.spikes.rasters import get_movement_aligned_rasters
from neurokinematics.ephys.lfp.preprocessing import preprocess_lfp
from neurokinematics.ephys.lfp.epochs import get_movement_aligned_erps

# multimodal
from neurokinematics.multi_modal.alignment import get_camera_events, align_movements_to_ephys



class ExperimentSession:
    """Class for orchestrating preprocessing and alignment of ephys and pose data

    Example:
        >>> session = ExperimentSession(
        ...     session_id = "demo_session",
        ...     ephys_data_path = "path/containing/ephys/data",
        ...     pose_data_path = "path/containing/pose/data",
        ...     output_root_path = "path/to/store/neurokinematics/session"
        ... )
        >>> session.preprocess_and_align()
    """
    def __init__(self, session_id: str, ephys_data_path: Path | str, pose_data_path: Path | str, output_root_path: Path | str | None = None, cfg: str ='demo_session.yaml'):
  
        # set session id
        self.session_id = session_id

        # ensure input paths are Path
        self.ephys_data_path = Path(ephys_data_path)
        self.pose_data_path = Path(pose_data_path)
        
        # then ensure that paths exist...
        if not self.ephys_data_path.exists():
            raise FileNotFoundError(f"Ephys path does not exist: {self.ephys_data_path}")
        if not self.pose_data_path.exists():
            raise FileNotFoundError(f"Pose path does not exist: {self.pose_data_path}")
        
        # load configs
        if cfg is not None:
            self._load_configs(cfg)
            self._set_metadata()
            cfg_output_root_path = self.cfg.get('session', {}).get('output_root', None)
        else:
            self.cfg = {}
            self.metadata = {}
            cfg_output_root_path = None


        # resolve output root
        output_root = output_root_path or cfg_output_root_path
        if output_root is None:
            self.output_root = Path.cwd() / "nk_sessions"
        else:
            self.output_root = Path(output_root)

        # create session directory
        self.session_path = self.output_root / f"{self.session_id}_nk"
        self.dirs = create_session_dirs(self.session_path)

        if cfg is not None:
            self._save_session_config()


    @classmethod
    def from_existing(cls, session_path: Path | str):
        """Loads previously created neurokinematics session

        Args:
            session_path (Path | str): Path to previously created session 

        Returns:
            ExperimentSession: Object replicating the original instatiated session
        """
        session_path = Path(session_path)

        session_config_path = session_path / "session_config.yaml"

        # load previously created config file
        with open(session_config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # get runtime config information
        runtime = cfg['session_runtime']

        # setup session
        session = cls(
            session_id = runtime['session_id'],
            ephys_data_path = runtime['ephys_data_path'],
            pose_data_path = runtime['pose_data_path'],
            output_root_path = runtime['output_root'],
            cfg = None
        )

        # set configs 
        session.cfg = cfg['configs']['session']
        session.pose_cfg = cfg['configs']['pose']
        session.lfp_preprocessing_cfg = cfg['configs']['lfp']
        session.multimodal_cfg = cfg['configs']['multimodal']
        session.sorting_cfg = cfg['configs']['spikes']

        # get metadata
        session.metadata = cfg.get('session_metadata', {})
        if not session.metadata:
            session._set_metadata()

        # set paths
        session.session_path = session_path

        # recreate dirs
        if 'session_dirs' in runtime:
            session.dirs = {key: Path(val) for key, val in runtime['session_dirs'].items()}
        else:
            session.dirs = create_session_dirs(session.session_path)

        return session

    def __str__(self):
        """
        Returns basic details about the session.
        """
        return "".join(

            [
                "\nExperiment Session Object\n",
                f"\n    Directory: {self.session_path}",
                f"\n    Session ID: {self.session_id}"
            ]
        )
    
    def _load_configs(self, cfg):
        """Loads sub configs listed in the main session config file

        Args:
            cfg (str): Session config file
        """

        self.cfg = load_config(cfg, config_type='session')
        cfg_group = self.cfg['configs'] # dict with the names of the sub configs used in the session
        self.sorting_cfg = load_config(cfg_group['spikes'], config_type='spksorting') # spike sorting config
        self.pose_cfg = load_config(cfg_group['pose'], config_type='pose') # pose config
        self.lfp_preprocessing_cfg = load_config(cfg_group['lfp'], config_type='lfp') # lfp preprocessing config
        self.multimodal_cfg = load_config(cfg_group['multi_modal'], config_type='multimodal') # multimodal alignment config

    def _save_session_config(self):
        """Freezes session config so session can be loaded at another time
        """

        cfg = {
            'session_runtime':{
                "session_id": self.session_id,
                "ephys_data_path": str(self.ephys_data_path),
                "pose_data_path": str(self.pose_data_path),
                "output_root": str(self.output_root),
                "session_path": str(self.session_path),
                "session_dirs": {key: str(val) for key, val in self.dirs.items()}
                },
            'configs':{
                'session': self.cfg,
                'pose': self.pose_cfg,
                'spikes': self.sorting_cfg,
                'lfp': self.lfp_preprocessing_cfg,
                'multimodal': self.multimodal_cfg
                }
            }
        

        session_config_path = self.session_path / "session_config.yaml"

        with open(session_config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    
    def _set_metadata(self):
        """Sets metadata for session - will be expanded in future updates
        """

        self.metadata = {
            'ephys': {
                'acquisition': self.cfg['session']['ephys']['acquisition'],
                'sample_rate': self.sorting_cfg['sample_rate'],
                'lfp_node_idx': self.cfg['session']['ephys']['lfp']['node_idx'],
                'lfp_rec_idx': self.cfg['session']['ephys']['lfp']['rec_idx'],
                'spike_sorter': self.sorting_cfg['sorter']
            },
            'pose':{
                'frame_rate': self.pose_cfg['pose_format']['frame_rate'],
                'pose_type': self.pose_cfg['pose_format']['pose_type'],
                'node_list': self.pose_cfg['movement_detection']['node_list']
            }
        }
    
    def _handle_existing_output(self, path: Path, mode: str):
        """Deals with processing/alignment calls if session was already created to avoid accidental overwriting

        Args:
            path (Path): Path to expected data location
            mode (str): Mode to execute, options are 'skip' to skip over function call, 'overwrite' to perform function call, and 'error' to check for accidental overlap

        Raises:
            ValueError: Error if the incorrect mode was selected
            FileExistsError: Error check if the expected output already exists

        Returns:
            bool: Boolean determines whether downstream function call is executed
        """

        if mode not in {"skip", "overwrite", "error"}:
            raise ValueError("Mode must be one of: 'skip', 'overwrite', 'error'")
        
        if not path.exists():
            return True
        
        if mode == "skip":
            return False
        
        if mode == "error":
            raise FileExistsError(f"Output already exists: {path}")
        
        if mode == "overwrite":
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True

    def preprocess_and_align(self):
        """Performs all preprocessing and alignment steps
        """
        self.run_pose_processing() # process pose data
        self.run_spike_sorting() # run spike sorting
        self.run_lfp_processing() # process LFP data
        self.align_video() # align video to ephys
        self.align_movements() # align movement events to ephys

    @log_call(label='pose preprocessing', type='run')
    def run_pose_processing(self, mode: str = "skip"):
        """Run preprocessing on markerless pose data and store results

        Args:
            mode (str, optional): Determines whether to perform processing step. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".

        Returns:
            Path: If processing occured and mode is 'skip', returns path to processed data
        """
        # overwrite utility
        expected_output = self.dirs['pose'] / 'pose_data.csv'

        should_run = self._handle_existing_output(expected_output, mode)

        if not should_run:
            return expected_output
        
        self.pose_processed = process_sleap(
            data_path = self.pose_data_path,
            pose_cfg = self.cfg['configs']['pose'],
            save_path = self.dirs['pose']
        )
    
    @log_call(label='spike sorting', type='run')
    def run_spike_sorting(self, mode: str = "skip"):
        """Run spike sorting through SpikeInterface and store results

        Args:
            mode (str, optional): Determines whether to perform processing step. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".

        Returns:
            Path: If spike sorting occured and mode is 'skip', returns path to sorting results
        """

        
        # overwrite utility
        expected_output = self.dirs['spikes'] / 'sorting_analyzer'

        should_run = self._handle_existing_output(expected_output, mode)

        if not should_run:
            return expected_output

        self.sorter, self.recording, self.probe, self.analyzer = sort(
            data_path = self.ephys_data_path,
            cfg_file = self.cfg['configs']['spikes'],
            save_path = self.dirs['spikes']
        )
    
    @log_call(label='lfp preprocessing', type='run')
    def run_lfp_processing(self, mode: str = "skip"):
        """Run preprocessing on raw LFP data (chunk, filter, downsample) and store results

        Args:
            mode (str, optional): Determines whether to perform processing step. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".

        Returns:
            Path: If processing occured and mode is 'skip', returns path to processed data
        """

        # overwrite utility
        expected_output = self.dirs['lfp'] / 'lfp_preprocessed'

        should_run = self._handle_existing_output(expected_output, mode)

        if not should_run:
            return expected_output

        if self.cfg['session']['ephys']['acquisition'] == 'openephys':
            self.lfp_processed = preprocess_lfp(
                data_path = self.ephys_data_path,
                node_idx = self.metadata['ephys']['lfp_node_idx'],#self.cfg['session']['ephys']['lfp']['node_idx'],
                rec_idx = self.metadata['ephys']['lfp_rec_idx'],#self.cfg['session']['ephys']['lfp']['rec_idx'],
                fs_new = self.lfp_preprocessing_cfg['downsample_rate'],
                chunk_duration_s = self.lfp_preprocessing_cfg['chunking']['chunk_duration_s'],
                pad_duration_s = self.lfp_preprocessing_cfg['chunking']['pad_duration_s'],
                filter_info = {
                    "n_": self.lfp_preprocessing_cfg['filters']["notch"],
                    "bp_": self.lfp_preprocessing_cfg['filters']["bandpass"],
                    "quality": self.lfp_preprocessing_cfg['filters']["quality"]
                },
                dtype = self.lfp_preprocessing_cfg['dtype'],
                save_path = self.dirs['lfp'],
                storage_format = self.lfp_preprocessing_cfg['storage_format']
            )

    def align_video(self, mode:str = "skip"):
        """Align frame captures to ephys data

        Args:
            mode (str, optional): Determines whether to perform alignment. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".

        Returns:
            Path: If alignment exists and mode is 'skip', returns path to alignment
        """

        expected_output = self.dirs['alignment'] / 'video_alignment.csv'

        should_run = self._handle_existing_output(expected_output, mode)

        if not should_run:
            return expected_output

        _, _, _, _, _ = get_camera_events(
            directory = self.ephys_data_path,
            camera_cfg_file = self.cfg['configs']['multi_modal'],
            save_path = self.dirs['alignment']
        )

    def align_movements(self, mode: str = "skip"):
        """Align movements to ephys data

        Args:
            mode (str, optional): Determines whether to perform alignment. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".

        Raises:
            FileNotFoundError: Checks to make sure the necessary files for running alignment have been created

        Returns:
            Path: If alignment exists and mode is 'skip', returns path to alignment
        """

        # set required files for running alignment that should have been created in the session
        required = [
            self.dirs['pose'] / 'movement_events.pkl',
            self.dirs['pose'] / 'pose_data.csv',
            self.dirs['alignment'] / 'video_alignment.csv'
        ]

        # indicate missing files
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot align movements. Missing required files:\n"
                + "\n".join(str(p) for p in missing)
            )
        
        # check whether to skip or overwrite alignment
        expected_output = self.dirs['alignment'] / 'movement_event_alignment.csv'
        should_run = self._handle_existing_output(expected_output, mode)
        if not should_run:
            return expected_output
        
        # set alignment dirs - 'events' dir is redundant, remove in future updates
        dirs_for_alignment = {
            'events': self.dirs['pose'],
            'pose': self.dirs['pose'],
            'alignment': self.dirs['alignment']
        }

        # run alignment
        self.aligned_movements = align_movements_to_ephys(
            dirs = dirs_for_alignment,
            fs = self.ephys_sample_rate, #self.sorting_cfg['sample_rate'],
            fps = self.pose_sample_rate,   #self.pose_cfg['pose_format']['frame_rate'],
            save_path = self.dirs['alignment']
        )
    
    def epoch_lfp(self, mode: str = "skip"):
        """Epoch lfp data with respect to movement events

        Args:
            mode (str, optional): Determines whether to epoch data. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".
        """

        # set required files for running alignment that should have been created in the session
        required = [
            self.dirs['alignment'] / 'movement_event_alignment.csv',
            self.dirs['lfp'] / 'lfp_preprocessed'
        ]

        # indicate missing files
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot epoch LFP. Missing required files:\n"
                + "\n".join(str(p) for p in missing)
            )

        # check whether to skip or overwrite epoching
        expected_output = self.dirs['lfp'] / 'lfp_epoched'
        should_run = self._handle_existing_output(expected_output, mode)
        if not should_run:
            return expected_output
        
        self.epoch_lfp_root = get_movement_aligned_erps(
            alignment = self.dirs['alignment'] / 'movement_event_alignment.csv',
            lfp_data = self.dirs['lfp'] / 'lfp_preprocessed',
            save_path = self.dirs['lfp'] / 'lfp_epoched'
        )

    def epoch_spikes(self, mode: str = "skip"):
        """Epoch spike rasters with respect to movement events

        Args:
            mode (str, optional): Determines whether to epoch data. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".
        """

        # set required files for running alignment that should have been created in the session
        required = [
            self.dirs['alignment'] / 'movement_event_alignment.csv',
            self.dir['spikes'] / self.spike_sorting_method
        ]

        # indicate missing files
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot epoch LFP. Missing required files:\n"
                + "\n".join(str(p) for p in missing)
            )

        # check whether to skip or overwrite epoching
        expected_output = self.dirs['spikes'] / 'rasters' / 'movement_aligned_rasters.pkl'
        should_run = self._handle_existing_output(expected_output, mode)
        if not should_run:
            return expected_output
        
        self.sorter = load_phy_sorting(self.dir['spikes'] / self.spike_sorting_method)
        self.spike_raster_obj = get_movement_aligned_rasters(
            alignment = self.dirs['alignment'] / 'movement_event_alignment.csv',
            sorter = self.sorter,
            save_path = self.dirs['spikes']
        )
    
    @property
    def ephys_sample_rate(self):
        """ Returns sample rate of ephys acquisition
        """
        return self.metadata['ephys']['sample_rate']
    
    @property
    def pose_sample_rate(self):
        """ Returns frame rate of original pose data
        """
        return self.metadata['pose']['frame_rate']
    
    @property
    def acquisition_system(self):
        """ Returns name of ephys acquisition system
        """
        return self.metadata['ephys']['acquisition']
    
    @property
    def pose_package(self):
        """ Returns name of package used for pose data 
        """
        return self.metadata['pose']['pose_type']
    
    @property
    def spike_sorter(self):
        """ Returns name of spike sorter used in session
        """
        return self.metadata['spikes']['spike_sorter']

# class ClimbingSession:
#     """
#     Class for loading in all relevant pose and ephys data during climbing session.
#     Requires processing of video data for pose estimation to be stored within data directory in 'PoseData' folder.
    
#     Use:
#     from neurokinematics.data.session_data import ClimbingSessionData
#     csession = ClimbingSessionData('path/to/data/directory')
#     """
#     def __init__(self, session_path, cfg='climbing_sorting_cfg.yaml'):
#         self.session_path = Path(session_path)
#         self.pose_path = self.session_path / 'pose'
#         self.sorting_cfg = get_sorting_cfg(cfg) # refactor this to config...
#         self.ephys_data = None
#         self.pose_data = None
#         self.analyzer = None
#         self.metadata = {}

#         # For running/loading sorting
#         self.rec_node = 'Record Node 109'
#         self.experiment_no = 'experiment1'
#         self.recording_no = 'recording1'
#         self.sorter_method = 'kilosort4'

#         # For data initialization
#         self.lfp_recording_loaded = False

#         # Run data checks and get events
#         self.check_pose()
#         self.check_ephys()
#         self.get_event_data()
    
#     def __str__(self):
#         """
#         Returns details about the climbing session.
#         """
#         return "".join(

#             [
#                 "\nClimbing Session Object\n",
#                 f"\n    Directory: {self.session_path}",
#                 f"\n    Trial count: {self.trial_number}",
#                 f"\n    Node count: {self.node_number}",
#                 f"\n    Recording information:",
#                 f"\n        {self.probe}",
#                 f"\n        {self.recording}"
#             ]
#         )

#     @log_call(label='pose data check')
#     def check_pose(self):
#         """
#         Checks that pose data exists, then loads
#         """
#         if self.session_path.is_dir():
#             self.get_pose_data(self.pose_path)
#         else:
#             print('no pose data available.')

#     @log_call(label='ephys data check')
#     def check_ephys(self):
#         """
#         Checks that ephys data exists, then loads
#         """
#         self.sorting_path = self.session_path.joinpath(self.rec_node, self.experiment_no, self.recording_no, self.sorter_method, 'phy_output')
#         if self.sorting_path.exists():
#             self.get_spike_data()
#             self.get_recording()
#             #self.get_lfp_data()

#     @log_call(label='pose data', type='load')
#     def get_pose_data(self, pose_path):
#         """
#         Gets pose data
#         stores:
#             self.pose_df_list: list of data frames containing x,y coordinates of each ROI for each trial
#             self.stances: dictionary of start, end, and maximum velocity time points for reaching events of fore and hind paws
#         """
#         # load pose data
#         self.pose_df_list = load_df_list(str(pose_path / 'pose.h5')) # convert to string for linux systems
#         self.stances = load_pickle(str(pose_path / 'stances.pkl')) # convert to string for linux systems

#     @log_call(label='spike data', type='load')
#     def get_spike_data(self):
#         """
#         Gets sorted spike data
#         stores:
#             self.sorter: spikeinterface sorting object, containing spike sorted data
#         """
#         self.sorter = load_phy_sorting(self.sorting_path) # return sorting object
#         self.cluster_df = pd.read_csv(self.sorting_path / 'cluster_group.tsv',sep='\t') # this will return the labeled clusters from phy2 GUI

#     @log_call(label='lfp data', type='load')
#     def get_lfp_samples(self, start_sample_index, end_sample_index):
#         """
#         Gets LFP data (currently stored in a separate recording node that is hardcoded)
#         stores:
#             self.lfp: open ephys object, containing lfp data
#         """
#         self.lfp = get_lfp(self.session_path)
        
#         #self.lfp_samples = self.lfp.get_samples(start_sample_index=0, end_sample_index=-1)
#         #self.lfp_shape = self.lfp.samples.shape
#         #self.lfp_recording_loaded = True
#         return self.lfp.get_samples(start_sample_index=start_sample_index, end_sample_index=end_sample_index)
    
#     @log_call(label='preprocess lfp', type='process')
#     def _preprocess_lfp(self):
#         "Chunks, filters, and downsamples the data, storing as a memmap."
#         preprocess_lfp(self.session_path)
        
#     @log_call(label='channel map', type='load')
#     def get_acquisition_channel_map(self, chan_count=64):
#         """
#         Gets channel mapping if stored in acquisition settings *CURRENTLY FOR OPEN EPHYS ACQUISTION ONLY*
#         """
        
#         self.acquisition_channel_map = np.zeros((chan_count,1))
#         xml_loc = self.session_path / 'Record Node 9/settings.xml'
#         with open(xml_loc, 'r') as f:
#             xml_data = xmltodict.parse(f.read())
#         processors = xml_data['SETTINGS']['SIGNALCHAIN']['PROCESSOR']
#         for proc in processors:
#             if proc['@name'] == 'Channel Map':
#                 chan_map = proc['CUSTOM_PARAMETERS']['STREAM'][0]['CH']
#                 for i in range(chan_count):
#                     self.acquisition_channel_map[i,0]=int(chan_map[i]['@index'])


#     @log_call(label='recording', type='load')
#     def get_recording(self):
#         """
#         Gets ephys recording
#         stores:
#             self.recording: spikeinterface recording object containing raw data traces, with relevant probe information
#         """
#         self.recording = read_data(data_path=str(self.session_path), rec_type=self.sorting_cfg['rec_type'])
#         self.probe = create_probe(self.sorting_cfg['probe_manufacturer'],self.sorting_cfg['probe_id'], self.sorting_cfg['channel_map'])
#         self.recording = self.recording.set_probe(self.probe) #,group_mode='by_shank' # have to rename to set probe


#     @log_call(label='event data', type='load')
#     def get_event_data(self):
#         """
#         Gets camera event data from analog channel - currently hard coded to specific channel
#         stores:
#             self.frame_captures: list containing lists of sample indices for each captured frame for each trial
#         """
#         prefix = 'Record Node'
#         self.has_ephys = any(p.is_dir() and p.name.startswith(prefix) for p in self.session_path.iterdir())
#         if self.has_ephys:
#             df_fc = dd.read_csv(self.session_path / 'events' / 'video_alignment.csv')
#             self.frame_captures = [df_fc[df_fc['video_index']==i+1].compute()['sample_index'].values for i in np.arange(0,df_fc['video_index'].unique().compute().__len__(),1)] # this is a very inefficient and dirty load scheme
#             #self.frame_captures = pd.read_csv(self.session_path / 'events' / 'video_alignment.csv')

#     @log_call(label='analyzer', type='load')
#     def get_analyzer(self):
#         """
#         Gets the analyzer data:
#             self.analyzer: stores the spikeinterface analyzer object, use this for plotting spike waveforms
#         """
#         analyzer_path = self.sorting_path.resolve().parent.parent / 'sorting_analyzer'
#         if analyzer_path.is_dir():
#             self.analyzer = load_analyzer(analyzer_path)
    
#     @log_call(label='waveforms', type='load')
#     def get_waveforms(self):
#         if self.analyzer:
#             self.waveforms = self.analyzer.get_extension('waveforms')
        


#     @log_call(label='psth', type='plot')
#     def plot_psth(self, unit_id, node='r_hindpaw', epoch_loc='start', ylim_=[0,100],save_fig=False):
#         """
#         Plot peri-stimulus time histograms from the climbing session for different nodes with respect to different movements
#         inputs:
#             node: str -> dependent on self.pose_df_list nodes, current choices are: 'r_hindpaw', 'r_forepaw', 'l_hindpaw', 'l_forepaw'
#             epoch_loc: str -> 'start', 'end', 'max'; to align data to start of movement, end of movement, or maximum velocity of movement
#             save_fig: bool | str -> False if no plot, directory string if you want to store plot. Will update in the future
#         """
#         if (unit_id in self.unit_ids) and (self.sorter) and (self.pose_df_list) and (self.stances) and (self.frame_captures):
#             spikes, kinematics, mirror_kinematics = plot_session_psth([unit_id],self.sorter, self.pose_df_list, self.frame_captures, self.stances, node=node, epoch_loc=epoch_loc, ylim_=ylim_,save_fig=save_fig)
#         else:
#             print('not all data loaded.')
    
#     @log_call(label='morlet spectrogram', type='plot')
#     def plot_morlet_spectrogram(self, channel, node='r_hindpaw', epoch_loc='start', freqs=np.arange(2,40,1), n_cycles=None, xlim_=[-0.5,0.5],save_fig=False):
#         """
#         Plots spectrogram for given channel using morlet wavelet from MNE package.
#         """
#         #if self.lfp_recording_loaded == False:
#         #    self.get_lfp_data()
        
#         #lfp_chan = self.lfp.get_samples(start_sample_index=0,end_sample_index=self.lfp_shape[0],selected_channels=[channel])
#         #power_z = morlet_lfp(lfp_chan[:,0],self.pose_df_list,self.frame_captures,self.stances,node=node,epoch_loc=epoch_loc,freqs=freqs, n_cycles=n_cycles, xlim_=xlim_, save_fig=save_fig)
#         power_z = 0
#         return power_z
    
#     @property
#     def unit_ids(self):
#         """
#         Property storing a list of unit ids from the sorting object
#         """
#         if self.sorter:
#             return self.sorter.get_unit_ids()
#         else:
#             raise AttributeError('sorting object does not exist, please run spike sorting.')
    
#     @property
#     def good_units(self):
#         """
#         Property storing the 'good' units from the cluster groups. This will only work if data has been labeled
#         """
#         if self.cluster_df is not None:
#             return self.cluster_df.loc[self.cluster_df['group']=='good']['cluster_id'].to_numpy()
#         else:
#             raise AttributeError('cluster data frame does not exist, please run spike sorting.')
    
#     @property
#     def good_unit_idx(self):
#         """
#         Property storing the 'good' unit indices from the cluster groups, which is what is needed for plotting.
#         """
#         if self.cluster_df is not None:
#             return self.cluster_df[self.cluster_df['group']=='good'].index.to_list()
#         else:
#             raise AttributeError('cluster data frame does not exist, please run spike sorting.')
        
#     @property
#     def trial_number(self):
#         """
#         Property storing number of trials for the session, extracted from the length of the self.pose_df_list.
#         """
#         if self.pose_df_list:
#             return len(self.pose_df_list)
#         else:
#             raise AttributeError('pose data does not exist, please add to root folder.')
        
#     @property
#     def node_number(self):
#         """
#         Property storing number of nodes for the session, computed as 0.5*length of the number of keys in the pose df list (x and y coordinates).
#         """
#         if self.pose_df_list:
#             return int(self.pose_df_list[0].shape[1]/2)
#         else:
#             raise AttributeError('pose data does not exist, please add to root folder.')