"""Subject management for neurokinematics.

This module provides a hierarchical class that serves to maintain and execute session workflows for an individual subject.

The subject object encapsulates:
    - Subject configuration and metadata
    - Creation and loading of reproducible subject level workflows


"""


from pathlib import Path
import contextlib
import os

from dask import delayed, compute
import yaml

from tqdm import tqdm
from tqdm.dask import TqdmCallback
from tqdm.auto import tqdm as atqdm

from neurokinematics.io import load_yaml, save_yaml

# data
from neurokinematics.data.session import ExperimentSession
from neurokinematics.data.project import NKProject
from neurokinematics.data.utils import IndexedList

MANDATORY_KEYS = ['subject_id', 'output_root', 'process']

class ExperimentSubject:
    """Class for orchestrating multiple session workflows for one subject

    Example:
        >>> subject = ExperimentSubject(
        ...     subject_specs = "path/to/subject/spec.yaml", # see templates/ for example subject_spec files
        ...     project_path = "path/to/project/root", # location for where you want your project to be stored
        ...     name = "project_name" # desired name for project -> defaults to "NK"
        ... )
        >>> subject.process_sessions()
    """

    def __init__(self, subject_specs: dict | str | Path | None, project_path: str | Path | None = None, name: str = 'NK'):
        
        project = NKProject(root = project_path, name = name)
        if subject_specs is not None:
            self.subject_specs = self._load_subject_specs(subject_specs)

            self.subject_id = self.subject_specs['subject_id']
            self.project_name = project.name
            self.subject_specs['project_name'] = self.project_name

            # if output_root is None:
            #     self.output_root_path = self.subject_specs['output_root']
            # else:
            #     self.output_root_path = output_root
            
            self.subject_root = Path(project.subject_root)
            self.subject_path = self.subject_root / self.subject_id #Path(self.output_root_path) / self.subject_id
            self.subject_path.mkdir(parents=True, exist_ok=True)
            

            self.session_processes = self.subject_specs['process']
            self.session_log = self.subject_specs.get('sessions', None)

            if self.session_log:
                self.create_sessions_from_log()
            self._save_subject_specs()
    
    @classmethod
    def from_existing(cls, subject_path: str | Path):
        """Reload previous ExperimentSubject from locally stored spec file.

        Args:
            subject_path (str | Path): Path of initial ExperimentSubject.

        Returns:
            subject (ExperimentSubject): Pre-loaded experiment subject class built from previously saved subject_spec.yaml file.

        Example:
            >>> subject = ExperimentSession.from_existing(subject_path = 'path/to/existing/subject')
        """

        subject_path = Path(subject_path)

        subject = cls(subject_specs = None) # avoid running main calls in __init__
        subject.subject_specs = load_yaml(subject_path / 'subject_spec.yaml')
        subject.subject_id = subject.subject_specs['subject_id']
        subject.subject_path = subject_path
        subject.project_name = subject.subject_specs['project_name']
        subject.subject_root = subject.subject_path.parent
        #subject.output_root_path = subject.subject_specs['output_root']
        subject.session_processes = subject.subject_specs['process']
        subject.session_log = subject.subject_specs['runtime']['sessions']
        subject.session_processes = subject.subject_specs['process']
        
        # reload prior sessions using the ExperimentSession from_existing class method
        subject.sessions = IndexedList([
            ExperimentSession.from_existing(subject_path / s['path']) for s in subject.session_log
        ], id_attr='session_id')

        return subject

    def _load_subject_specs(self, subject_specs: str | Path | dict):
        """Load subject information from spec file

        Args:
            subject_specs (str | Path | dict): Location of .yaml file or corresponding dictionary containing subject spec information. See templates/ for more information

        Raises:
            ValueError: Check that the file type is correct if using a string or Path variable.
            ValueError: Checks that the right input type is used.
            ValueError: Checks that the right keys are available.

        Returns:
            subject_specs (dict): Dictionary of subject information to build sessions.
        """
        if isinstance(subject_specs, (str, Path)):
            subject_specs = Path(subject_specs)
            if subject_specs.suffix in ['.yaml', '.yml']:
                with open(subject_specs, "r") as f:
                    subject_specs = yaml.safe_load(f)
                    #return yaml.safe_load(f)
            else:
                raise ValueError("subject_specs must be .yaml or .yml")
            
        elif isinstance(subject_specs, dict):
            pass

        
        else:
            raise ValueError("subject_specs must either be a str, Path, or dict.")
        
        for key in MANDATORY_KEYS:
            if key not in subject_specs.keys():
                raise ValueError(f"Missing {key} in subject_specs file.")

        return subject_specs
    
    def _save_subject_specs(self):
        """Simple helper function to save the subject spec file for persistence.
        """
        
        self.subject_spec_path = self.subject_path / 'subject_spec.yaml'
        save_yaml(self.subject_specs, self.subject_spec_path)


    def _create_session(self, sesh: dict) -> ExperimentSession:
        ephys_dp = sesh['ephys_data_path']
        pose_dp = sesh['pose_data_path']

        if ephys_dp is None and pose_dp is None:
            raise ValueError("At least one modality must be provided")
        if isinstance(ephys_dp, (str, Path)):
            ephys_dp = Path(ephys_dp)
            if not ephys_dp.exists():
                raise ValueError("Ephys data path does not exist. Please enter valid path.")
        if isinstance(pose_dp, (str, Path)):
            pose_dp = Path(pose_dp)
            if not pose_dp.exists():
                raise ValueError("Pose data path does not exist. Please enter valid path.")
        session = ExperimentSession(
            session_id=sesh['session_id'],
            ephys_data_path=ephys_dp,
            pose_data_path=pose_dp,
            output_root_path=self.subject_path,
            cfg = sesh['session_config']
        )
        self.subject_specs['runtime']['sessions'].append(
            {
                'path': str(session.session_path.relative_to(self.subject_path)),
                'session_id': session.session_id,
                'ephys_data_path': str(session.ephys_data_path) if isinstance(session.ephys_data_path, (str, Path)) else None,
                'pose_data_path': str(session.pose_data_path) if isinstance(session.pose_data_path, (str, Path)) else None,
                'session_config': sesh['session_config']
            }
        )
        return session

    def create_sessions_from_log(self):
        self.sessions = IndexedList(id_attr='session_id')#[]
        self.subject_specs['runtime'] = {'sessions': []}
        for sesh in self.session_log:
            self.sessions.append(self._create_session(sesh))

    def add_sessions(self, sessions):
        if isinstance(sessions, dict):
            sessions = [sessions]
        for s in sessions:
            self.sessions.append(self._create_session(s))
        self._save_subject_specs()

    def process(self, type: str, mode: str = "skip"):
        pose_proc = self.session_processes['pose']
        spike_proc = self.session_processes['spike']
        lfp_proc = self.session_processes['lfp']

        if type == 'pose':
            for session in tqdm(self.sessions, desc=f"Processing {self.subject_id} session pose data", total=len(self.sessions), unit='sessions'):
                if pose_proc and session.pose_data_path:
                    #self._run_pose_processing(session, mode)
                    self._process(session, type, mode)
                else:
                    raise FileExistsError('No pose data available.')
        elif type == 'spikes':
             for session in tqdm(self.sessions, desc=f"Processing {self.subject_id} session spiking data", total=len(self.sessions), unit='sessions'):
                if spike_proc and session.ephys_data_path:
                    #self._run_spike_sorting(session, mode)
                    self._process(session, type, mode)
                else:
                    raise FileExistsError('No ephys data available.')      

        elif type == 'lfp':
              for session in tqdm(self.sessions, desc=f"Processing {self.subject_id} session lfp data", total=len(self.sessions), unit='sessions'):
                if lfp_proc and session.ephys_data_path:
                    #self._run_lfp_processing(session, mode)
                    self._process(session, type, mode)
                else:
                    raise FileExistsError('No ephys data available.')                      
        
    def align(self, type: str, mode: str = 'skip'):

        for session in tqdm(self.sessions, desc=f"Aligning {self.subject_id} {type}s", total=len(self.sessions), unit='sessions'):
            session.align(type, mode)

    def epoch(self, type: str, mode: str = 'skip'):

        for session in tqdm(self.sessions, desc=f"Epoching {self.subject_id} {type}s", total=len(self.sessions), unit='sessions'):
            session.epoch(type, mode)

    def extract(self, type: str, feature: str, mode: str = 'skip'):

        for session in tqdm(self.sessions, desc=f"Extracting {self.subject_id} {type} features", total = len(self.sessions), unit='sessions'):
            session.extract(type, feature, mode)


    def _process(self, session, type: str, mode: str = 'skip'):
        with contextlib.redirect_stdout(open(os.devnull,'w')):
            session.process(type, mode)

    def par_process_sessions(self):
        #parallel processing of sessions - faster depending on data size
        proc_list = []
        for session in self.sessions:
            proc_list.append(delayed(self._run_pose_processing)(session))
        with TqdmCallback(desc="compute"):
            print(compute(*proc_list, scheduler='processes'))


    def _run_pose_processing(self, session, mode: str = "skip"):
        """Helper function to run session pose processing

        Args:
            session (ExperimentSession): Instantiated ExperimentSession class
        """
        with contextlib.redirect_stdout(open(os.devnull,'w')):
            session.run_pose_processing(mode)

    def _run_spike_sorting(self, session, mode: str = "skip"):

        with contextlib.redirect_stdout(open(os.devnull, "w")):
            session.run_spike_sorting(mode)

    def _run_lfp_processing(self, session, mode: str = "skip"):

        with contextlib.redirect_stdout(open(os.devnull, "w")):
            session.run_lfp_processing(mode)
