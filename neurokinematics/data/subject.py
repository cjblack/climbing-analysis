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
from neurokinematics.data.session import ExperimentSession
from neurokinematics.data.project import NKProject

MANDATORY_KEYS = ['subject_id', 'output_root', 'sessions', 'process']

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
            self.session_log = self.subject_specs['sessions']

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
        subject.sessions = [
            ExperimentSession.from_existing(subject_path / s['path']) for s in subject.session_log
        ]

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


    def create_sessions_from_log(self):

        self.sessions = []
        self.subject_specs['runtime'] = {'sessions': []}
        for sesh in self.session_log:
            sesh_id = sesh['session_id']
            ephys_dp = sesh['ephys_data_path']
            pose_dp = sesh['pose_data_path']
            if ephys_dp is None and pose_dp is None:
                raise ValueError("At least one modailty must be provided")
            if isinstance(ephys_dp, (str, Path)):
                ephys_dp = Path(ephys_dp)
                if not ephys_dp.exists():
                    raise ValueError("Ephys data path does not exist. Please enter valid path.")
            if isinstance(pose_dp, (str, Path)):
                pose_dp = Path(pose_dp)
                if not pose_dp.exists():
                    raise ValueError("Pose data path does not exist. Please enter valid path.")
            session = ExperimentSession(session_id = sesh_id, ephys_data_path = ephys_dp, pose_data_path = pose_dp, output_root_path = self.subject_path )
            self.sessions.append(session)
            self.subject_specs['runtime']['sessions'].append(
                {
                    'path': str(session.session_path.relative_to(self.subject_path)), 
                    'session_id':session.session_id, 
                    'ephys_data_path': str(session.ephys_data_path) if isinstance(session.ephys_data_path, (str, Path)) else None,
                    'pose_data_path': str(session.pose_data_path) if isinstance(session.pose_data_path, (str, Path)) else None
                    }
                )    

    def add_sessions(self):
        # this will be used to add either a single, or multiple sessions to the experiment subject...
        print('do stuff')

    def process_sessions(self):
        pose_proc = self.session_processes['pose']
        spike_proc = self.session_processes['spike']
        lfp_proc = self.session_processes['lfp']

        for session in tqdm(self.sessions, desc=f"Processing {self.subject_id} session pose data", total=len(self.sessions), unit='sessions'):
            if pose_proc:
                self._run_pose_processing(session)

    def par_process_sessions(self):
        #parallel processing of sessions - faster depending on data size
        proc_list = []
        for session in self.sessions:
            proc_list.append(delayed(self._run_pose_processing)(session))
        with TqdmCallback(desc="compute"):
            print(compute(*proc_list, scheduler='processes'))


    def _run_pose_processing(self, session):
        """Helper function to run session pose processing

        Args:
            session (ExperimentSession): Instantiated ExperimentSession class
        """
        with contextlib.redirect_stdout(open(os.devnull,'w')):
            session.run_pose_processing()
