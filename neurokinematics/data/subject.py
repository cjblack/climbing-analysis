from pathlib import Path
import contextlib
import os

from dask import delayed, compute
import yaml

from tqdm import tqdm
from tqdm.dask import TqdmCallback
from tqdm.auto import tqdm as atqdm

from neurokinematics.data.session import ExperimentSession

MANDATORY_KEYS = ['subject_id', 'output_root', 'sessions', 'process']

class ExperimentSubject:

    def __init__(self, subject_specs: dict | str | Path, output_root: str | Path | None = None):
        

        self.subject_specs = self._load_session_specs(subject_specs)

        self.subject_id = self.subject_specs['subject_id']

        if output_root is None:
            self.output_root_path = self.subject_specs['output_root']
        else:
            self.output_root_path = output_root
        
        self.subject_path = Path(self.output_root_path) / self.subject_id
        self.subject_path.mkdir(parents=True, exist_ok=True)

        self.session_processes = self.subject_specs['process']
        self.session_log = self.subject_specs['sessions']

        if self.session_log:
            self.create_sessions_from_log()


    def _load_session_specs(self, session_specs):

        if isinstance(session_specs, (str, Path)):
            session_specs = Path(session_specs)
            if session_specs.suffix in ['.yaml', '.yml']:
                with open(session_specs, "r") as f:
                    session_specs = yaml.safe_load(f)
                    #return yaml.safe_load(f)
            else:
                raise ValueError("session_specs must be .yaml or .yml")
            
        elif isinstance(session_specs, dict):
            pass

        
        else:
            raise ValueError("subject_specs must either be a str, Path, or dict.")
        
        for key in MANDATORY_KEYS:
            if key not in session_specs.keys():
                raise ValueError(f"Missing {key} in session_specs file.")

        return session_specs


    def create_sessions_from_log(self):

        self.sessions = []

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
                
            self.sessions.append(ExperimentSession(session_id = sesh_id, ephys_data_path = ephys_dp, pose_data_path = pose_dp, output_root_path = self.subject_path ))    

    def add_sessions(self):
        # this will be used to add either a single, or multiple sessions to the experiment subject...
        print('do stuff')

    def process_sessions(self):
        pose_proc = self.session_processes['pose']
        spike_proc = self.session_processes['spike']
        lfp_proc = self.session_processes['lfp']

        for session in tqdm(self.sessions, desc="Processing individual session pose data", total=len(self.sessions), unit='sessions'):
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
        with contextlib.redirect_stdout(open(os.devnull,'w')):
            session.run_pose_processing()
