from pathlib import Path
import yaml

import pandas as pd

from tqdm import tqdm
from tqdm.dask import TqdmCallback
from tqdm.auto import tqdm as atqdm

# data
from neurokinematics.data.subject import ExperimentSubject
from neurokinematics.data.project import NKProject
from neurokinematics.data.utils import IndexedList

# io
from neurokinematics.io import load_yaml, save_yaml, load_zarr, load_parquet

# pose
from neurokinematics.pose.utils import pixels_to_cm
from neurokinematics.pose.features import velocity_summary, acceleration_summary, extract_metadata

# stats
from neurokinematics.stats import get_model as get_stats_model

MANDATORY_GROUP_SPEC_KEYS = ['group_id', 'subjects']

class ExperimentGroup:
    """Class for orchestrating group workflows for multiple subjects.

    Example:
        >>> group = ExperimentGroup(
        ...     group_specs = "path/to/group_specs.yaml", # see templates/ for example group_spec file
        ...     project_path = "path/to/project/root", # location for where you want your project to be stored
        ...     name = "project_name" # desired name for project -> defaults to "NK"
        ... )
        >>> group.process_subjects()
    """

    def __init__(self, group_specs: str | Path | dict | None, project_path: str | Path | None = None, name: str = 'NK'):

        project = NKProject(root = project_path, name = name)
        if group_specs is not None:
            self.project_path = project.root#Path(project_path)
            self.project_name = project.name
            self.group_specs = self._load_group_specs(group_specs)
            self.group_specs['project_name'] = self.project_name
            self.group_id = self.group_specs['group_id']
            self.output_root = Path(self.group_specs['output_root'])
            self.subjects_log = self.group_specs['subjects']

            #self.group_path = self.output_root / self.group_id
            self.group_path = project.group_root / self.group_id
            self.subject_root = project.subject_root
            self.group_path.mkdir(parents=True, exist_ok=True)
            
            self.create_subjects_from_log()
            self._init_directory_structure()
            self._save_group_specs()


    @classmethod
    def from_existing(cls, group_path: Path | str):
        group_path = Path(group_path)
        group_spec_path = group_path / "group_spec.yaml"

        group = cls(group_specs = None)
        group.group_path = group_path
        group.group_specs = load_yaml(group_spec_path)
        group.group_id = group.group_specs['group_id']
        group.dirs = {k: group_path / v for k, v in group.group_specs['folders'].items()}
        group.project_path = group.group_path.parent.parent
        group.subject_root = group.project_path / "Subjects" 
        group.subjects_log = group.group_specs['runtime']['subjects']
        
        #group.group_path = group.output_root / group.group_id

        group.subjects = IndexedList([
            ExperimentSubject.from_existing(group.subject_root / s['spec']) for s in group.subjects_log
        ], id_attr='subject_id')

        return group
    
    @classmethod
    def from_subject_ids(cls, group_id: str, subject_ids: list, project_path: str | Path):
        project_root = Path(project_path)
        group = cls(group_specs = None)
        group.group_id = group_id
        group.project_path = project_root
        group.project_name = project_root.name
        group.subject_root = project_root / 'Subjects'
        group.group_path = project_root / 'Groups' / group_id
        group.group_path.mkdir(parents=True, exist_ok=True)

        group.subjects = IndexedList(
            [ExperimentSubject.from_existing(group.subject_root / sid) for sid in subject_ids],
            id_attr='subject_id'
        )
        
        group.group_specs = {
            'group_id': group_id,
            'project_name': group.project_name,
            'runtime': {
                'subjects': [{'spec': str(Path(sid))} for sid in subject_ids]
            }
        }

        group._init_directory_structure()
        group._save_group_specs()

        return group
    
    def _init_directory_structure(self):
        
        self.group_specs['folders'] = dict()

        dirs = {
            'summaries': self.group_path / 'summaries',
            'stats': self.group_path / 'stats',
            'results': self.group_path / 'results'
        }

        for name, folder in dirs.items():
            folder.mkdir(parents=True, exist_ok=True)
            self.group_specs['folders'][name] = str(folder.relative_to(self.group_path))
        self.dirs = dirs

    def _load_group_specs(self, group_specs):
        if isinstance(group_specs, (str, Path)):
            group_specs = Path(group_specs)
            group_specs = load_yaml(group_specs)
        elif isinstance(group_specs, dict):
            pass
        else:
            raise ValueError("group specs must be of type str, Path, or dict.")
        
        for key in MANDATORY_GROUP_SPEC_KEYS:
            if key not in group_specs.keys():
                raise ValueError(f"{key} key is missing from group_specs.")

        return group_specs   

    def _save_group_specs(self):

        save_yaml(self.group_specs, self.group_path / 'group_spec.yaml')

    def create_subjects_from_log(self):
        self.subjects = IndexedList(id_attr='subject_id')#[]
        self.group_specs['runtime'] = {'subjects': []}

        for subj in self.subjects_log:
            s = ExperimentSubject(subject_specs = subj['spec'], project_path = self.project_path.parent, name = self.project_name)
            self.subjects.append(s)
            self.group_specs['runtime']['subjects'].append({'spec': str(s.subject_spec_path.relative_to(self.subject_root).parent)})

    def process(self, type: str, mode: str = 'skip', method: str = 'sequential'):
        if method == 'sequential':
            for subj in self.subjects:
                subj.process(type, mode)
        elif method == 'parallel':
            for subject in self.subjects:
                subject.par_process_sessions()
        else:
            raise ValueError("Incorrect method selected. Please use either 'sequential' for sequential processing, or 'parallel' for parallel processing.")

    def align(self, type: str, mode: str = "skip"):
        for subj in self.subjects:
            subj.align(type, mode)

    def epoch(self, type: str, mode: str = "skip"):
        for subj in self.subjects:
            subj.epoch(type, mode)
    
    def add_subjects(self, subjects: dict):
        for subj in subjects:
            s = ExperimentSubject(subject_specs=subj['spec'], project_path = self.project_path.parent, name = self.project_name)
            self.subjects.append(s)
            self.group_specs['runtime']['subjects'].append({'spec': str(s.subject_spec_path.relative_to(self.subject_root).parent)})
        self._save_group_specs()

    def summarize(self, type: str):
        """High-level function to make creating summaries easier.

        Args:
            type (str): Name of the summary to create
        """
        if type == 'pose':
            self.summarize_pose()

        if type == 'spikes':
            pass
        
        if type == 'lfp':
            pass


    def load_summary(self, type: str):
        """Lazy load summary data to group.

        Args:
            type (str): Name of the summary to load as a dask dataframe
        """
        if type == 'pose':
            self.pose_summary = load_parquet(self.dirs['summaries'] / 'pose_metrics.parquet', method='dask')
        if type == 'spikes':
            pass
        if type == 'lfp':
            pass


    def summarize_pose(self):
        # start by just collecting velocity data...

        rows = []
        for subject in tqdm(self.subjects, desc=f"Extracting pose features from subject sessions", total = len(self.subjects), unit='subjects'):
            subj_path = Path(subject.subject_path)
            
            for session in subject.sessions:
                sesh_path = session.session_id
                data_path = session.session_outputs.get('movement_features', {}).get('path', None)
                
                if data_path is None:
                    print(f"Data for {subject.subject_id} on session {session.session_id} is None.")
                else:
                    data_path = subj_path / sesh_path / data_path
                    
                    if data_path.exists():
                        ds = load_zarr(data_path, method='xarray')
                        nodes = ds.node.values
                        
                        for node in nodes:
                            
                            date, subject_id, trials, experiment_type = extract_metadata(ds, mask=node)
                            vel_summary = velocity_summary(ds, node)
                            acc_summary = acceleration_summary(ds, node)
                            n_trials = len(next(iter(vel_summary.values())))
                            
                            meta_summary = {
                                'date': [date]*n_trials,
                                'id': [subject_id]*n_trials,
                                'node': [node]*n_trials,
                                'trial': trials.values,
                                'experiment_type': experiment_type.values
                            }

                            summary = meta_summary | vel_summary | acc_summary
                            
                            rows.append(pd.DataFrame(summary))

        if not rows:
            return pd.DataFrame()
        
        df = pd.concat(rows, ignore_index = True)
        df['session_number'] = df.groupby('id')['date'].transform(lambda x: pd.Categorical(x).codes)

        # drop rows where values are NaN

        nan_rows = df.isnull().any(axis=1).sum()
        if nan_rows > 0:
            print(f"Dropping {nan_rows} rows with NaN values "
                  f"({nan_rows/len(df)*100:.1f}% of data)"
                  )
            df = df.dropna()

        df.to_parquet(self.dirs['summaries']/'pose_metrics.parquet')
        self.pose_summary = df

    def analyse(self, framework: str, model: str, data: str, params: dict):
        
        data_path = self.dirs['summaries'] / data
        
        if not data_path.exists():
            raise FileNotFoundError(f"No summary data found at {data_path}. Run summarize() first.")
        
        df = load_parquet(data_path, method='pandas')

        model_fn = get_stats_model(framework = framework, model = model)
        
        return model_fn(df, params, save_path = self.dirs['results'])
