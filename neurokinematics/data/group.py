from pathlib import Path
import yaml

import pandas as pd

# data
from neurokinematics.data.subject import ExperimentSubject
from neurokinematics.data.project import NKProject

# io
from neurokinematics.io import load_yaml, save_yaml, load_zarr

# pose
from neurokinematics.pose.utils import pixels_to_cm
from neurokinematics.pose.features import extract_max_velocity, extract_metadata

MANDATORY_GROUP_SPEC_KEYS = ['group_id', 'output_root', 'subjects']

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
            self.project_path = Path(project_path)
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
            self._save_group_specs()


    @classmethod
    def from_existing(cls, group_path: Path | str):
        group_path = Path(group_path)
        group_spec_path = group_path / "group_spec.yaml"

        group = cls(group_specs = None)
        group.group_path = group_path
        group.group_specs = load_yaml(group_spec_path)
        group.group_id = group.group_specs['group_id']
        #group.output_root = Path(group.group_specs['output_root'])
        group.project_path = group.group_path.parent.parent
        group.subject_root = group.project_path / "Subjects" 
        group.subjects_log = group.group_specs['runtime']['subjects']
        
        #group.group_path = group.output_root / group.group_id

        group.subjects = [
            ExperimentSubject.from_existing(group.subject_root / s['spec']) for s in group.subjects_log
        ]

        return group

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
        self.subjects = []
        self.group_specs['runtime'] = {'subjects': []}

        for subj in self.subjects_log:
            s = ExperimentSubject(subject_specs = subj['spec'], project_path = self.project_path, name = self.project_name)
            self.subjects.append(s)
            self.group_specs['runtime']['subjects'].append({'spec': str(s.subject_spec_path.relative_to(self.subject_root).parent)})

    def process_subjects(self):
        for subj in self.subjects:
            subj.process_sessions()
    
    def par_process_subjects(self):
        for subject in self.subjects:
            subject.par_process_sessions()
    def add_subjects(self):
        pass

    def pose_summary(self, feature: str = 'velocity'):
        # start by just collecting velocity data...
        ds_list = []
        rows = []
        for subject in self.subjects:
            subj_path = Path(subject.subject_path)
            for session in subject.sessions:
                sesh_path = session.session_id
                data_path = session.session_outputs.get('movement_features', {}).get('path', None)
                if data_path == None:
                    print(f"Data for {subj.subject_id} on session {session.session_id} is None.")
                else:
                    data_path = subj_path / Path(sesh_path) / Path(data_path)
                    if data_path.exists():
                        ds = load_zarr(data_path, method='xarray')
                        nodes = ds[0].node.values
                        for node in nodes:
                            date, subject_id = extract_metadata(ds)
                            vx, vy = extract_max_velocity(ds)

                        ds_list.append(ds)
        nodes = ds_list[0].node.values
        for node in nodes:
            for ds in ds_list:
                date, id = extract_metadata(ds)
                vx, vy = extract_max_velocity(ds, node=node)
                df_data = pd.DataFrame({
                    'vx': vx.values,
                    'vy': vy.values,
                    'id': subject_id,
                    'date': date
                })