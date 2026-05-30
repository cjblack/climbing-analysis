from pathlib import Path
import yaml

# data
from neurokinematics.data.subject import ExperimentSubject

# io
from neurokinematics.io import load_yaml, save_yaml

MANDATORY_GROUP_SPEC_KEYS = ['group_id', 'output_root', 'subjects']

class ExperimentGroup:

    def __init__(self, group_specs: str | Path | dict | None):
        
        if group_specs is not None:
            self.group_specs = self._load_group_specs(group_specs)
            self.group_id = self.group_specs['group_id']
            self.output_root = Path(self.group_specs['output_root'])
            self.subjects_log = self.group_specs['subjects']

            self.group_path = self.output_root / self.group_id
            self.group_path.mkdir(parents=True, exist_ok=True)
            self.create_subjects_from_log()
            self._save_group_specs()


    @classmethod
    def from_existing(cls, group_path: Path | str):
        group_path = Path(group_path)
        group_spec_path = group_path / "group_spec.yaml"

        group = cls(group_specs = None)
        group.group_specs = load_yaml(group_spec_path)
        group.group_id = group.group_specs['group_id']
        group.output_root = Path(group.group_specs['output_root'])
        group.subjects_log = group.group_specs['runtime']['subjects']
        group.group_path = group.output_root / group.group_id

        group.subjects = [
            ExperimentSubject.from_existing(group_path / s['spec']) for s in group.subjects_log
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
            s = ExperimentSubject(subject_specs = subj['spec'], output_root = self.group_path)
            self.subjects.append(s)
            self.group_specs['runtime']['subjects'].append({'spec': str(s.subject_spec_path.relative_to(self.group_path).parent)})

    def process_subjects(self):
        for subj in self.subjects:
            subj.process_sessions()
    
    def add_subjects(self):
        pass