from pathlib import Path
import yaml

# data
from neurokinematics.data.subject import ExperimentSubject

# io
from neurokinematics.io import load_yaml

MANDATORY_GROUP_SPEC_KEYS = ['group_id', 'output_root', 'subjects']

class ExperimentGroup:

    def __init__(self, group_specs: str | Path | dict):
        
        self.group_specs = self._load_group_specs(group_specs)
        self.group_id = self.group_specs['group_id']
        self.output_root = Path(self.group_specs['output_root'])
        self.subjects_log = self.group_specs['subjects']

        self.subjects_path = self.output_root / self.group_id

        self.create_subjects_from_log()
        self.subjects_path.mkdir(parents=True, exist_ok=True)


    @classmethod
    def from_existing(cls, group_path: Path | str):
        group_path = Path(group_path)
        group_spec_path = group_path / "group_spec.yaml"

        spec = load_yaml(group_spec_path)
        

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

    def create_subjects_from_log(self):
        self.subjects = []

        for subj in self.subjects_log:

            self.subjects.append(ExperimentSubject(subject_specs = subj['spec'], output_root = self.subjects_path))

    def process_subjects(self):
        for subj in self.subjects:
            subj.process_sessions()
    
    def add_subjects(self):
        pass