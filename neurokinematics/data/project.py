from pathlib import Path

class NKProject:
    def __init__(self, root: str | Path | None = None, name: str = 'NK'):
        if root is None:
            root = Path.home()

        self.root = Path(root) / name
        self.name = name
        self._init_structure()


    def _init_structure(self):
        self.subject_root = self.root / "Subjects"
        self.group_root = self.root / "Groups"

        self.subject_root.mkdir(parents=True, exist_ok=True)
        self.group_root.mkdir(parents=True, exist_ok=True)
