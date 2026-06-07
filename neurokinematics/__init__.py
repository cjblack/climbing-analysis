from importlib.metadata import version, PackageNotFoundError

# set version info

try:
    __version__ = version("neurokinematics")
except PackageNotFoundError:
    __version__ = "development"



# Core top-level API
from .data.session import ExperimentSession
from .data.subject import ExperimentSubject
from .data.group import ExperimentGroup
from . import io