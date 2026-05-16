from importlib.metadata import version, PackageNotFoundError

# set version info

try:
    __version__ = version("neurokinematics")
except PackageNotFoundError:
    __version__ = "development"



# Import specific modules for simplicity
from .data.session import ExperimentSession