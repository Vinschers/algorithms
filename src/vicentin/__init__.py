import importlib.metadata

# from . import image
from .data_structures import *

from . import deep_learning as dl

try:
    __version__ = importlib.metadata.version("vicentin")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
