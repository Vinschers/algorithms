import importlib.metadata

from .data_structures import *

try:
    __version__ = importlib.metadata.version("vicentin")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
