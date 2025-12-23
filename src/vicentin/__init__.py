import importlib.metadata

# from . import image
from .data_structures import *


try:
    __version__ = importlib.metadata.version("vicentin")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
