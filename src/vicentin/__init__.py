import importlib.metadata

try:
    __version__ = importlib.metadata.version("vicentin")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
