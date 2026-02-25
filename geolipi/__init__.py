"""GeoLIPI package."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["symbolic", "torch_compute", "geometry_nodes"]

try:
    __version__ = version("geolipi")
except PackageNotFoundError:
    # Fallback for editable installs without metadata
    __version__ = "0.0.0"
