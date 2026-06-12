"""SIE Server - Search Inference Engine."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sie_server")
except PackageNotFoundError:
    # Package is not installed (e.g. source tree without `pip install -e`).
    __version__ = "0.0.0+unknown"
