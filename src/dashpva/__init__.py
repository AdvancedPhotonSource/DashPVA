"""DashPVA: Distributed Analysis and Streaming Hub with Process Variable Access."""

from importlib.metadata import version

__version__ = version("DashPVA")

try:
    import hdf5plugin  # noqa: F401
except Exception:
    pass
