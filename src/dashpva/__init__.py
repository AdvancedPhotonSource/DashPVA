"""DashPVA: Distributed Analysis and Streaming Hub with Process Variable Access."""

__version__ = "1.0.0"

try:
    import hdf5plugin  # noqa: F401
except Exception:
    pass
