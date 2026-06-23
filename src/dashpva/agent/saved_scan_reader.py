"""``SavedScanReader`` — drive the existing analysis tools over a recorded ``.h5``.


HDF5 → cache mapping (schema in ``hdf5_writer.py::h5_save``):

==================================================  =========================
HDF5 dataset                                        reader attribute
==================================================  =========================
``/entry/data/data`` (n, H, W)                      ``cached_images`` (lazy)
``/entry/data/frame_ids``                           ``cached_frame_ids``
``/entry/data/timestamps``                          ``cached_timestamps``
``/entry/data/feature_vectors`` (vlen JSON)         ``feature_vector_cache``
``/entry/data/sampled_descriptions`` (vlen JSON)    ``sampled_descriptions``
``/entry/data/blob_detections`` (n, maxN, 5)        ``blob_detections_cache``
``/entry/data/metadata/ca/<friendly>``              ``cached_ca[pv_name]``
``…/ca/<friendly>__frame_ids`` / ``__timestamps``   ``cached_ca_*[pv_name]``
==================================================  =========================

Two contract details that matter (see HANDOFF_PHASE4 §6):

  * caches are stored as :class:`collections.deque` so
    :meth:`LiveHistoryStore._flatten` handles them unchanged;
  * ``cached_images`` holds **lazy** per-frame proxies — each materializes its
    frame (raveled 1-D C-order, to match ``PVAReader``) only when ``np.asarray``
    is called on it, so a large scan is never loaded into RAM all at once.

PV history (``cached_ca``) is keyed by **PV name** (not the friendly name) so
``LiveHistoryStore`` and ``correlate_series('pv:…')`` find it; it only exists if
the scan was recorded in *scan* caching mode (else PV tools honestly report
"no data").
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import h5py
import numpy as np

_DATA_GROUP = "entry/data"
_IMAGE_DATASET = "entry/data/data"
_CA_GROUP = "entry/data/metadata/ca"


class _LazyFrame:
    """A single detector frame, read from the ``.h5`` only when materialized.

    Returned as a **raveled 1-D C-order** array to match
    ``PVAReader.cached_images`` (``analysis_tools._image_2d`` reshapes it back to
    2-D via the reader's ``shape``). Re-opens the file per access, so it holds no
    long-lived handle and is safe to read from a worker thread.
    """

    __slots__ = ("_path", "_idx")

    def __init__(self, path: str, idx: int):
        self._path = path
        self._idx = int(idx)

    def __array__(self, dtype=None, copy=None):
        with h5py.File(self._path, "r") as f:
            frame = np.asarray(f[_IMAGE_DATASET][self._idx])
        arr = np.ravel(frame)  # C-order, matches np.ravel(self.image) in PVAReader
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_LazyFrame(idx={self._idx})"


class SavedScanReader:
    """A read-only, ``ScanReader``-shaped view over a recorded scan ``.h5``.

    Args:
        h5_path: path to a scan file written by :class:`HDF5Writer`.
        settings: the DashPVA settings module (or any object exposing
            ``CONFIG`` and ``METADATA_CA``). When omitted, ``dashpva.settings``
            is imported — load the same config the scan was taken with so
            friendly PV names line up.
    """

    def __init__(self, h5_path, settings=None):
        self.h5_path = str(Path(h5_path).expanduser())
        self._settings = settings if settings is not None else _import_settings()

        # --- ScanReader attributes (defaults; filled by _load) ---------------
        self.cached_images: deque | None = None
        self.cached_frame_ids: deque | None = None
        self.cached_timestamps: deque | None = None
        self.feature_vector_cache: list = []
        self.blob_detections_cache: list = []
        self.sampled_descriptions: list = []
        self.cached_ca: dict = {}
        self.cached_ca_frame_ids: dict = {}
        self.cached_ca_timestamps: dict = {}
        self.cached_attributes = None  # saved files store CA series, not per-frame dicts
        self.shape: tuple = (0, 0)
        self.image_is_transposed = False
        self.frames_received = 0
        self.config: dict = dict(getattr(self._settings, "CONFIG", {}) or {})
        self.CACHING_MODE = "saved"

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with h5py.File(self.h5_path, "r") as h5:
            dgrp = h5.get(_DATA_GROUP)
            if dgrp is None:
                raise ValueError(
                    f"{self.h5_path!r} is not a DashPVA scan file "
                    f"(missing /{_DATA_GROUP})"
                )
            data = dgrp.get("data")
            if data is None:
                raise ValueError(
                    f"{self.h5_path!r} has no image stack (missing /{_IMAGE_DATASET})"
                )

            n, shape = self._image_geometry(data)
            self.shape = shape
            self.frames_received = n

            # Lazy image proxies — no frame is read here.
            self.cached_images = deque(_LazyFrame(self.h5_path, i) for i in range(n))

            self.cached_frame_ids = deque(self._read_int_axis(dgrp, "frame_ids", n))
            self.cached_timestamps = deque(self._read_float_axis(dgrp, "timestamps", n))

            self.feature_vector_cache = self._read_json_list(dgrp, "feature_vectors")
            self.sampled_descriptions = self._read_json_list(dgrp, "sampled_descriptions")
            self.blob_detections_cache = self._read_blob_detections(dgrp)
            self._read_ca_series(h5)

    @staticmethod
    def _image_geometry(data) -> tuple[int, tuple]:
        """(n_frames, (H, W)) for the image dataset, tolerating a single 2-D frame."""
        if data.ndim == 3:
            return int(data.shape[0]), (int(data.shape[1]), int(data.shape[2]))
        if data.ndim == 2:
            return 1, (int(data.shape[0]), int(data.shape[1]))
        raise ValueError(f"unexpected image dataset ndim={data.ndim}; expected 2 or 3")

    @staticmethod
    def _read_int_axis(dgrp, name: str, n: int) -> list:
        ds = dgrp.get(name)
        if ds is not None and len(ds) >= n:
            return [int(x) for x in ds[:n]]
        return list(range(n))  # positional fallback

    @staticmethod
    def _read_float_axis(dgrp, name: str, n: int) -> list:
        ds = dgrp.get(name)
        if ds is not None and len(ds) >= n:
            return [float(x) for x in ds[:n]]
        return [0.0] * n

    @staticmethod
    def _read_json_list(dgrp, name: str) -> list:
        ds = dgrp.get(name)
        if ds is None:
            return []
        out: list = []
        for raw in ds[:]:
            s = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
            try:
                out.append(json.loads(s))
            except Exception:
                continue
        return out

    @staticmethod
    def _read_blob_detections(dgrp) -> list:
        """``(n, maxN, 5)`` NaN-padded → list of ``(N_i, 5)`` arrays (pad dropped)."""
        ds = dgrp.get("blob_detections")
        if ds is None:
            return []
        padded = np.asarray(ds[:])
        if padded.ndim != 3:
            return []
        out: list = []
        for frame in padded:
            rows = frame[~np.isnan(frame).any(axis=1)]
            out.append(np.asarray(rows))
        return out

    def _read_ca_series(self, h5) -> None:
        """Reconstruct ``cached_ca`` keyed by PV name from ``metadata/ca/*``.

        Each friendly dataset carries its real PV name in a ``pv_name`` attr; we
        fall back to ``settings.METADATA_CA`` (friendly → PV) when it is absent.
        Keying by PV name is what lets ``LiveHistoryStore`` and
        ``correlate_series('pv:…')`` find the series unchanged.
        """
        ca = h5.get(_CA_GROUP)
        if ca is None:
            return
        friendly_to_pv = {
            str(k): str(v)
            for k, v in (getattr(self._settings, "METADATA_CA", {}) or {}).items()
        }
        for key in ca.keys():
            if key.endswith("__frame_ids") or key.endswith("__timestamps"):
                continue
            dset = ca[key]
            pv_name = dset.attrs.get("pv_name") or friendly_to_pv.get(key) or key
            pv_name = str(pv_name)
            vals = list(dset[:])
            if not vals:
                continue
            self.cached_ca[pv_name] = vals
            fid_key = f"{key}__frame_ids"
            ts_key = f"{key}__timestamps"
            if fid_key in ca:
                self.cached_ca_frame_ids[pv_name] = [int(x) for x in ca[fid_key][:]]
            if ts_key in ca:
                self.cached_ca_timestamps[pv_name] = [float(x) for x in ca[ts_key][:]]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def close(self) -> None:
        """No-op: the reader holds no open file handle (frames re-open on access).
        Provided for API symmetry and forward-compatibility."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"SavedScanReader(path={self.h5_path!r}, frames={self.frames_received}, "
            f"shape={self.shape}, features={len(self.feature_vector_cache)}, "
            f"ca_pvs={sorted(self.cached_ca)})"
        )


def _import_settings():
    import dashpva.settings as settings

    return settings