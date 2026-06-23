"""The ``ScanReader`` contract the analysis tools depend on.

The tools in :mod:`dashpva.analysis.tools` were written against the live
:class:`~dashpva.utils.pva_reader.PVAReader`, but they only ever touch a small,
well-defined subset of its attributes. This module *names that subset* as a
:class:`typing.Protocol` so any data source — the live reader, a saved-scan
reader, a future live agent reader — can satisfy it and drive the tools
unchanged.

This is documentation + type-safety, not behavior: ``Protocol`` is structural,
so :class:`~dashpva.utils.pva_reader.PVAReader` already conforms without
importing or subclassing anything here. New readers should be annotated
``ScanReader`` and will be checked against this contract by a type checker.

The attribute set was confirmed by reading every cache access in
``analysis/tools/*`` and ``analysis/history_store.py``:

  * frame caches    — ``cached_images``, ``cached_frame_ids``,
    ``cached_timestamps`` (deques, or list-of-deques in bin mode; flattened by
    :meth:`LiveHistoryStore._flatten`).
  * feature data    — ``feature_vector_cache`` (list[dict], one per frame),
    ``blob_detections_cache`` (list of (N,5) arrays), ``sampled_descriptions``.
  * PV history      — ``cached_ca`` / ``cached_ca_frame_ids`` /
    ``cached_ca_timestamps`` (keyed by **PV name**; only populated when the scan
    was recorded in *scan* caching mode), and ``cached_attributes`` (per-frame
    attribute dicts) as a fallback series source.
  * geometry/state  — ``shape`` (H, W), ``image_is_transposed``,
    ``frames_received``, ``config`` (the TOML config dict), ``CACHING_MODE``.

The agent reads a *lean* subset of ``PVAReader`` — it deliberately does NOT
require ``cached_qx/qy/qz`` (hkl3d-only) or ``blob_tracks*`` (area-det dock-only).
"""

from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable


@runtime_checkable
class ScanReader(Protocol):
    """Structural contract: the attributes the analysis tools read off a reader.

    Any object exposing these attributes can be passed wherever the tools expect
    a ``pva_reader``. ``@runtime_checkable`` allows ``isinstance(obj, ScanReader)``
    (attribute-presence check only — it does not validate types).
    """

    # --- frame caches (deque, or list[deque] in bin mode; may be None) -------
    cached_images: deque | list | None
    """Per-frame detector images as **raveled 1-D C-order** arrays. Reshaped to
    2-D via :func:`analysis_tools._image_2d` using ``shape`` / ``image_is_transposed``."""

    cached_frame_ids: deque | list | None
    """Per-frame detector uniqueIds, parallel to ``cached_images``."""

    cached_timestamps: deque | list | None
    """Per-frame POSIX timestamps, parallel to ``cached_images``."""

    # --- feature / detection data -------------------------------------------
    feature_vector_cache: list
    """One feature dict per frame: ``{frame_id, timestamp, n_blobs, blobs, frame:{...}}``."""

    blob_detections_cache: list
    """Per-frame blob detections, each an ``(N, 5)`` array ``[x1, y1, x2, y2, score]``."""

    sampled_descriptions: list
    """Sparse VLM (vision) text samples taken during the session, if any."""

    # --- PV history (keyed by PV name; scan-mode only) ----------------------
    cached_ca: dict
    """``{pv_name: [values]}`` captured per scan step. Empty unless recorded in
    *scan* caching mode (or reconstructed from a saved file)."""

    cached_ca_frame_ids: dict
    """``{pv_name: [frame_id]}`` parallel to ``cached_ca``."""

    cached_ca_timestamps: dict
    """``{pv_name: [timestamp]}`` parallel to ``cached_ca``."""

    cached_attributes: deque | list | None
    """Per-frame attribute dicts (PV name -> value); fallback PV-history source."""

    # --- geometry / state ----------------------------------------------------
    shape: tuple
    """Image shape ``(H, W)``."""

    image_is_transposed: bool
    """Whether stored frames need an axis swap before display/analysis."""

    frames_received: int
    """Total frames ingested (a saved reader reports the file's frame count)."""

    config: dict
    """The TOML config dict (top-level keys: METADATA, ROI, STATS, HKL, ...)."""

    CACHING_MODE: str
    """``'alignment'`` / ``'scan'`` / ``'bin'`` for a live reader; ``'saved'`` for a file."""