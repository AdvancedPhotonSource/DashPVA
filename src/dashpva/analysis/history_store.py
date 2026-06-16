"""
Historical PV lookup over per-frame caches.

A :class:`HistoryStore` answers three shapes of question about a PV's value
over time:

  * point   — value at a specific detector frame id or POSIX timestamp
  * range   — every sample in a [start, end] window
  * summary — mean/min/max/std/count over a window

Two backings share the same lookup logic (see :class:`HistoryStore`):

  * :class:`LiveHistoryStore` reads the in-memory caches on a live
    :class:`~dashpva.utils.pva_reader.PVAReader`.
  * :class:`H5HistoryStore` reads a saved scan ``.h5`` written by
    :class:`~dashpva.utils.hdf5_writer.HDF5Writer` (Phase B datasets).

Frame ids are monotonic increasing, so lookups bisect rather than scan.
When an exact frame/time isn't present (dropped frames are normal) the store
returns the **nearest** sample and flags it via ``exact: False`` plus the
``matched_frame_id`` actually used — the caller (and the LLM) can see the
answer wasn't an exact hit.

A PV's values can live in one of two places, tried in this order:

  1. A dedicated CA series (``cached_ca`` / ``metadata/ca/<friendly>``) — only
     populated for ``[METADATA.CA]`` PVs, and only during a scan.
  2. The per-frame attribute dicts (``cached_attributes``) keyed by the PV name.
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from collections import deque

import numpy as np


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.number)) and not isinstance(v, bool)


def _nearest_index(sorted_keys: list, target: float) -> int | None:
    """Index of the entry in *sorted_keys* closest to *target* (None if empty)."""
    n = len(sorted_keys)
    if n == 0:
        return None
    pos = bisect.bisect_left(sorted_keys, target)
    if pos <= 0:
        return 0
    if pos >= n:
        return n - 1
    before = sorted_keys[pos - 1]
    after = sorted_keys[pos]
    return pos if (after - target) < (target - before) else pos - 1


class HistoryStore(ABC):
    """Read-only historical lookup for a single data source (live or one h5 file)."""

    source_label: str = 'unknown'

    @abstractmethod
    def _series(self, pv_name: str) -> tuple[list, list, list] | None:
        """Return (frame_ids, timestamps, values) aligned lists for *pv_name*,
        sorted by frame_id, or None if the PV has no history here."""

    def get_at_frame(self, pv_name: str, frame_id: int) -> dict:
        series = self._series(pv_name)
        if not series:
            return self._no_history(pv_name)
        fids, ts, vals = series
        idx = _nearest_index(fids, float(frame_id))
        if idx is None:
            return self._no_history(pv_name)
        matched = fids[idx]
        return {
            'pv_name': pv_name,
            'value': vals[idx],
            'requested_frame_id': int(frame_id),
            'matched_frame_id': matched,
            'exact': matched == int(frame_id),
            'timestamp': ts[idx],
            'source': self.source_label,
        }

    def get_at_time(self, pv_name: str, timestamp: float) -> dict:
        series = self._series(pv_name)
        if not series:
            return self._no_history(pv_name)
        fids, ts, vals = series
        # timestamps rise with frame ids, so bisect on ts directly
        idx = _nearest_index(ts, float(timestamp))
        if idx is None:
            return self._no_history(pv_name)
        matched_t = ts[idx]
        return {
            'pv_name': pv_name,
            'value': vals[idx],
            'requested_timestamp': float(timestamp),
            'matched_timestamp': matched_t,
            'exact': abs(matched_t - float(timestamp)) < 1e-6,
            'matched_frame_id': fids[idx],
            'source': self.source_label,
        }

    def get_range(self, pv_name: str, start: float, end: float,
                  by: str = 'frame', max_points: int | None = None) -> dict:
        series = self._series(pv_name)
        if not series:
            return self._no_history(pv_name)
        fids, ts, vals = series
        keys = fids if by == 'frame' else ts
        lo, hi = (start, end) if start <= end else (end, start)
        points = [
            {'frame_id': fids[i], 'timestamp': ts[i], 'value': vals[i]}
            for i in range(len(keys)) if lo <= keys[i] <= hi
        ]
        truncated = False
        if max_points is not None and len(points) > max_points:
            # Even stride so the window is still represented end-to-end.
            stride = max(1, len(points) // max_points)
            points = points[::stride][:max_points]
            truncated = True
        return {
            'pv_name': pv_name,
            'by': by,
            'start': lo,
            'end': hi,
            'n_points': len(points),
            'truncated': truncated,
            'points': points,
            'source': self.source_label,
        }

    def get_summary(self, pv_name: str, start: float, end: float,
                    by: str = 'frame') -> dict:
        rng = self.get_range(pv_name, start, end, by=by, max_points=None)
        if 'error' in rng:
            return rng
        numeric = [p['value'] for p in rng['points'] if _is_number(p['value'])]
        if not numeric:
            return {
                'pv_name': pv_name, 'by': by, 'start': rng['start'],
                'end': rng['end'], 'n': 0,
                'error': 'no numeric samples in range',
                'source': self.source_label,
            }
        arr = np.asarray(numeric, dtype=np.float64)
        return {
            'pv_name': pv_name,
            'by': by,
            'start': rng['start'],
            'end': rng['end'],
            'n': int(arr.size),
            'mean': float(arr.mean()),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'std': float(arr.std()),
            'source': self.source_label,
        }

    def _no_history(self, pv_name: str) -> dict:
        return {
            'pv_name': pv_name,
            'error': f'no cached history for {pv_name!r} in {self.source_label} source',
            'source': self.source_label,
        }


# ---------------------------------------------------------------------------
# Live store — reads the in-memory PVAReader caches
# ---------------------------------------------------------------------------


class LiveHistoryStore(HistoryStore):
    """Historical lookup over a live :class:`PVAReader`'s in-memory caches."""

    source_label = 'live'

    def __init__(self, reader):
        self.reader = reader

    def _series(self, pv_name: str):
        # 1) Dedicated CA series (only populated during a scan).
        cached_ca = getattr(self.reader, 'cached_ca', None) or {}
        if pv_name in cached_ca and cached_ca[pv_name]:
            vals = list(cached_ca[pv_name])
            fids = list((getattr(self.reader, 'cached_ca_frame_ids', {}) or {}).get(pv_name, []))
            ts = list((getattr(self.reader, 'cached_ca_timestamps', {}) or {}).get(pv_name, []))
            series = self._zip_aligned(fids, ts, vals)
            if series:
                return series

        # 2) Per-frame attribute dicts.
        attrs = self._flatten(getattr(self.reader, 'cached_attributes', None))
        fids = self._flatten(getattr(self.reader, 'cached_frame_ids', None))
        ts = self._flatten(getattr(self.reader, 'cached_timestamps', None))
        if not attrs:
            return None
        triples = []
        for i, d in enumerate(attrs):
            if not isinstance(d, dict) or pv_name not in d:
                continue
            fid = fids[i] if i < len(fids) else i
            t = ts[i] if i < len(ts) else 0.0
            triples.append((fid, t, d[pv_name]))
        if not triples:
            return None
        triples.sort(key=lambda x: x[0])
        return ([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])

    @staticmethod
    def _zip_aligned(fids: list, ts: list, vals: list):
        """Zip parallel CA lists, sort by frame id. Falls back gracefully when
        frame_id/timestamp arrays are shorter than values (legacy data)."""
        n = len(vals)
        if n == 0:
            return None
        fids = list(fids) + [i for i in range(len(fids), n)]
        ts = list(ts) + [0.0] * (n - len(ts)) if len(ts) < n else list(ts)
        triples = sorted(zip(fids[:n], ts[:n], vals), key=lambda x: x[0])
        return ([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])

    @staticmethod
    def _flatten(cache) -> list:
        """deque → list, list[deque] (bin mode) → concatenated list."""
        if cache is None:
            return []
        if isinstance(cache, deque):
            return list(cache)
        if isinstance(cache, list) and cache and isinstance(cache[0], deque):
            out: list = []
            for d in cache:
                out.extend(d)
            return out
        return list(cache)


# ---------------------------------------------------------------------------
# HDF5 store — reads a saved scan file (lazy, cached per instance)
# ---------------------------------------------------------------------------


class H5HistoryStore(HistoryStore):
    """Historical lookup over a saved scan ``.h5`` (Phase B datasets).

    Datasets used (all written by :class:`HDF5Writer`):
      ``/entry/data/frame_ids``           int64, parallel to image stack
      ``/entry/data/timestamps``          float64
      ``/entry/data/metadata/ca/<name>``  per-CA value array
      ``/entry/data/metadata/ca/<name>__frame_ids`` / ``__timestamps``

    PVs are addressable by EITHER the CA dataset's friendly name or its
    ``pv_name`` attribute. The class builds a pv_name → friendly map on open.
    """

    source_label = 'h5'

    def __init__(self, path: str):
        self.path = path
        self._series_cache: dict[str, tuple | None] = {}
        self._ca_index: dict[str, str] | None = None  # pv_name/friendly -> friendly

    def _build_ca_index(self, h5) -> dict[str, str]:
        index: dict[str, str] = {}
        ca = h5.get('entry/data/metadata/ca')
        if ca is None:
            return index
        for key in ca.keys():
            if key.endswith('__frame_ids') or key.endswith('__timestamps'):
                continue
            index[key] = key                      # friendly name
            pv_name = ca[key].attrs.get('pv_name')
            if pv_name:
                index[str(pv_name)] = key         # and its real PV name
        return index

    def _series(self, pv_name: str):
        if pv_name in self._series_cache:
            return self._series_cache[pv_name]
        try:
            import h5py
            with h5py.File(self.path, 'r') as h5:
                if self._ca_index is None:
                    self._ca_index = self._build_ca_index(h5)
                friendly = self._ca_index.get(pv_name)
                series = None
                if friendly is not None:
                    series = self._read_ca_series(h5, friendly)
                self._series_cache[pv_name] = series
                return series
        except Exception:
            self._series_cache[pv_name] = None
            return None

    @staticmethod
    def _read_ca_series(h5, friendly: str):
        ca = h5['entry/data/metadata/ca']
        vals = list(ca[friendly][:])
        n = len(vals)
        if n == 0:
            return None
        fid_key = f'{friendly}__frame_ids'
        ts_key = f'{friendly}__timestamps'
        if fid_key in ca:
            fids = [int(x) for x in ca[fid_key][:]]
        else:
            # Fall back to the global frame_ids dataset, then to positional.
            g = h5.get('entry/data/frame_ids')
            fids = [int(x) for x in g[:n]] if g is not None and len(g) >= n else list(range(n))
        if ts_key in ca:
            ts = [float(x) for x in ca[ts_key][:]]
        else:
            g = h5.get('entry/data/timestamps')
            ts = [float(x) for x in g[:n]] if g is not None and len(g) >= n else [0.0] * n
        triples = sorted(zip(fids[:n], ts[:n], vals), key=lambda x: x[0])
        return ([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])

    def available_pvs(self) -> list[str]:
        """Friendly names + pv_names that have history in this file."""
        try:
            import h5py
            with h5py.File(self.path, 'r') as h5:
                if self._ca_index is None:
                    self._ca_index = self._build_ca_index(h5)
                return sorted(self._ca_index.keys())
        except Exception:
            return []
