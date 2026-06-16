"""Tests for HistoryStore — LiveHistoryStore (in-memory) and H5HistoryStore (saved scan)."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from dashpva.analysis.history_store import (
    H5HistoryStore,
    LiveHistoryStore,
    _nearest_index,
)

# ---------------------------------------------------------------------------
# bisect helper
# ---------------------------------------------------------------------------


class TestNearestIndex:
    def test_empty(self):
        assert _nearest_index([], 5) is None

    def test_exact(self):
        assert _nearest_index([10, 20, 30], 20) == 1

    def test_below_range(self):
        assert _nearest_index([10, 20, 30], 3) == 0

    def test_above_range(self):
        assert _nearest_index([10, 20, 30], 99) == 2

    def test_nearest_rounds_to_closer(self):
        # 23 is closer to 20 than 30
        assert _nearest_index([10, 20, 30], 23) == 1
        # 27 is closer to 30
        assert _nearest_index([10, 20, 30], 27) == 2


# ---------------------------------------------------------------------------
# Live store
# ---------------------------------------------------------------------------


class _LiveReader:
    """Stand-in PVAReader exposing the caches LiveHistoryStore reads."""

    def __init__(self, *, cached_ca=None, cached_ca_frame_ids=None,
                 cached_ca_timestamps=None, cached_attributes=None,
                 cached_frame_ids=None, cached_timestamps=None):
        self.cached_ca = cached_ca or {}
        self.cached_ca_frame_ids = cached_ca_frame_ids or {}
        self.cached_ca_timestamps = cached_ca_timestamps or {}
        self.cached_attributes = cached_attributes
        self.cached_frame_ids = cached_frame_ids
        self.cached_timestamps = cached_timestamps


class TestLiveHistoryStoreCASeries:

    def _reader(self):
        # Frames 100,101,103 (102 dropped); energy ramps.
        return _LiveReader(
            cached_ca={'sim:energy': [8.0, 8.1, 8.3]},
            cached_ca_frame_ids={'sim:energy': [100, 101, 103]},
            cached_ca_timestamps={'sim:energy': [1.0, 1.1, 1.3]},
        )

    def test_get_at_frame_exact(self):
        s = LiveHistoryStore(self._reader())
        r = s.get_at_frame('sim:energy', 101)
        assert r['value'] == 8.1
        assert r['matched_frame_id'] == 101
        assert r['exact'] is True
        assert r['source'] == 'live'

    def test_get_at_frame_nearest_flags_inexact(self):
        s = LiveHistoryStore(self._reader())
        r = s.get_at_frame('sim:energy', 102)  # dropped frame
        assert r['exact'] is False
        assert r['matched_frame_id'] in (101, 103)
        assert r['requested_frame_id'] == 102

    def test_get_at_time_nearest(self):
        s = LiveHistoryStore(self._reader())
        r = s.get_at_time('sim:energy', 1.05)
        assert r['value'] in (8.0, 8.1)
        assert 'matched_timestamp' in r

    def test_get_range_by_frame(self):
        s = LiveHistoryStore(self._reader())
        r = s.get_range('sim:energy', 100, 101, by='frame')
        assert r['n_points'] == 2
        assert [p['value'] for p in r['points']] == [8.0, 8.1]

    def test_get_range_max_points_truncates(self):
        s = LiveHistoryStore(self._reader())
        r = s.get_range('sim:energy', 0, 1000, by='frame', max_points=2)
        assert r['truncated'] is True
        assert r['n_points'] <= 2

    def test_get_summary_stats(self):
        s = LiveHistoryStore(self._reader())
        r = s.get_summary('sim:energy', 0, 1000, by='frame')
        assert r['n'] == 3
        assert pytest.approx(r['min']) == 8.0
        assert pytest.approx(r['max']) == 8.3
        assert pytest.approx(r['mean']) == pytest.approx((8.0 + 8.1 + 8.3) / 3)

    def test_unknown_pv_returns_error(self):
        s = LiveHistoryStore(self._reader())
        r = s.get_at_frame('sim:nonexistent', 100)
        assert 'error' in r


class TestLiveHistoryStorePerFrameAttributes:

    def _reader(self):
        # PV not in cached_ca; lives in per-frame attribute dicts instead.
        attrs = deque([
            {'sim:m1': 1.0}, {'sim:m1': 2.0}, {'sim:m1': 3.0},
        ])
        fids = deque([10, 11, 12])
        ts = deque([0.0, 0.5, 1.0])
        return _LiveReader(cached_attributes=attrs, cached_frame_ids=fids,
                           cached_timestamps=ts)

    def test_falls_back_to_per_frame_attrs(self):
        s = LiveHistoryStore(self._reader())
        r = s.get_at_frame('sim:m1', 11)
        assert r['value'] == 2.0
        assert r['matched_frame_id'] == 11

    def test_summary_over_per_frame_attrs(self):
        s = LiveHistoryStore(self._reader())
        r = s.get_summary('sim:m1', 10, 12, by='frame')
        assert r['n'] == 3
        assert pytest.approx(r['mean']) == 2.0

    def test_bin_mode_list_of_deques_flattened(self):
        # bin mode: cached_attributes is list[deque]
        attrs = [deque([{'sim:m1': 1.0}]), deque([{'sim:m1': 2.0}])]
        fids = [deque([10]), deque([11])]
        ts = [deque([0.0]), deque([0.1])]
        reader = _LiveReader(cached_attributes=attrs, cached_frame_ids=fids,
                             cached_timestamps=ts)
        s = LiveHistoryStore(reader)
        r = s.get_at_frame('sim:m1', 11)
        assert r['value'] == 2.0


# ---------------------------------------------------------------------------
# H5 store — round-trip through the real HDF5Writer
# ---------------------------------------------------------------------------


def _write_scan_h5(tmp_path, monkeypatch):
    """Write a small scan .h5 with CA history via the real HDF5Writer."""
    import dashpva.settings as settings
    from dashpva.utils.hdf5_writer import HDF5Writer

    monkeypatch.setattr(settings, 'METADATA_CA', {'EnergyKeV': 'sim:energy'})

    class _Reader:
        cached_frame_ids = deque([100, 101, 103])
        cached_timestamps = deque([1.0, 1.1, 1.3])
        cached_ca = {'sim:energy': [8.0, 8.1, 8.3]}
        cached_ca_frame_ids = {'sim:energy': [100, 101, 103]}
        cached_ca_timestamps = {'sim:energy': [1.0, 1.1, 1.3]}
        feature_vector_cache = []
        sampled_descriptions = []
        blob_detections_cache = []
        blob_tracks_cache = []
        config = {}

    writer = HDF5Writer(str(tmp_path / 'scan.h5'), _Reader())
    n = 3
    flat_img = np.zeros(16, dtype=np.uint16)
    data = {
        'images': [flat_img.copy() for _ in range(n)],
        'attributes': [{'sim:energy': v} for v in (8.0, 8.1, 8.3)],
        'rsm': ([], [], []),
        'shape': (4, 4),
        'len_images': n,
        'len_attributes': n,
        'HKL_IN_CONFIG': False,
        'metadata': writer.merge_metadata([{'sim:energy': v} for v in (8.0, 8.1, 8.3)]),
        'cached_ca': {'sim:energy': [8.0, 8.1, 8.3]},
    }
    out = tmp_path / 'scan.h5'
    writer.h5_save(str(out), data, compress=False)
    return str(out)


class TestH5HistoryStore:

    def test_lookup_by_friendly_name(self, tmp_path, monkeypatch):
        path = _write_scan_h5(tmp_path, monkeypatch)
        s = H5HistoryStore(path)
        r = s.get_at_frame('EnergyKeV', 101)
        assert r['value'] == pytest.approx(8.1)
        assert r['exact'] is True
        assert r['source'] == 'h5'

    def test_lookup_by_pv_name(self, tmp_path, monkeypatch):
        path = _write_scan_h5(tmp_path, monkeypatch)
        s = H5HistoryStore(path)
        r = s.get_at_frame('sim:energy', 103)
        assert r['value'] == pytest.approx(8.3)

    def test_nearest_on_dropped_frame(self, tmp_path, monkeypatch):
        path = _write_scan_h5(tmp_path, monkeypatch)
        s = H5HistoryStore(path)
        r = s.get_at_frame('EnergyKeV', 102)
        assert r['exact'] is False
        assert r['matched_frame_id'] in (101, 103)

    def test_summary(self, tmp_path, monkeypatch):
        path = _write_scan_h5(tmp_path, monkeypatch)
        s = H5HistoryStore(path)
        r = s.get_summary('EnergyKeV', 0, 1000, by='frame')
        assert r['n'] == 3
        assert pytest.approx(r['max']) == 8.3

    def test_available_pvs_lists_both_names(self, tmp_path, monkeypatch):
        path = _write_scan_h5(tmp_path, monkeypatch)
        s = H5HistoryStore(path)
        names = s.available_pvs()
        assert 'EnergyKeV' in names
        assert 'sim:energy' in names

    def test_missing_file_returns_error(self):
        s = H5HistoryStore('/nonexistent/path.h5')
        r = s.get_at_frame('EnergyKeV', 100)
        assert 'error' in r
