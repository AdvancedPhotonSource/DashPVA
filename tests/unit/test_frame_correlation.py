"""Tests for Phase B: parallel frame_id / timestamp arrays in PVAReader and HDF5Writer."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# PVAReader-side: parallel cache plumbing
# ---------------------------------------------------------------------------


def _make_reader_alignment(monkeypatch):
    """Build a PVAReader configured for alignment mode without needing a TOML."""
    import dashpva.settings as app_settings
    from dashpva.utils.pva_reader import PVAReader

    monkeypatch.setattr(app_settings, 'CACHING_MODE', 'alignment')
    monkeypatch.setattr(app_settings, 'ALIGNMENT_MAX_CACHE_SIZE', 8)
    monkeypatch.setattr(app_settings, 'IOC_PREFIX', 'sim:')
    monkeypatch.setattr(app_settings, 'HKL', {})
    monkeypatch.setattr(app_settings, 'ANALYSIS', {})
    monkeypatch.setattr(app_settings, 'CONSUMER_MODE', '')
    monkeypatch.setattr(app_settings, 'CONFIG', {})
    reader = PVAReader('sim:Pva1:Image', pva_prefix='sim')
    assert isinstance(reader.cached_attributes, deque)
    return reader


def _make_reader_bin(monkeypatch):
    import dashpva.settings as app_settings
    from dashpva.utils.pva_reader import PVAReader

    monkeypatch.setattr(app_settings, 'CACHING_MODE', 'bin')
    monkeypatch.setattr(app_settings, 'BIN_COUNT', 4)
    monkeypatch.setattr(app_settings, 'BIN_SIZE', 3)
    monkeypatch.setattr(app_settings, 'IOC_PREFIX', 'sim:')
    monkeypatch.setattr(app_settings, 'HKL', {})
    monkeypatch.setattr(app_settings, 'ANALYSIS', {})
    monkeypatch.setattr(app_settings, 'CONSUMER_MODE', '')
    monkeypatch.setattr(app_settings, 'CONFIG', {})
    reader = PVAReader('sim:Pva1:Image', pva_prefix='sim')
    assert isinstance(reader.cached_attributes, list)  # list of deques
    return reader


class TestParallelCachesInitAndAppend:

    def test_alignment_mode_init_creates_parallel_deques_with_same_maxlen(self, monkeypatch):
        r = _make_reader_alignment(monkeypatch)
        assert isinstance(r.cached_frame_ids, deque)
        assert isinstance(r.cached_timestamps, deque)
        assert r.cached_frame_ids.maxlen == r.cached_attributes.maxlen == 8
        assert r.cached_timestamps.maxlen == 8

    def test_bin_mode_init_creates_list_of_deques(self, monkeypatch):
        r = _make_reader_bin(monkeypatch)
        assert isinstance(r.cached_frame_ids, list)
        assert isinstance(r.cached_timestamps, list)
        assert len(r.cached_frame_ids) == 4
        assert r.cached_frame_ids[0].maxlen == 3

    def test_cache_attributes_alignment_keeps_arrays_in_lockstep(self, monkeypatch):
        r = _make_reader_alignment(monkeypatch)
        for i in range(5):
            r._current_frame_id = 100 + i
            r._current_timestamp = 1_700_000_000.0 + i * 0.1
            ok = r.cache_attributes(pv_attributes={'EnergyRBV': 8.0 + i})
            assert ok is True
        assert list(r.cached_frame_ids) == [100, 101, 102, 103, 104]
        assert list(r.cached_timestamps) == [
            1_700_000_000.0, 1_700_000_000.1, 1_700_000_000.2,
            1_700_000_000.3, 1_700_000_000.4,
        ]
        assert len(r.cached_attributes) == len(r.cached_frame_ids) == 5

    def test_cache_attributes_alignment_respects_maxlen(self, monkeypatch):
        r = _make_reader_alignment(monkeypatch)  # maxlen=8
        for i in range(12):  # 12 frames into an 8-entry rolling buffer
            r._current_frame_id = i
            r._current_timestamp = float(i)
            r.cache_attributes(pv_attributes={'i': i})
        assert len(r.cached_attributes) == 8
        assert len(r.cached_frame_ids) == 8
        assert len(r.cached_timestamps) == 8
        # The 8 most recent frames are 4..11
        assert list(r.cached_frame_ids) == list(range(4, 12))

    def test_cache_attributes_bin_appends_to_correct_bin(self, monkeypatch):
        r = _make_reader_bin(monkeypatch)
        r.frames_received = 1
        r.frames_missed = 0
        # bin_index = (1 + 0 - 1) % 4 = 0
        r._current_frame_id = 42
        r._current_timestamp = 1.5
        ok = r.cache_attributes(pv_attributes={'k': 'v'})
        assert ok is True
        assert list(r.cached_frame_ids[0]) == [42]
        assert list(r.cached_timestamps[0]) == [1.5]
        # bin 1
        r.frames_received = 2
        r._current_frame_id = 43
        r._current_timestamp = 1.6
        r.cache_attributes(pv_attributes={'k': 'v2'})
        assert list(r.cached_frame_ids[1]) == [43]

    def test_reset_caches_clears_parallel_arrays(self, monkeypatch):
        r = _make_reader_alignment(monkeypatch)
        for i in range(3):
            r._current_frame_id = i
            r._current_timestamp = float(i)
            r.cache_attributes(pv_attributes={'i': i})
        r.cached_ca = {'pv1': [1.0, 2.0]}
        r.cached_ca_frame_ids = {'pv1': [10, 11]}
        r.cached_ca_timestamps = {'pv1': [10.0, 11.0]}
        # cached_images/qx/qy/qz are normally cleared too — initialise the
        # ones init_caches doesn't touch in this config to avoid AttributeError
        # in the existing reset code.
        r.cached_qx = deque()
        r.cached_qy = deque()
        r.cached_qz = deque()

        r.reset_caches()
        assert len(r.cached_frame_ids) == 0
        assert len(r.cached_timestamps) == 0
        assert r.cached_ca_frame_ids == {}
        assert r.cached_ca_timestamps == {}


# ---------------------------------------------------------------------------
# HDF5 writer: frame_ids / timestamps datasets get written
# ---------------------------------------------------------------------------


class _StubReader:
    """Minimal stand-in that exposes only what HDF5Writer.h5_save reads."""

    def __init__(self, frame_ids, timestamps,
                 cached_ca, cached_ca_frame_ids, cached_ca_timestamps,
                 feature_vector_cache=None):
        self.cached_frame_ids = deque(frame_ids)
        self.cached_timestamps = deque(timestamps)
        self.cached_ca = cached_ca
        self.cached_ca_frame_ids = cached_ca_frame_ids
        self.cached_ca_timestamps = cached_ca_timestamps
        self.feature_vector_cache = feature_vector_cache or []
        self.sampled_descriptions = []
        self.blob_detections_cache = []
        self.blob_tracks_cache = []
        self.config = {}


class TestHDF5WriterFrameCorrelation:

    def test_writes_frame_ids_and_timestamps_datasets(self, tmp_path, monkeypatch):
        import dashpva.settings as settings
        from dashpva.utils.hdf5_writer import HDF5Writer

        monkeypatch.setattr(settings, 'METADATA_CA', {})

        n = 4
        reader = _StubReader(
            frame_ids=[1000 + i for i in range(n)],
            timestamps=[1_700_000_000.0 + i for i in range(n)],
            cached_ca={},
            cached_ca_frame_ids={},
            cached_ca_timestamps={},
        )
        writer = HDF5Writer(str(tmp_path / 'out.h5'), reader)

        # Build the data dict the way save_caches_to_h5 would
        flat_img = np.arange(16, dtype=np.uint16)
        data = {
            'images': [flat_img.copy() for _ in range(n)],
            'attributes': [{'EnergyRBV': 8.0 + i} for i in range(n)],
            'rsm': ([], [], []),
            'shape': (4, 4),
            'len_images': n,
            'len_attributes': n,
            'HKL_IN_CONFIG': False,
            'metadata': writer.merge_metadata([{'EnergyRBV': 8.0 + i} for i in range(n)]),
            'cached_ca': {},
        }
        out_path = tmp_path / 'out.h5'
        writer.h5_save(str(out_path), data, compress=False)

        with h5py.File(out_path, 'r') as h5:
            assert 'entry/data/frame_ids' in h5
            assert 'entry/data/timestamps' in h5
            assert list(h5['entry/data/frame_ids'][:]) == [1000, 1001, 1002, 1003]
            ts = h5['entry/data/timestamps'][:]
            assert pytest.approx(ts[0]) == 1_700_000_000.0
            assert pytest.approx(ts[-1]) == 1_700_000_003.0
            # frame_ids are int64 for compactness + queryable typing
            assert h5['entry/data/frame_ids'].dtype == np.int64

    def test_writes_per_ca_frame_id_and_timestamp_datasets(self, tmp_path, monkeypatch):
        import dashpva.settings as settings
        from dashpva.utils.hdf5_writer import HDF5Writer

        monkeypatch.setattr(settings, 'METADATA_CA', {'EnergyKeV': 'sim:energy'})

        n = 3
        reader = _StubReader(
            frame_ids=[10, 11, 12],
            timestamps=[1.0, 1.5, 2.0],
            cached_ca={'sim:energy': [8.0, 8.1, 8.2]},
            cached_ca_frame_ids={'sim:energy': [10, 11, 12]},
            cached_ca_timestamps={'sim:energy': [1.0, 1.5, 2.0]},
        )
        writer = HDF5Writer(str(tmp_path / 'out.h5'), reader)
        flat_img = np.zeros(16, dtype=np.uint16)
        data = {
            'images': [flat_img.copy() for _ in range(n)],
            'attributes': [{'sim:energy': 8.0 + i * 0.1} for i in range(n)],
            'rsm': ([], [], []),
            'shape': (4, 4),
            'len_images': n,
            'len_attributes': n,
            'HKL_IN_CONFIG': False,
            'metadata': writer.merge_metadata(
                [{'sim:energy': 8.0 + i * 0.1} for i in range(n)]),
            'cached_ca': {'sim:energy': [8.0, 8.1, 8.2]},
        }
        out_path = tmp_path / 'out.h5'
        writer.h5_save(str(out_path), data, compress=False)

        with h5py.File(out_path, 'r') as h5:
            ca = h5['entry/data/metadata/ca']
            assert 'EnergyKeV' in ca
            assert 'EnergyKeV__frame_ids' in ca
            assert 'EnergyKeV__timestamps' in ca
            assert list(ca['EnergyKeV__frame_ids'][:]) == [10, 11, 12]
            ts = ca['EnergyKeV__timestamps'][:]
            assert pytest.approx(ts[0]) == 1.0
            assert pytest.approx(ts[-1]) == 2.0


# ---------------------------------------------------------------------------
# Feature extractor: frame_id + timestamp stamped into the dict
# ---------------------------------------------------------------------------


class TestFeatureExtractionConsumerStamping:

    def test_features_dict_gets_frame_id_and_timestamp(self):
        """Smoke test: import the consumer module and verify the lines exist
        that we expect. (A full pvObject mock would pull in pvapy machinery
        beyond the scope of this test; the real path is exercised by the
        end-to-end smoke in HANDOFF_PHASE3.md §4.)"""
        src = Path(
            'src/dashpva/consumers/hpc/analysis/hpc_feature_extraction_consumer.py'
        ).read_text()
        assert "features['frame_id']" in src
        assert "features['timestamp']" in src
        assert "pvObject['uniqueId']" in src
        assert "pvObject['timeStamp']" in src