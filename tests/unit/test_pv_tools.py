"""Tests for PvTools — allowlist, friendly-name resolution, cached_ca short-circuit, history."""

from __future__ import annotations

import types

import pytest

from dashpva.analysis.tools.pv_tools import PvTools


def _settings(metadata_ca=None, extra_prefixes=None, ioc='sim:', detector=None):
    return types.SimpleNamespace(
        METADATA_CA=metadata_ca or {},
        IOC_PREFIX=ioc,
        DETECTOR_PREFIX=detector,
        CHAT_TOOLS={
            'EXTRA_PV_PREFIXES': extra_prefixes or [],
            'TOOL_TIMEOUT_S': 5.0,
            'HISTORY_MAX_POINTS': 500,
        },
    )


class _Reader:
    def __init__(self, cached_ca=None, cached_ca_frame_ids=None,
                 cached_ca_timestamps=None):
        self.cached_ca = cached_ca or {}
        self.cached_ca_frame_ids = cached_ca_frame_ids or {}
        self.cached_ca_timestamps = cached_ca_timestamps or {}
        self.cached_attributes = None
        self.cached_frame_ids = None
        self.cached_timestamps = None


class TestAllowlist:

    def test_rejects_out_of_allowlist(self):
        pt = PvTools(_Reader(), _settings(ioc='sim:'))
        r = pt.read_pv('6idb1:m1')   # not under sim:
        assert 'error' in r
        assert 'allowlist' in r['error']

    def test_allows_ioc_prefix(self, monkeypatch):
        pt = PvTools(_Reader(), _settings(ioc='sim:'))
        called = {}

        def fake_caget(name, timeout=None):
            called['name'] = name
            return 42.0

        monkeypatch.setattr('epics.caget', fake_caget, raising=False)
        r = pt.read_pv('sim:m1')
        assert r.get('value') == 42.0
        assert called['name'] == 'sim:m1'

    def test_detector_prefix_normalized_with_colon(self):
        pt = PvTools(_Reader(), _settings(ioc=None, detector='13SIM1'))
        assert '13SIM1:' in pt._allowed_prefixes()

    def test_extra_prefixes_allowed(self, monkeypatch):
        pt = PvTools(_Reader(), _settings(ioc='sim:', extra_prefixes=['6idb1:']))
        monkeypatch.setattr('epics.caget', lambda name, timeout=None: 1.0, raising=False)
        r = pt.read_pv('6idb1:m1')
        assert r.get('value') == 1.0

    def test_friendly_mapped_pv_always_allowed(self, monkeypatch):
        # friendly maps to a PV outside the prefix allowlist — still allowed.
        pt = PvTools(_Reader(), _settings(metadata_ca={'energy': 'XYZ:energy'}, ioc='sim:'))
        monkeypatch.setattr('epics.caget', lambda name, timeout=None: 8.0, raising=False)
        r = pt.read_pv('energy')   # friendly name
        assert r.get('value') == 8.0


class TestReadPvCachedShortCircuit:

    def test_cached_ca_returns_latest_without_caget(self, monkeypatch):
        reader = _Reader(cached_ca={'sim:energy': [8.0, 8.1, 8.2]})
        pt = PvTools(reader, _settings(metadata_ca={'energy': 'sim:energy'}, ioc='sim:'))

        def boom(*a, **k):
            raise AssertionError('caget should not be called when cached')

        monkeypatch.setattr('epics.caget', boom, raising=False)
        r = pt.read_pv('energy')
        assert r['value'] == 8.2
        assert r['source'] == 'cached_ca'
        assert r['samples_in_session'] == 3


class TestReadPvErrors:

    def test_caget_none_is_error(self, monkeypatch):
        pt = PvTools(_Reader(), _settings(ioc='sim:'))
        monkeypatch.setattr('epics.caget', lambda name, timeout=None: None, raising=False)
        r = pt.read_pv('sim:dead')
        assert 'error' in r

    def test_caget_exception_wrapped(self, monkeypatch):
        pt = PvTools(_Reader(), _settings(ioc='sim:'))

        def raises(name, timeout=None):
            raise RuntimeError('no CA')

        monkeypatch.setattr('epics.caget', raises, raising=False)
        r = pt.read_pv('sim:x')
        assert 'error' in r
        assert 'RuntimeError' in r['error']


class TestHistoryTools:

    def _pt(self):
        reader = _Reader(
            cached_ca={'sim:energy': [8.0, 8.1, 8.3]},
            cached_ca_frame_ids={'sim:energy': [100, 101, 103]},
            cached_ca_timestamps={'sim:energy': [1.0, 1.1, 1.3]},
        )
        return PvTools(reader, _settings(metadata_ca={'energy': 'sim:energy'}, ioc='sim:'))

    def test_read_pv_at_frame_live(self):
        r = self._pt().read_pv_at_frame('energy', 101)
        assert r['value'] == 8.1
        assert r['exact'] is True

    def test_read_pv_at_frame_nearest_flag(self):
        r = self._pt().read_pv_at_frame('energy', 102)
        assert r['exact'] is False

    def test_read_pv_history_capped(self):
        r = self._pt().read_pv_history('energy', 0, 1000, by='frame')
        assert r['n_points'] == 3

    def test_get_pv_summary(self):
        r = self._pt().get_pv_summary('energy', 0, 1000, by='frame')
        assert r['n'] == 3
        assert pytest.approx(r['min']) == 8.0

    def test_h5_source_without_file_errors(self):
        r = self._pt().read_pv_at_frame('energy', 101, source='h5')
        assert 'error' in r
        assert 'history file' in r['error']

    def test_history_tool_respects_allowlist(self):
        r = self._pt().read_pv_at_frame('6idb1:secret', 101)
        assert 'error' in r
        assert 'allowlist' in r['error']


class TestListKnownPvs:

    def test_shape(self):
        pt = PvTools(_Reader(), _settings(metadata_ca={'energy': 'sim:energy'}, ioc='sim:'))
        r = pt.list_known_pvs()
        assert r['friendly_names'] == {'energy': 'sim:energy'}
        assert 'sim:' in r['allowed_prefixes']
        assert r['history_file_loaded'] is False
