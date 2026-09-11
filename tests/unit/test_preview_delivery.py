# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

from types import SimpleNamespace

import numpy as np
import pytest

pva = pytest.importorskip('pvaccess')
from dashpva import settings
from dashpva.utils.preview import plot_interval_ms, sampled_percentiles
from tests.unit.test_reader_frame_errors import frame, reader  # noqa: F401


@pytest.mark.parametrize('preview, expected', [(True, [3]), (False, [1, 2, 3])])
def test_native_queue_preview_drains_occupancy_and_ordered_keeps_every_frame(reader, preview, expected):
    queue = pva.PvObjectQueue(3)
    for number in (1, 2, 3):
        queue.put(pva.PvInt(number))
    reader._queue = queue
    reader._preview_delivery = preview
    reader._preview_policy['QUEUE_FRAMES'] = 2
    reader._monitor_caching_mode = reader.CACHING_MODE
    reader._consuming = True
    seen = []
    def consume(value):
        seen.append(value['value'])
        if len(queue) == 0:
            reader._consuming = False
    reader._process_callback = consume
    reader._consume_loop()
    assert seen == expected
    assert reader.preview_frames_superseded == (2 if preview else 0)
    assert len(queue) == 0


def test_explicit_preview_cannot_feed_science_or_custom_callbacks(reader):
    reader.delivery_mode = 'preview'
    reader.CACHING_MODE = 'scan'
    with pytest.raises(ValueError, match='scientific'):
        reader.start_channel_monitor()
    reader.CACHING_MODE = ''
    with pytest.raises(ValueError, match='custom'):
        reader.start_channel_monitor(lambda value: None)


def test_mode_change_stops_instead_of_silently_switching_delivery(reader):
    calls = []
    reader.channel = SimpleNamespace(stopMonitor=lambda: calls.append('stop'))
    reader._queue = pva.PvObjectQueue(2)
    reader._monitor_caching_mode = ''
    reader.CACHING_MODE = 'scan'
    reader._consuming = True
    reader._consume_loop()
    assert calls == ['stop']
    assert not reader._consuming


def test_scientific_processing_is_not_rejected_by_preview_budget(reader):
    reader._preview_policy['MAX_ARRAY_BYTES'] = 1
    cached = []
    reader.caches_initialized = True
    reader.cache_attributes = lambda *args: True
    reader.cache_image = cached.append
    reader.pva_callbackSuccess(frame())
    assert len(cached) == 1
    assert reader.preview_frames_rejected == 1
    assert reader.processing_errors == 0
    assert reader.latest_frame is None


def test_static_ca_fallback_is_labelled_in_owned_packet(reader):
    reader.hkl_values = {'motor': 99., 'energy': 12.}
    reader.pva_callbackSuccess(frame(attrs=[('motor', 5.)]))
    snapshot = reader.take_latest_frame()
    assert snapshot.attributes['motor'] == 5.
    assert snapshot.fallback_channels == ('energy',)
    assert 'timeStamp-secondsPastEpoch' in snapshot.attributes
    reader.hkl_values['energy'] = 20.
    assert snapshot.attributes['energy'] == 12.


def test_packet_records_source_timestamp_shape_and_bounded_error_categories(reader):
    reader.pva_callbackSuccess(frame())
    snapshot = reader.take_latest_frame()
    assert snapshot.source_timestamp == 1.
    assert snapshot.shape == (4, 6)
    assert snapshot.published_monotonic >= snapshot.dequeued_monotonic
    broken = frame(2)
    broken['codec']['name'] = 'unsupported'
    reader.pva_callbackSuccess(broken)
    metrics = reader.performance_snapshot()
    assert metrics['processing_errors'] == 1
    assert metrics['processing_error_counts'] == {'ValueError': 1}


def test_reader_snapshot_separates_decode_processing_and_preview_age(reader, monkeypatch):
    ticks = iter(value / 10 for value in range(10, 100))
    monkeypatch.setattr('dashpva.utils.pva_reader.time.monotonic', lambda: next(ticks))
    reader.pva_callbackSuccess(frame())
    reader.take_latest_frame()
    metrics = reader.performance_snapshot()
    assert metrics['decode_latency']['samples'] == 1
    assert metrics['processing_latency']['samples'] == 1
    assert metrics['preview_age_at_take']['samples'] == 1
    reader.reset_performance_metrics()
    assert reader.performance_snapshot()['decode_latency']['samples'] == 0


def test_stop_clears_latest_frame_without_configured_caches(reader):
    reader.pva_callbackSuccess(frame())
    reader.stop_channel_monitor()
    assert reader.latest_frame is None


def test_alive_worker_blocks_restart_and_cleanup(reader):
    reader._consumer_thread = SimpleNamespace(is_alive=lambda: True, join=lambda timeout: None)
    with pytest.raises(RuntimeError, match='still running'):
        reader.start_channel_monitor()
    with pytest.raises(RuntimeError, match='still stopping'):
        reader.stop_channel_monitor()
    assert reader._consumer_thread is not None


@pytest.mark.parametrize('rate', [1, 14, 1001, 999999999])
def test_plot_timer_interval_is_positive(rate):
    assert plot_interval_ms(rate) >= 1


@pytest.mark.parametrize('rate', [0, -1, np.nan, np.inf])
def test_plot_timer_rejects_invalid_rates(rate):
    with pytest.raises(ValueError):
        plot_interval_ms(rate)


def test_autoscale_is_bounded_and_resolves_current_settings(monkeypatch):
    monkeypatch.setitem(settings.PREVIEW, 'AUTOSCALE_SAMPLES_PER_AXIS', 4)
    original = np.percentile
    sizes = []
    def percentile(values, limits):
        sizes.append(values.size)
        return original(values, limits)
    monkeypatch.setattr(np, 'percentile', percentile)
    assert sampled_percentiles(np.arange(10000.).reshape(100, 100)) is not None
    assert sizes == [16]
    monkeypatch.setitem(settings.PREVIEW, 'AUTOSCALE_SAMPLES_PER_AXIS', 2)
    sampled_percentiles(np.arange(10000.).reshape(100, 100))
    assert sizes == [16, 4]


@pytest.mark.parametrize('value', [0, -1, True, 3.5, '2'])
def test_invalid_profile_budgets_are_rejected(value):
    with pytest.raises(ValueError, match='positive integer'):
        settings.preview_settings({'PREVIEW': {'QUEUE_FRAMES': value}})


def test_preview_defaults_round_trip_toml():
    import toml
    from dashpva.scripts.seed_profile_defaults_sql import get_default_profile_data
    assert settings.preview_settings(toml.loads(toml.dumps(get_default_profile_data()))) == settings.PREVIEW_DEFAULTS


def test_late_worker_callback_cannot_republish_after_stop(reader):
    old_epoch = reader._preview.epoch
    reader.stop_channel_monitor()
    reader.pva_callbackSuccess(frame(), stream_epoch=old_epoch)
    assert reader.latest_frame is None


def test_packet_identity_advances_when_source_reuses_an_id(reader):
    reader.pva_callbackSuccess(frame())
    first = reader.take_latest_frame()
    reader.pva_callbackSuccess(frame())
    second = reader.take_latest_frame()
    assert first.unique_id == second.unique_id
    assert first.identity != second.identity


@pytest.mark.parametrize('mode, cache_mode, custom, preview', [
    ('auto', '', False, True), ('ordered', '', False, False),
    ('auto', 'scan', False, False), ('auto', '', True, False),
])
def test_monitor_selects_short_queue_only_for_display_readers(reader, monkeypatch, mode, cache_mode, custom, preview):
    from dashpva.utils import pva_reader
    class Thread:
        def __init__(self, target, daemon):
            self.target = target
        def start(self):
            pass
        def is_alive(self):
            return False
        def join(self, timeout):
            pass
    monkeypatch.setattr(pva_reader.threading, 'Thread', Thread)
    requests = []
    reader.channel = SimpleNamespace(qMonitor=lambda queue, request: requests.append(request), stopMonitor=lambda: None)
    reader.delivery_mode = mode
    reader.CACHING_MODE = cache_mode
    reader.FLAG_PV = ''
    callback = (lambda value: None) if custom else None
    reader.start_channel_monitor(callback)
    assert reader._preview_delivery is preview
    expected = f"field() record[queueSize={settings.PREVIEW['QUEUE_FRAMES']}]" if preview else reader.MONITOR_REQUEST
    assert requests == [expected]
    reader.stop_channel_monitor()


def test_settings_reload_resolves_preview_override_from_active_source(monkeypatch):
    for name in tuple(key for key in vars(settings) if key.isupper()):
        monkeypatch.setattr(settings, name, getattr(settings, name))
    source = SimpleNamespace(load=lambda: {'PREVIEW': {'HKL_MAX_POINTS': 1234}}, ensure_path=lambda: None, source_type='db')
    monkeypatch.setattr(settings, 'ConfigSource', lambda locator: source)
    monkeypatch.setattr(settings, '_get_effective_locator', lambda: None)
    settings.reload()
    assert settings.PREVIEW['HKL_MAX_POINTS'] == 1234
    assert settings.PREVIEW['QUEUE_FRAMES'] == settings.PREVIEW_DEFAULTS['QUEUE_FRAMES']
