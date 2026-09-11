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

import gc
import weakref
from threading import Thread

import numpy as np
import pytest

from dashpva.utils.frame_delivery import FramePacket, LatestFrame


def packet(sequence=1, epoch=0, image=None, **overrides):
    values = dict(
        stream_epoch=epoch, sequence=sequence, unique_id=7,
        source_timestamp=42.25, dequeued_monotonic=1., published_monotonic=1.5,
        image=np.arange(6).reshape(2, 3) if image is None else image, shape=(2, 3),
        pixel_ordering='F', attributes={'timeStamp-secondsPastEpoch': 42},
        rsm_attributes={}, fallback_channels=(), geometry_revision=None,
        max_array_bytes=1024,
    )
    values.update(overrides)
    return FramePacket.capture(**values)


def test_packet_owns_writable_input_and_nested_metadata():
    raw = np.asfortranarray(np.arange(6).reshape(2, 3))
    meta = {'motor': {'values': np.arange(2.)}, 'labels': ['H', 'K']}
    result = packet(image=raw, attributes=meta)
    raw[:] = -1
    meta['motor']['values'][:] = -2
    meta['labels'].append('L')
    np.testing.assert_array_equal(result.image, np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(result.attributes['motor']['values'], np.arange(2.))
    assert result.image.flags.f_contiguous
    assert result.attributes['labels'] == ('H', 'K')
    with pytest.raises(ValueError):
        result.image.setflags(write=True)
    with pytest.raises(ValueError):
        result.attributes['motor']['values'][0] = 5
    with pytest.raises(TypeError):
        result.attributes['motor']['new'] = 5


def test_packet_records_source_and_local_time_domains():
    result = packet()
    assert result.source_timestamp == 42.25
    assert result.dequeued_monotonic == 1.
    assert result.published_monotonic == 1.5
    assert result.shape == (2, 3)


def test_immutable_decode_buffer_can_be_retained_without_another_copy():
    raw = np.frombuffer(bytes(range(6)), dtype=np.uint8).reshape(2, 3)
    result = packet(image=raw)
    assert np.shares_memory(raw, result.image)
    with pytest.raises(ValueError):
        result.image.setflags(write=True)


def test_byte_budget_includes_coordinates_and_nested_arrays():
    with pytest.raises(ValueError, match='budget'):
        packet(max_array_bytes=60, rsm_attributes={'qx': np.arange(3.)})


def test_overload_has_one_wakeup_and_releases_superseded_packets():
    slot = LatestFrame()
    old = packet()
    reference = weakref.ref(old)
    assert slot.publish(old)
    del old
    for sequence in range(2, 1001):
        assert not slot.publish(packet(sequence))
    gc.collect()
    assert reference() is None
    assert slot.superseded == 999
    assert slot.take().sequence == 1000
    assert slot.publish(packet(1001))


def test_reset_rejects_inflight_result_even_if_ids_are_reused():
    slot = LatestFrame()
    old = packet()
    epoch = slot.reset()
    assert not slot.publish(old)
    assert slot.peek() is None
    current = packet(epoch=epoch)
    assert slot.publish(current)
    assert current.identity != old.identity


def test_concurrent_publication_and_consumption_never_tear_packets():
    slot = LatestFrame()
    def publish():
        for sequence in range(1, 501):
            slot.publish(packet(sequence, image=np.full((2, 3), sequence)))
    producer = Thread(target=publish)
    producer.start()
    seen = []
    while producer.is_alive():
        frame = slot.take()
        if frame is not None:
            assert np.all(frame.image == frame.sequence)
            seen.append(frame.sequence)
    producer.join()
    assert slot.take().sequence == 500
    assert seen == sorted(seen)
