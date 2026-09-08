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

import bitshuffle
import blosc2
import lz4.block
import numpy as np
import pytest

pytest.importorskip('pvaccess')
from dashpva.utils import pva_reader


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.setattr(pva_reader.pva, 'Channel', lambda *_args: SimpleNamespace())
    monkeypatch.setattr(pva_reader.PVAReader, '_configure', lambda _self: None)
    result = pva_reader.PVAReader('test:image')
    result.analysis_cache_dict = {name: {} for name in ('Position', 'Intensity', 'ComX', 'ComY')}
    return result


def frame(frame_id=1, attrs=(), codec=''):
    pixels = np.arange(24, dtype=np.uint16)
    payload = pixels
    if codec == 'lz4':
        payload = np.frombuffer(lz4.block.compress(pixels.tobytes(), store_size=False), dtype=np.uint8)
    elif codec == 'blosc':
        payload = np.frombuffer(blosc2.compress(pixels.tobytes()), dtype=np.uint8)
    elif codec == 'bslz4':
        payload = bitshuffle.compress_lz4(pixels)
    return {
        'uniqueId': frame_id,
        'dimension': [{'size': 4}, {'size': 6}],
        'value': [{'ubyteValue' if codec else 'ushortValue': payload}],
        'codec': {'name': codec, 'parameters': [{'value': pva_reader.pva.USHORT}]},
        'uncompressedSize': pixels.nbytes,
        'timeStamp': {'secondsPastEpoch': frame_id, 'nanoseconds': 0},
        'attribute': [{'name': name, 'value': [{'value': value}]} for name, value in attrs],
    }


def rsm(size=24):
    return {'codec': {'name': ''}, **{axis: {'value': np.arange(size)} for axis in ('qx', 'qy', 'qz')}}


@pytest.mark.parametrize('codec', ['', 'lz4', 'blosc', 'bslz4'])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_valid_decode_preserves_nonsquare_pixel_layout(reader, codec, order):
    reader.pixel_ordering = order
    calls = []
    reader.reader_new_frame.connect(lambda: calls.append(True))
    reader.pva_callbackSuccess(frame(codec=codec))
    np.testing.assert_array_equal(reader.image, np.arange(24).reshape((4, 6), order=order))
    assert calls == [True]
    assert reader.processing_errors == 0


def test_missing_rsm_does_not_reuse_previous_coordinates(reader):
    reader.HKL_IN_CONFIG = True
    reader.pva_callbackSuccess(frame(attrs=[('RSM', rsm())]))
    assert reader.rsm_attributes
    reader.pva_callbackSuccess(frame(2))
    assert reader.rsm_attributes == {}
    assert reader.image is not None


@pytest.mark.parametrize('failure', ['shape', 'codec', 'rsm'])
def test_bad_frame_is_counted_not_published_and_next_frame_recovers(reader, failure):
    reader.HKL_IN_CONFIG = True
    calls = []
    reader.reader_new_frame.connect(lambda: calls.append(reader.last_array_id))
    reader.pva_callbackSuccess(frame())
    broken = frame(2, [('RSM', rsm(3))] if failure == 'rsm' else [])
    if failure == 'shape':
        broken['dimension'][0]['size'] = 17
    elif failure == 'codec':
        broken['codec']['name'] = 'unsupported'
    reader.pva_callbackSuccess(broken)
    assert reader.processing_errors == 1
    assert reader.image is None
    assert reader.rsm_attributes == {}
    assert calls == [1]
    reader.pva_callbackSuccess(frame(3))
    assert calls == [1, 3]


def test_continuous_analysis_uses_actual_attribute_index_and_validates_before_accumulating(reader):
    reader.ANALYSIS_IN_CONFIG = True
    reader.CONSUMER_MODE = 'continuous'
    result = dict(Axis1=2., Axis2=3., Intensity=4., ComX=5., ComY=6.)
    reader.pva_callbackSuccess(frame(attrs=[('unrelated', 9), ('Analysis', result)]))
    assert reader.analysis_index == 1
    assert reader.analysis_attributes == result
    assert reader.analysis_cache_dict['Position'] == {(2., 3.): (2., 3.)}
    assert reader.analysis_cache_dict['Intensity'] == {(2., 3.): 4.}
    reader.pva_callbackSuccess(frame(2, [('Analysis', {**result, 'ComY': np.nan})]))
    assert reader.processing_errors == 1
    assert reader.analysis_cache_dict['Intensity'] == {(2., 3.): 4.}
    reader.pva_callbackSuccess(frame(3))
    assert reader.analysis_index is None
    assert reader.analysis_attributes == {}


def test_vectorized_analysis_accepts_matching_arrays(reader):
    reader.ANALYSIS_IN_CONFIG = True
    reader.CONSUMER_MODE = 'vectorized'
    result = {key: np.arange(3.) for key in ('Intensity', 'ComX', 'ComY')}
    reader.pva_callbackSuccess(frame(attrs=[('Analysis', result)]))
    assert reader.processing_errors == 0
    assert reader.analysis_index == 0
    assert reader.attributes[0]['name'] == 'Analysis'
    reader.pva_callbackSuccess(frame(2, [('Analysis', {**result, 'ComY': np.arange(2.)})]))
    assert reader.processing_errors == 1
