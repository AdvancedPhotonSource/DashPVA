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

from pathlib import Path

import lz4.block
import numpy as np
import pytest

pva = pytest.importorskip('pvaccess')
from dashpva.consumers.hpc.analysis.hpc_rsm_consumer import HpcRsmProcessor


@pytest.fixture
def processor(monkeypatch):
    profile = Path(__file__).parents[2] / 'pv_configs' / 'sample_config.toml'
    result = HpcRsmProcessor({'path': str(profile)})
    result.published = []
    result.updateOutputChannel = result.published.append
    result.parse_hkl_ndattributes = lambda _frame: {}
    result.create_rsm = lambda _attrs, shape: tuple(np.arange(np.prod(shape), dtype=np.float64) for _ in range(3))
    result.decompress_image = lambda _frame: pytest.fail('RSM must not decode unused detector pixels')
    return result


def frame(codec='', shape=(4, 6)):
    result = pva.NtNdArray()
    pixels = np.arange(np.prod(shape), dtype=np.uint16)
    payload = np.frombuffer(lz4.block.compress(pixels.tobytes(), store_size=False), dtype=np.uint8) if codec else pixels
    result['value'] = ({'ubyteValue' if codec else 'ushortValue': payload},)
    result['codec'] = {'name': codec, 'parameters': pva.PvInt(int(pva.USHORT))}
    result['uncompressedSize'] = pixels.nbytes
    result['compressedSize'] = payload.nbytes
    result['dimension'] = [{'size': size, 'fullSize': size, 'offset': 0, 'binning': 1, 'reverse': False} for size in shape]
    result['uniqueId'] = 42
    return result


@pytest.mark.parametrize('codec', ['', 'lz4'])
def test_published_rsm_preserves_actual_ntndarray_image_and_codec(processor, codec):
    incoming = frame(codec)
    original = incoming.toDict()
    processor.process(incoming)
    assert processor.nFrameErrors == 0
    assert processor.published == [incoming]
    actual = incoming.toDict()
    for key in ('codec', 'dimension', 'uniqueId', 'compressedSize', 'uncompressedSize'):
        assert actual[key] == original[key]
    dtype = next(iter(original['value'][0]))
    np.testing.assert_array_equal(actual['value'][0][dtype], original['value'][0][dtype])
    rsm = next(attr['value'][0]['value'] for attr in actual['attribute'] if attr['name'] == 'RSM')
    for axis in ('qx', 'qy', 'qz'):
        values = rsm[axis]['value']
        if codec:
            values = np.frombuffer(lz4.block.decompress(values, uncompressed_size=rsm[axis]['uncompressedSize']), dtype=np.float64)
        np.testing.assert_array_equal(values, np.arange(24.))


def test_failed_geometry_clears_old_coordinates_and_retries_same_metadata(processor):
    processor.process(frame())
    compute = processor.create_rsm
    processor.create_rsm = lambda *_args: None
    processor.process(frame(shape=(3, 6)))
    assert processor.qx is None
    assert processor.old_attrbutes is None
    processor.create_rsm = compute
    processor.process(frame(shape=(3, 6)))
    assert processor.qx.size == 18
    assert len(processor.published) == 3


def test_shape_and_codec_changes_invalidate_cached_q(processor):
    processor.process(frame())
    processor.process(frame('lz4', (3, 6)))
    assert processor.codec_name == 'lz4'
    assert processor.uncompressed_size == 18 * 8
    processor.process(frame('', (3, 6)))
    assert processor.codec_name == ''
    assert processor.qx.size == 18
