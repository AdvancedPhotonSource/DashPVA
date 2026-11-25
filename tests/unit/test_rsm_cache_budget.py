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

"""HKL3D frame caches are bounded by bytes, not only by a frame count."""

from collections import deque

import numpy as np

import dashpva.settings as app_settings
from dashpva.utils.pva_reader import PVAReader


def _reader(viewer_type, max_frames=1000):
    reader = PVAReader.__new__(PVAReader)
    reader.VIEWER_TYPE_MAP = {"image": "i", "analysis": "a", "rsm": "r"}
    reader.viewer_type = viewer_type
    reader.CACHING_MODE = "alignment"
    reader.MAX_CACHE_SIZE = max_frames
    reader.cache_limit_notice = ""
    for name in ("cached_images", "cached_attributes", "cached_qx", "cached_qy", "cached_qz"):
        setattr(reader, name, deque(maxlen=max_frames))
    reader.image = np.zeros(100, dtype=np.uint16)
    return reader


def _q():
    return {axis: np.zeros(100, dtype=np.float64) for axis in ("qx", "qy", "qz")}


def test_rsm_cache_is_capped_by_the_byte_budget(monkeypatch):
    frame_bytes = 100 * 2 + 3 * 100 * 8
    monkeypatch.setattr(app_settings, "RSM_CACHE_MAX_BYTES", frame_bytes * 5)
    reader = _reader("r")
    for _ in range(20):
        if reader.cache_attributes({}, _q()):
            reader.cache_image(reader.image)

    assert reader.MAX_CACHE_SIZE == 5
    assert len(reader.cached_qx) == len(reader.cached_images) == 5
    assert "limited to 5 frames" in reader.cache_limit_notice


def test_non_rsm_viewers_keep_their_frame_count(monkeypatch):
    monkeypatch.setattr(app_settings, "RSM_CACHE_MAX_BYTES", 1)
    reader = _reader("i")
    reader.cache_attributes({}, _q())
    assert reader.MAX_CACHE_SIZE == 1000
    assert reader.cache_limit_notice == ""
