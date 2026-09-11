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

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType

import numpy as np


def _array_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_array_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(item) for item in value)
    return 0


def _freeze(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays cannot be published as detector data')
        owner = value
        while isinstance(owner, np.ndarray) and owner.base is not None:
            owner = owner.base
        if isinstance(owner, bytes):
            return value
        order = 'F' if value.flags.f_contiguous else 'C'
        return np.frombuffer(value.tobytes(order=order), dtype=value.dtype).reshape(value.shape, order=order)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class FramePacket:
    stream_epoch: int
    sequence: int
    unique_id: int | None
    source_timestamp: float | None
    dequeued_monotonic: float
    published_monotonic: float
    image: np.ndarray
    shape: tuple[int, ...]
    pixel_ordering: str
    attributes: Mapping
    rsm_attributes: Mapping
    fallback_channels: tuple[str, ...]
    geometry_revision: str | None

    @property
    def identity(self):
        return self.stream_epoch, self.sequence

    @classmethod
    def capture(cls, *, max_array_bytes, **values):
        if _array_bytes(values) > max_array_bytes:
            raise ValueError('Decoded preview exceeds the configured array byte budget')
        return cls(**{key: _freeze(value) for key, value in values.items()})


class LatestFrame:
    """One owned packet and at most one outstanding payload-free GUI wakeup."""

    def __init__(self):
        self._lock = Lock()
        self._frame = None
        self._pending = False
        self.epoch = 0
        self.superseded = 0

    def reset(self):
        with self._lock:
            self.epoch += 1
            self._frame = None
            self._pending = False
            return self.epoch

    def publish(self, frame):
        with self._lock:
            if frame.stream_epoch != self.epoch:
                return False
            notify = not self._pending
            if self._pending:
                self.superseded += 1
            self._frame = frame
            self._pending = True
            return notify

    def peek(self):
        with self._lock:
            return self._frame

    def take(self):
        with self._lock:
            self._pending = False
            return self._frame
