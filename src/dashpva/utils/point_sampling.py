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

import numpy as np


def evenly_spaced_indices(size: int, budget: int) -> np.ndarray:
    """Return at most ``budget`` ordered indices spanning an input array."""
    if size < 0 or budget <= 0:
        raise ValueError("size must be non-negative and budget must be positive")
    if size <= budget:
        return np.arange(size, dtype=np.intp)
    return np.linspace(0, size - 1, num=budget, dtype=np.intp)


def sampled_point_cloud(
    intensity,
    qx,
    qy,
    qz,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select corresponding intensity and HKL values before float conversion."""
    arrays = [np.asarray(values) for values in (intensity, qx, qy, qz)]
    sizes = {array.size for array in arrays}
    if len(sizes) != 1:
        raise ValueError("intensity and HKL arrays must have matching sizes")
    indices = evenly_spaced_indices(arrays[0].size, budget)
    points = np.empty((indices.size, 3), dtype=np.float32)
    for column, values in enumerate(arrays[1:]):
        points[:, column] = values.flat[indices]
    return points, arrays[0].flat[indices].astype(np.float32, copy=False)


def sampled_point_cloud_chunks(
    intensity_chunks,
    qx_chunks,
    qy_chunks,
    qz_chunks,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample corresponding chunked arrays without concatenating full scans."""
    chunk_groups = [list(chunks) for chunks in (
        intensity_chunks,
        qx_chunks,
        qy_chunks,
        qz_chunks,
    )]
    chunk_counts = {len(chunks) for chunks in chunk_groups}
    if len(chunk_counts) != 1:
        raise ValueError("intensity and HKL chunk counts must match")
    sizes = [np.asarray(chunk).size for chunk in chunk_groups[0]]
    for chunks in chunk_groups[1:]:
        if [np.asarray(chunk).size for chunk in chunks] != sizes:
            raise ValueError("intensity and HKL chunk sizes must match")
    indices = evenly_spaced_indices(sum(sizes), budget)
    points = np.empty((indices.size, 3), dtype=np.float32)
    intensity = np.empty(indices.size, dtype=np.float32)
    offset = 0
    for chunk_index, size in enumerate(sizes):
        selected = (indices >= offset) & (indices < offset + size)
        if not np.any(selected):
            offset += size
            continue
        destination = np.flatnonzero(selected)
        local_indices = indices[selected] - offset
        intensity[destination] = np.asarray(
            chunk_groups[0][chunk_index]
        ).flat[local_indices]
        for column in range(3):
            points[destination, column] = np.asarray(
                chunk_groups[column + 1][chunk_index]
            ).flat[local_indices]
        offset += size
    return points, intensity
