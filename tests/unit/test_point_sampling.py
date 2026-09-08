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
import pytest

from dashpva.utils.point_sampling import (
    evenly_spaced_indices,
    sampled_point_cloud,
    sampled_point_cloud_chunks,
)


def test_evenly_spaced_indices_respect_budget_and_endpoints():
    indices = evenly_spaced_indices(16_777_216, 10_000)
    assert len(indices) == 10_000
    assert indices[0] == 0
    assert indices[-1] == 16_777_215
    assert np.all(np.diff(indices) > 0)


def test_evenly_spaced_indices_keep_small_inputs():
    np.testing.assert_array_equal(evenly_spaced_indices(4, 10), np.arange(4))


def test_sampled_point_cloud_preserves_correspondence_before_conversion():
    intensity = np.arange(100, dtype=np.uint32)
    points, sampled_intensity = sampled_point_cloud(
        intensity,
        intensity + 100,
        intensity + 200,
        intensity + 300,
        budget=7,
    )
    indices = evenly_spaced_indices(100, 7)
    np.testing.assert_array_equal(sampled_intensity, intensity[indices])
    np.testing.assert_array_equal(points[:, 0], intensity[indices] + 100)
    np.testing.assert_array_equal(points[:, 1], intensity[indices] + 200)
    np.testing.assert_array_equal(points[:, 2], intensity[indices] + 300)
    assert points.dtype == np.float32
    assert sampled_intensity.dtype == np.float32


def test_sampled_point_cloud_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match="matching sizes"):
        sampled_point_cloud(np.arange(3), np.arange(2), np.arange(3), np.arange(3), 3)


def test_chunked_sampling_matches_contiguous_sampling():
    intensity = np.arange(20, dtype=np.uint32)
    chunks = [intensity[:3], intensity[3:11], intensity[11:]]
    expected = sampled_point_cloud(
        intensity, intensity + 100, intensity + 200, intensity + 300, 7
    )
    actual = sampled_point_cloud_chunks(
        chunks,
        [chunk + 100 for chunk in chunks],
        [chunk + 200 for chunk in chunks],
        [chunk + 300 for chunk in chunks],
        7,
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_chunked_sampling_rejects_mismatched_chunk_sizes():
    with pytest.raises(ValueError, match="chunk sizes"):
        sampled_point_cloud_chunks(
            [np.arange(3)], [np.arange(2)], [np.arange(3)], [np.arange(3)], 3
        )


def test_fortran_sampling_does_not_allocate_a_full_image_copy():
    import tracemalloc
    image = np.asfortranarray(np.arange(2048 * 2048, dtype=np.uint32).reshape(2048, 2048))
    tracemalloc.start()
    try:
        points, values = sampled_point_cloud(image, image, image, image, 1000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000
    indices = evenly_spaced_indices(image.size, 1000)
    expected = image[indices // image.shape[1], indices % image.shape[1]].astype(np.float32)
    np.testing.assert_array_equal(values, expected)
    for column in range(3):
        np.testing.assert_array_equal(points[:, column], expected)


def test_nonsquare_rotated_chunk_sampling_keeps_intensity_coordinate_pairing():
    image = np.rot90(np.arange(35).reshape(5, 7))
    points, values = sampled_point_cloud_chunks([image], [image + 1], [image + 2], [image + 3], 6)
    np.testing.assert_array_equal(points, np.column_stack([values + 1, values + 2, values + 3]))
