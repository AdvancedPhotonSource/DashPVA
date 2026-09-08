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

"""CPU/allocation proxy only: no acquisition, decoder, Qt, VTK or paint timing."""
import argparse
import json
import platform
import statistics
import time
import tracemalloc

import numpy as np

from dashpva.utils.point_sampling import evenly_spaced_indices, sampled_point_cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--points', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if min(args.size, args.points, args.repeats) <= 0:
        parser.error('size, points and repeats must be positive')
    image = np.asfortranarray(np.arange(args.size ** 2, dtype=np.uint32).reshape(args.size, args.size))
    coordinates = np.arange(image.size, dtype=np.float64)
    indices = evenly_spaced_indices(image.size, args.points)

    def before():
        intensity = np.ravel(image).astype(np.float32)
        qx, qy, qz = (coordinates.astype(np.float32) for _ in range(3))
        return np.column_stack((qx[indices], qy[indices], qz[indices])), intensity[indices]

    def after():
        return sampled_point_cloud(image, coordinates, coordinates, coordinates, args.points)

    for old, new in zip(before(), after()):
        np.testing.assert_array_equal(old, new)
    result = {'scope': __doc__, 'python': platform.python_version(), 'numpy': np.__version__,
              'platform': platform.platform(), 'shape': list(image.shape), 'points': len(indices),
              'repeats': args.repeats, 'selected_values_equal': True}
    for name, operation in [('full_convert_then_sample', before), ('sample_before_convert', after)]:
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            operation()
            times.append((time.perf_counter() - start) * 1000)
        tracemalloc.start()
        operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result[name] = {'median_ms': statistics.median(times), 'temporary_peak_bytes': peak}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
