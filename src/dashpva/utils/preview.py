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

import dashpva.settings as app_settings


def sampled_percentiles(
    image: np.ndarray,
    percentiles: tuple[float, float] = (5.0, 95.0),
    max_samples_per_axis: int | None = None,
) -> tuple[float, float] | None:
    """Estimate display levels from a bounded, deterministic image sample."""
    if max_samples_per_axis is None:
        max_samples_per_axis = app_settings.PREVIEW['AUTOSCALE_SAMPLES_PER_AXIS']
    if max_samples_per_axis <= 0:
        raise ValueError('autoscale sample budget must be positive')
    if image.ndim != 2:
        raise ValueError("display autoscale requires a 2-D image")
    row_stride = max(1, int(np.ceil(image.shape[0] / max_samples_per_axis)))
    column_stride = max(1, int(np.ceil(image.shape[1] / max_samples_per_axis)))
    sample = image[::row_stride, ::column_stride].ravel(order="K")
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, percentiles)
    return float(low), float(high)


def plot_interval_ms(rate):
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('Plotting rate must be finite and positive')
    return max(app_settings.PREVIEW['MIN_TIMER_INTERVAL_MS'], int(1000 / rate))
