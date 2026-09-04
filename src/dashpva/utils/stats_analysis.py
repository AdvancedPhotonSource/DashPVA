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


def calculate_1d_analysis(positions, intensities):
    """
    Compute Peak, Center-of-Mass, and FWHM from 1D intensity data.

    Parameters
    ----------
    positions : array-like
        X-axis values (e.g., frame indices or motor positions).
    intensities : array-like
        Corresponding intensity/stat values.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - peak_pos, peak_intensity
        - com_pos, com_intensity
        - fwhm_value, fwhm_center, fwhm_center_intensity
        - fwhm_left, fwhm_right
        - half_max, baseline_intensity
        Returns None if input is empty or invalid.
    """
    if len(positions) == 0 or len(intensities) == 0:
        return None

    positions = np.asarray(positions, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)

    if len(positions) != len(intensities):
        return None

    # Peak
    peak_idx = np.argmax(intensities)
    peak_pos = positions[peak_idx]
    peak_intensity = intensities[peak_idx]

    # Center of Mass
    total_intensity = np.sum(intensities)
    if total_intensity == 0:
        return None

    com_pos = np.sum(positions * intensities) / total_intensity
    com_intensity = np.interp(com_pos, positions, intensities)

    # FWHM
    baseline_intensity = np.min(intensities)
    half_max = baseline_intensity + (peak_intensity - baseline_intensity) / 2.0

    above_half_max = intensities >= half_max
    indices_above = np.where(above_half_max)[0]

    if len(indices_above) == 0:
        fwhm_value = 0.0
        fwhm_center = peak_pos
        fwhm_center_intensity = peak_intensity
        fwhm_left = peak_pos
        fwhm_right = peak_pos
    else:
        # Left edge interpolation
        left_idx = indices_above[0]
        if left_idx > 0:
            x1, x2 = positions[left_idx - 1], positions[left_idx]
            y1, y2 = intensities[left_idx - 1], intensities[left_idx]
            fwhm_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
        else:
            fwhm_left = positions[left_idx]

        # Right edge interpolation
        right_idx = indices_above[-1]
        if right_idx < len(positions) - 1:
            x1, x2 = positions[right_idx], positions[right_idx + 1]
            y1, y2 = intensities[right_idx], intensities[right_idx + 1]
            fwhm_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x2
        else:
            fwhm_right = positions[right_idx]

        fwhm_value = fwhm_right - fwhm_left
        fwhm_center = (fwhm_left + fwhm_right) / 2.0
        fwhm_center_intensity = np.interp(fwhm_center, positions, intensities)

    return {
        'peak_pos': peak_pos,
        'peak_intensity': peak_intensity,
        'baseline_intensity': baseline_intensity,
        'com_pos': com_pos,
        'com_intensity': com_intensity,
        'fwhm_value': fwhm_value,
        'fwhm_center': fwhm_center,
        'fwhm_center_intensity': fwhm_center_intensity,
        'fwhm_left': fwhm_left,
        'fwhm_right': fwhm_right,
        'half_max': half_max,
    }
