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
