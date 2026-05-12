"""Tests for dashpva.utils.stats_analysis — scientific correctness of 1D analysis."""

import numpy as np
import pytest

from dashpva.utils.stats_analysis import calculate_1d_analysis


class TestCalculate1dAnalysis:

    def test_gaussian_peak_position(self):
        x = np.linspace(-5, 5, 1001)
        center = 1.5
        sigma = 0.3
        y = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        result = calculate_1d_analysis(x, y)
        assert result is not None
        assert abs(result["peak_pos"] - center) < 0.02

    def test_gaussian_com(self):
        x = np.linspace(-5, 5, 1001)
        center = 0.0
        sigma = 1.0
        y = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        result = calculate_1d_analysis(x, y)
        assert abs(result["com_pos"] - center) < 0.05

    def test_symmetric_peak_com_equals_peak(self):
        x = np.linspace(-3, 3, 501)
        y = np.exp(-0.5 * (x / 0.5) ** 2)
        result = calculate_1d_analysis(x, y)
        assert abs(result["com_pos"] - result["peak_pos"]) < 0.05

    def test_asymmetric_peak_com_shifts(self):
        x = np.linspace(0, 10, 501)
        y = np.exp(-0.5 * ((x - 3) / 0.5) ** 2) + 0.5 * np.exp(-0.5 * ((x - 7) / 0.5) ** 2)
        result = calculate_1d_analysis(x, y)
        assert result["com_pos"] > result["peak_pos"]

    def test_fwhm_known_gaussian(self):
        x = np.linspace(-10, 10, 10001)
        sigma = 1.0
        y = np.exp(-0.5 * (x / sigma) ** 2)
        expected_fwhm = 2.3548 * sigma
        result = calculate_1d_analysis(x, y)
        assert abs(result["fwhm_value"] - expected_fwhm) < 0.01

    def test_empty_input_returns_none(self):
        assert calculate_1d_analysis([], []) is None

    def test_mismatched_lengths_returns_none(self):
        assert calculate_1d_analysis([1, 2, 3], [1, 2]) is None

    def test_zero_intensity_returns_none(self):
        x = np.arange(10, dtype=float)
        y = np.zeros(10)
        assert calculate_1d_analysis(x, y) is None

    def test_single_point(self):
        result = calculate_1d_analysis([5.0], [100.0])
        assert result is not None
        assert result["peak_pos"] == 5.0
        assert result["peak_intensity"] == 100.0

    def test_return_dict_keys(self):
        x = np.linspace(0, 1, 100)
        y = np.sin(np.pi * x)
        result = calculate_1d_analysis(x, y)
        expected_keys = {
            "peak_pos", "peak_intensity", "baseline_intensity",
            "com_pos", "com_intensity",
            "fwhm_value", "fwhm_center", "fwhm_center_intensity",
            "fwhm_left", "fwhm_right", "half_max",
        }
        assert set(result.keys()) == expected_keys
