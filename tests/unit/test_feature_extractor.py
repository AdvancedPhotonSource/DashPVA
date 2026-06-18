"""Tests for FrameFeatureExtractor — new frame features are additive and the
existing schema is preserved."""

from __future__ import annotations

import json

import numpy as np

from dashpva.analysis.feature_extractor import FrameFeatureExtractor

_OLD_FRAME_KEYS = [
    'total_intensity', 'background', 'snr', 'peak_x', 'peak_y',
    'com_x', 'com_y', 'active_fraction', 'background_texture',
    'radial_profile_peaks',
]
_NEW_FRAME_KEYS = [
    'max_value', 'n_pixels_at_max', 'max_plateau_fraction',
    'blank_frame', 'edge_intensity_fraction',
]


def _spot_image(n=64):
    img = np.zeros((n, n), dtype=np.uint16)
    img[30:34, 30:34] = 1000
    return img


class TestSchema:

    def test_old_keys_preserved(self):
        f = FrameFeatureExtractor().extract(_spot_image(), np.empty((0, 5)))
        for k in _OLD_FRAME_KEYS:
            assert k in f['frame'], f"missing old key {k}"
        assert 'n_blobs' in f and 'blobs' in f

    def test_new_keys_present(self):
        f = FrameFeatureExtractor().extract(_spot_image(), np.empty((0, 5)))
        for k in _NEW_FRAME_KEYS:
            assert k in f['frame'], f"missing new key {k}"

    def test_json_serializable(self):
        f = FrameFeatureExtractor().extract(_spot_image(),
                                            np.array([[28, 28, 36, 36, 0.9]]))
        json.dumps(f)  # must not raise


class TestSaturationEvidence:

    def test_plateau_counted(self):
        img = np.zeros((32, 32), dtype=np.uint16)
        img[10:20, 10:20] = 4095   # 100-pixel plateau at the max
        f = FrameFeatureExtractor().extract(img, np.empty((0, 5)))
        assert f['frame']['max_value'] == 4095.0
        assert f['frame']['n_pixels_at_max'] == 100

    def test_single_hot_pixel(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        img[0, 0] = 65535
        f = FrameFeatureExtractor().extract(img, np.empty((0, 5)))
        assert f['frame']['n_pixels_at_max'] == 1


class TestBlankAndEdge:

    def test_blank_frame_on_zeros(self):
        f = FrameFeatureExtractor().extract(np.zeros((32, 32)), np.empty((0, 5)))
        assert f['frame']['blank_frame'] is True

    def test_not_blank_with_signal(self):
        f = FrameFeatureExtractor().extract(_spot_image(), np.empty((0, 5)))
        assert f['frame']['blank_frame'] is False

    def test_edge_intensity_fraction_high_for_edge_signal(self):
        img = np.zeros((64, 64), dtype=np.float64)
        img[:3, :] = 100.0   # all signal in the top border
        f = FrameFeatureExtractor().extract(img, np.empty((0, 5)))
        assert f['frame']['edge_intensity_fraction'] > 0.5