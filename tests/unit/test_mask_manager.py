"""Tests for dashpva.utils.mask_manager — mask load/save/combine/apply."""

import json
import os

import numpy as np
import pytest

from dashpva.utils.mask_manager import MaskManager


class TestMaskManagerInit:

    def test_creates_masks_dir(self, tmp_path):
        masks_dir = tmp_path / "new_masks"
        mm = MaskManager(masks_dir=str(masks_dir))
        assert masks_dir.exists()

    def test_no_active_mask_initially(self, tmp_path):
        mm = MaskManager(masks_dir=str(tmp_path / "masks"))
        assert mm.mask is None


class TestMaskSaveLoad:

    def test_save_and_load_npy(self, tmp_path):
        masks_dir = str(tmp_path / "masks")
        mm = MaskManager(masks_dir=masks_dir)
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 30:40] = True
        mm.mask = mask
        mm.save_active_mask()

        mm2 = MaskManager(masks_dir=masks_dir)
        assert mm2.mask is not None
        assert np.array_equal(mm2.mask, mask)

    def test_load_json_epics_format(self, tmp_path):
        masks_dir = str(tmp_path / "masks")
        mm = MaskManager(masks_dir=masks_dir)

        json_path = tmp_path / "bad_pixels.json"
        data = {
            "Bad pixels": [
                {"Pixel": [10, 20]},
                {"Pixel": [30, 40]},
                {"Pixel": [50, 60]},
            ]
        }
        json_path.write_text(json.dumps(data))

        loaded = mm.load_mask(str(json_path), detector_shape=(100, 100))
        assert loaded is not None
        assert loaded[20, 10]  # JSON uses [X, Y] = [col, row]
        assert loaded[40, 30]

    def test_load_npy_file(self, tmp_path):
        masks_dir = str(tmp_path / "masks")
        mm = MaskManager(masks_dir=masks_dir)

        npy_path = tmp_path / "test_mask.npy"
        mask = np.zeros((50, 50), dtype=bool)
        mask[5, 5] = True
        np.save(str(npy_path), mask)

        loaded = mm.load_mask(str(npy_path))
        assert loaded is not None
        assert loaded[5, 5]


class TestMaskOperations:

    def test_combine_masks_or(self, tmp_path):
        mm = MaskManager(masks_dir=str(tmp_path / "masks"))
        m1 = np.zeros((50, 50), dtype=bool)
        m1[0, 0] = True
        mm.mask = m1

        m2 = np.zeros((50, 50), dtype=bool)
        m2[49, 49] = True
        mm.combine_masks(m2)

        assert mm.mask[0, 0]
        assert mm.mask[49, 49]

    def test_apply_to_image(self, tmp_path):
        mm = MaskManager(masks_dir=str(tmp_path / "masks"))
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        mm.mask = mask

        image = np.ones((10, 10), dtype=np.float32) * 100.0
        result = mm.apply_to_image(image)
        assert result[5, 5] == 0.0
        assert result[0, 0] == 100.0

    def test_num_masked_pixels(self, tmp_path):
        mm = MaskManager(masks_dir=str(tmp_path / "masks"))
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:10, 0:10] = True
        mm.mask = mask
        assert mm.num_masked_pixels == 100

    def test_mask_fraction(self, tmp_path):
        mm = MaskManager(masks_dir=str(tmp_path / "masks"))
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:50, :] = True
        mm.mask = mask
        assert abs(mm.mask_fraction - 0.5) < 1e-6

    def test_clear_mask(self, tmp_path):
        masks_dir = str(tmp_path / "masks")
        mm = MaskManager(masks_dir=masks_dir)
        mm.mask = np.ones((10, 10), dtype=bool)
        mm.save_active_mask()
        mm.clear_mask()
        assert mm.mask is None


class TestDeadHotPixelDetection:

    def test_detect_dead_pixels(self, tmp_path):
        mm = MaskManager(masks_dir=str(tmp_path / "masks"))
        rng = np.random.default_rng(42)
        frames = rng.normal(1000, 100, size=(10, 50, 50)).astype(np.float32)
        # Make pixel (25, 25) dead (constant zero)
        frames[:, 25, 25] = 0.0
        result = mm.detect_dead_pixels(frames)
        assert result is not None
        assert result[25, 25]

    def test_detect_needs_minimum_frames(self, tmp_path):
        mm = MaskManager(masks_dir=str(tmp_path / "masks"))
        frames = np.ones((2, 50, 50), dtype=np.float32)
        result = mm.detect_dead_pixels(frames)
        assert result is None

    def test_dead_detection_masks_negative_pixels(self, tmp_path):
        # A high-variance pixel (not "dead") that reads negative in any frame is
        # still masked as unphysical (e.g. a -1 bad-pixel sentinel).
        mm = MaskManager(masks_dir=str(tmp_path / "masks"))
        rng = np.random.default_rng(0)
        frames = rng.normal(1000, 200, size=(10, 20, 20)).astype(np.float32)
        frames[3, 7, 7] = -1.0  # one unphysical reading; variance stays high
        result = mm.detect_dead_pixels(frames)
        assert result is not None
        assert result[7, 7]        # masked because it went negative
        assert not result[0, 1]    # a normal varying pixel stays unmasked

    def test_hot_detection_masks_negative_but_not_zero(self, tmp_path):
        # Dark frames: hot detection flags HIGH pixels, so the low -1 sentinel is
        # only caught by the negative rule. A valid dark zero must stay unmasked.
        mm = MaskManager(masks_dir=str(tmp_path / "masks"))
        rng = np.random.default_rng(1)
        frames = np.abs(rng.normal(0, 1, size=(10, 20, 20))).astype(np.float32)
        frames[:, 2, 2] = -1.0   # persistent bad-pixel sentinel -> mask
        frames[:, 4, 4] = 0.0    # valid dark zero -> do NOT mask
        result = mm.detect_hot_pixels(frames)
        assert result is not None
        assert result[2, 2]        # negative -> masked
        assert not result[4, 4]    # zero -> not masked
