# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Mask editor paint-helper and coordinate-mapping regressions.

Exercises ``MaskViewerWindow``'s paint/geometry methods against a lightweight
fake self (no Qt widgets), following the pattern in test_roi_visibility.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest


def _import_mask_viewer():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("qtawesome")
    from dashpva.viewer.mask_viewer import MaskViewerWindow
    return MaskViewerWindow


def _fake(shape, is_transposed=False, rot_num=0):
    return SimpleNamespace(
        mask=np.zeros(shape, dtype=bool),
        _is_transposed=is_transposed,
        _rot_num=rot_num,
    )


class TestSizeConsistency:
    """Brush/Rectangle/Line at the same "Size" should paint the same pixel width."""

    def test_thickness_to_radius_matches_paint_line_width(self):
        MaskViewerWindow = _import_mask_viewer()
        fake = _fake((20, 20))
        radius = MaskViewerWindow._thickness_to_radius(3)
        MaskViewerWindow._paint_disk(fake, 10, 10, radius, True)
        rows = np.where(fake.mask[:, 10])[0]
        assert rows.max() - rows.min() + 1 == 3

    def test_brush_size_matches_rectangle_side(self):
        MaskViewerWindow = _import_mask_viewer()
        brush = _fake((20, 20))
        rect = _fake((20, 20))
        radius = MaskViewerWindow._thickness_to_radius(3)
        MaskViewerWindow._paint_disk(brush, 10, 10, radius, True)
        MaskViewerWindow._paint_square(rect, 10, 10, 3, True)
        brush_rows = np.where(brush.mask[:, 10])[0]
        rect_rows = np.where(rect.mask[:, 10])[0]
        assert brush_rows.max() - brush_rows.min() + 1 == 3
        assert rect_rows.max() - rect_rows.min() + 1 == 3

    def test_thickness_to_radius_never_below_one(self):
        MaskViewerWindow = _import_mask_viewer()
        assert MaskViewerWindow._thickness_to_radius(1) >= 1


class TestPaintHelpers:

    def test_paint_disk_centered(self):
        MaskViewerWindow = _import_mask_viewer()
        fake = _fake((20, 20))
        MaskViewerWindow._paint_disk(fake, 10, 10, 3, True)
        assert fake.mask[10, 10]
        assert not fake.mask[0, 0]

    def test_paint_rect_spans_corners(self):
        MaskViewerWindow = _import_mask_viewer()
        fake = _fake((20, 20))
        MaskViewerWindow._paint_rect(fake, (2, 2), (5, 5), True)
        assert fake.mask[2:6, 2:6].all()
        assert not fake.mask[6, 6]

    def test_paint_disk_erase(self):
        MaskViewerWindow = _import_mask_viewer()
        fake = _fake((20, 20))
        fake.mask[:] = True
        MaskViewerWindow._paint_disk(fake, 10, 10, 3, False)
        assert not fake.mask[10, 10]
        assert fake.mask[0, 0]


class TestDisplayToNativeClosedForm:
    """The closed-form rewrite must exactly match the old probe-array inverse
    for every transpose/rotation combination the editor actually uses."""

    @staticmethod
    def _oracle(native_shape, is_transposed, rot_num, disp_i, disp_j):
        """Original probe-array-based inverse, kept only as a test oracle."""
        m, n = (native_shape[1], native_shape[0]) if is_transposed else native_shape
        disp_shape = (n, m) if rot_num % 2 else (m, n)
        probe = np.zeros(disp_shape, dtype=bool)
        probe[disp_i, disp_j] = True
        if rot_num:
            probe = np.rot90(probe, k=(4 - rot_num))
        if is_transposed:
            probe = probe.T
        idx = np.argwhere(probe)
        return int(idx[0, 0]), int(idx[0, 1])

    @pytest.mark.parametrize("is_transposed", [False, True])
    @pytest.mark.parametrize("rot_num", [0, 1, 2, 3])
    def test_matches_oracle_across_display_grid(self, is_transposed, rot_num):
        MaskViewerWindow = _import_mask_viewer()
        native_shape = (4, 6)
        fake = _fake(native_shape, is_transposed=is_transposed, rot_num=rot_num)
        m, n = (native_shape[1], native_shape[0]) if is_transposed else native_shape
        disp_shape = (n, m) if rot_num % 2 else (m, n)
        for di in range(disp_shape[0]):
            for dj in range(disp_shape[1]):
                expected = self._oracle(native_shape, is_transposed, rot_num, di, dj)
                actual = MaskViewerWindow._display_to_native(fake, di, dj)
                assert actual == expected, (is_transposed, rot_num, di, dj)

    def test_display_shape_matches_actual_transform(self):
        MaskViewerWindow = _import_mask_viewer()
        for is_transposed in (False, True):
            for rot_num in range(4):
                fake = _fake((4, 6), is_transposed=is_transposed, rot_num=rot_num)
                display = fake.mask.T.copy() if is_transposed else fake.mask.copy()
                if rot_num:
                    display = np.rot90(display, k=rot_num)
                assert MaskViewerWindow._display_shape(fake) == display.shape
