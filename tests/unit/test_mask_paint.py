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
    fake = SimpleNamespace(
        mask=np.zeros(shape, dtype=bool),
        _is_transposed=is_transposed,
        _rot_num=rot_num,
    )
    MaskViewerWindow = _import_mask_viewer()
    fake._paint_disk = lambda *args: MaskViewerWindow._paint_disk(fake, *args)
    fake._stamp_along = lambda *args: MaskViewerWindow._stamp_along(fake, *args)
    return fake


class TestSizeConsistency:
    """Brush/Rectangle/Line at the same "Size" should paint the same pixel width."""

    @pytest.mark.parametrize("size", range(1, 7))
    def test_tools_use_exact_native_pixel_width(self, size):
        MaskViewerWindow = _import_mask_viewer()
        for paint in (
            lambda fake: MaskViewerWindow._paint_disk(fake, 15, 15, size, True),
            lambda fake: MaskViewerWindow._paint_square(fake, 15, 15, size, True),
            lambda fake: MaskViewerWindow._paint_line(
                fake, (10, 15), (20, 15), size, True
            ),
        ):
            fake = _fake((31, 31))
            paint(fake)
            cols = np.where(fake.mask[15])[0]
            assert cols.max() - cols.min() + 1 == size


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

    @pytest.mark.parametrize("canvas_shape", [(8, 12), (2, 3)])
    @pytest.mark.parametrize("is_transposed", [False, True])
    @pytest.mark.parametrize("rot_num", [0, 1, 2, 3])
    def test_canvas_scaling_happens_before_inverse_transform(
        self, canvas_shape, is_transposed, rot_num
    ):
        MaskViewerWindow = _import_mask_viewer()
        fake = _fake((4, 6), is_transposed=is_transposed, rot_num=rot_num)
        display_shape = MaskViewerWindow._display_shape(fake)
        for canvas_i, canvas_j in (
            (0, 0),
            (canvas_shape[0] - 1, canvas_shape[1] - 1),
            (canvas_shape[0] // 2, canvas_shape[1] // 2),
        ):
            display_i = MaskViewerWindow._scale_canvas_index(
                canvas_i, canvas_shape[0], display_shape[0]
            )
            display_j = MaskViewerWindow._scale_canvas_index(
                canvas_j, canvas_shape[1], display_shape[1]
            )
            actual = MaskViewerWindow._display_to_native(fake, display_i, display_j)
            assert 0 <= actual[0] < fake.mask.shape[0]
            assert 0 <= actual[1] < fake.mask.shape[1]


class TestBoundedHistory:
    def test_small_stroke_history_is_region_sized(self):
        MaskViewerWindow = _import_mask_viewer()
        fake = _fake((4096, 4096))
        fake._stroke_bounds = None
        fake._stroke_before = None
        MaskViewerWindow._capture_stroke_region(fake, (100, 106, 200, 206))
        entry = {
            "kind": "region",
            "bounds": fake._stroke_bounds,
            "before": fake._stroke_before,
            "after": fake._stroke_before.copy(),
        }
        assert MaskViewerWindow._history_size(entry) == 200
        assert MaskViewerWindow._history_size(entry) < fake.mask.nbytes // 1000

    @pytest.mark.parametrize("canvas_shape", [(8, 12), (2, 3)])
    @pytest.mark.parametrize("is_transposed", [False, True])
    @pytest.mark.parametrize("rot_num", [0, 1, 2, 3])
    def test_dirty_region_matches_full_redraw(
        self, canvas_shape, is_transposed, rot_num
    ):
        MaskViewerWindow = _import_mask_viewer()
        fake = _fake((4, 6), is_transposed=is_transposed, rot_num=rot_num)
        fake._canvas_shape = canvas_shape
        fake.mask_overlay = SimpleNamespace(
            image=np.zeros(canvas_shape, dtype=np.float32), update=lambda: None
        )
        fake.lbl_info = SimpleNamespace(setText=lambda *_: None)
        fake._info_text = lambda: ""
        fake._display_shape = lambda: MaskViewerWindow._display_shape(fake)
        fake._refresh_overlay = lambda **kwargs: None
        fake.mask[1:3, 2:5] = True

        MaskViewerWindow._refresh_native_region(fake, (1, 3, 2, 5), False)

        display = fake.mask.T if is_transposed else fake.mask
        if rot_num:
            display = np.rot90(display, rot_num)
        expected = MaskViewerWindow._nearest_resize(display, canvas_shape)
        np.testing.assert_array_equal(fake.mask_overlay.image, expected)
