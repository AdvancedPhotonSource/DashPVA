"""Manual ROIs must obey the global 'Show ROIs' toggle (regression).

The global ``display_rois`` checkbox once hid only the EPICS ROIs 1-4; the amber
manual ROIs (M1..M5) stayed on screen. These tests exercise
``DiffractionImageWindow._apply_roi_visibility`` against a lightweight fake self
(no Qt widgets) so the toggle logic is checked without a live viewer.
"""

from types import SimpleNamespace

import pytest


class _FakeRoi:
    def __init__(self):
        self.visible = True

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False


class _Chk:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


def _apply(global_on, manual_rois, *, reader=None, rois=None, chk_show_roi=None):
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    from dashpva.viewer.area_det.area_det_viewer import DiffractionImageWindow

    fake = SimpleNamespace(
        display_rois=_Chk(global_on),
        manual_rois=manual_rois,
        reader=reader,
        rois=rois or [],
        chk_show_roi=chk_show_roi or [],
    )
    DiffractionImageWindow._apply_roi_visibility(fake)
    return fake


class TestManualRoiVisibility:

    def test_global_off_hides_manual_rois(self):
        r1, r2 = _FakeRoi(), _FakeRoi()
        _apply(False, [{"roi": r1}, {"roi": r2}])
        assert not r1.visible
        assert not r2.visible

    def test_global_on_shows_manual_rois(self):
        r1, r2 = _FakeRoi(), _FakeRoi()
        r1.hide()
        r2.hide()
        _apply(True, [{"roi": r1}, {"roi": r2}])
        assert r1.visible
        assert r2.visible

    def test_manual_rois_toggled_even_without_reader(self):
        # reader=None must NOT skip manual ROIs (they can be restored pre-connect).
        r1 = _FakeRoi()
        _apply(False, [{"roi": r1}], reader=None)
        assert not r1.visible

    def test_none_roi_entry_is_skipped(self):
        r1 = _FakeRoi()
        _apply(False, [{"roi": None}, {"roi": r1}])
        assert not r1.visible
