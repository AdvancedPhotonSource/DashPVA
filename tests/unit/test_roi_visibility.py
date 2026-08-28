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
