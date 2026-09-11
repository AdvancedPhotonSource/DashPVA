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

from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.test_frame_delivery import packet


class Toggle:
    def __init__(self, value=False):
        self.value = value

    def isChecked(self):
        return self.value


class Plot:
    def __init__(self, visible=True):
        self.visible = visible
        self.calls = []

    def isVisible(self):
        return self.visible

    def setData(self, **kwargs):
        self.calls.append(kwargs)


def window():
    pytest.importorskip('pvaccess')
    from dashpva.viewer.area_det.area_det_viewer import DiffractionImageWindow
    frame = packet(image=np.arange(6, dtype=np.uint32).reshape(2, 3))
    submitted = []
    bottom = Plot(False)
    w = SimpleNamespace(
        reader=SimpleNamespace(take_latest_frame=lambda: frame, CACHING_MODE='', pixel_ordering='F'),
        _last_display_key=None, bottom_avg_plot=bottom, chk_threshold=Toggle(),
        chk_apply_mask=Toggle(), log_image=Toggle(), chk_autoscale=Toggle(),
        mask_manager=SimpleNamespace(mask=None, shape_mismatch_info=None),
        image_is_transposed=False, rot_num=0, call_id_plot=0, first_plot=False,
        _collect_dead_pixel_frame=lambda: None,
        image_view=SimpleNamespace(setImage=lambda image, **kw: submitted.append(image.copy())),
        _horizontal_avg_curve=Plot(), _bottom_avg_curve=bottom,
        _sync_bottom_margins=lambda: None,
        min_px_val=SimpleNamespace(setText=lambda value: None),
        max_px_val=SimpleNamespace(setText=lambda value: None),
        apply_threshold=lambda image: np.minimum(image, 3),
    )
    return w, submitted, DiffractionImageWindow.update_image


def test_unchanged_frame_is_skipped_but_static_display_controls_still_work():
    w, submitted, update = window()
    update(w)
    update(w)
    assert len(submitted) == 1
    original = submitted[0].copy()
    w.rot_num = 1
    update(w)
    np.testing.assert_array_equal(submitted[-1], np.rot90(original))
    w.log_image.value = True
    update(w)
    np.testing.assert_allclose(submitted[-1], np.log10(np.rot90(original) + 1))
    np.testing.assert_array_equal(w._manual_roi_source, np.rot90(original))
    w.chk_threshold.value = True
    update(w)
    np.testing.assert_array_equal(w._manual_roi_source, np.rot90(np.minimum(original, 3)))
    w.chk_apply_mask.value = True
    w.mask_manager.mask = np.ones_like(original)
    w.mask_manager.apply_to_image = lambda image: image * 0
    update(w)
    assert np.all(w._manual_roi_source == 0)
    w.bottom_avg_plot.visible = True
    update(w)
    assert len(w._bottom_avg_curve.calls) == 1


def test_reconnect_with_reused_frame_identity_is_not_skipped():
    w, submitted, update = window()
    update(w)
    w.reader = SimpleNamespace(take_latest_frame=lambda: packet(image=np.ones((2, 3))), CACHING_MODE='', pixel_ordering='F')
    update(w)
    assert len(submitted) == 2
    assert np.all(submitted[-1] == 1)
