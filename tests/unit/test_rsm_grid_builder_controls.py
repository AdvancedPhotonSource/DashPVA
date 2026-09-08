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

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

from dashpva.viewer.rsmgrid.grid_preview import GridBoxPreview
from dashpva.viewer.rsmgrid.rsm_grid_builder import RSMGridBuilderDialog


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(application):
    window = RSMGridBuilderDialog()
    yield window
    window.close()


def test_resolution_controls_are_voxel_counts(dialog):
    assert not hasattr(dialog, "cmb_resolution_units")
    assert dialog.lbl_resolution.text() == "Voxel count:"
    assert dialog._grid_dims() == (200, 200, 200)
    assert dialog.spn_resolution.suffix() == " voxels"


def test_per_axis_resolution_preserves_asymmetric_counts(dialog):
    dialog.chk_per_axis.setChecked(True)
    dialog.spn_nx.setValue(40)
    dialog.spn_ny.setValue(50)
    dialog.spn_nz.setValue(60)

    assert dialog._grid_dims() == (40, 50, 60)


def test_preview_accepts_asymmetric_shape_and_bounds(application):
    preview = GridBoxPreview()
    bounds = (-1.0, 1.0, -2.0, 2.0, 0.0, 8.0)

    preview.set_grid((40, 50, 60), bounds=bounds)

    assert preview._shape == (40, 50, 60)
    assert preview._bounds == bounds
    assert preview._axis_lengths() == pytest.approx((0.25, 0.5, 1.0))
    preview.close()


def test_invalid_fixed_range_is_rejected_before_worker_start(dialog, monkeypatch):
    dialog.lst_files.addItem("scan.h5")
    dialog.chk_auto_range.setChecked(False)
    dialog.spn_hmin.setValue(1.0)
    dialog.spn_hmax.setValue(1.0)
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    dialog._start()

    assert warnings == [("Invalid HKL Range", "H max must be greater than H min.")]
    assert dialog.worker is None
