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
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyvista")

from dashpva.viewer.workbench.workspace.workspace_3d import (
    Workspace3D,
    build_volume_grid,
    seed_intensity_spinboxes,
)


class _Control:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = enabled


def test_build_volume_grid_preserves_hkl_cell_geometry_and_order():
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    metadata = {
        "voxel_spacing": [0.1, 0.2, 0.3],
        "grid_origin": [-1.0, -2.0, -3.0],
        "array_order": "F",
    }

    grid = build_volume_grid(volume, metadata)

    assert tuple(grid.dimensions) == (3, 4, 5)
    assert tuple(grid.spacing) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(grid.origin) == pytest.approx((-1.0, -2.0, -3.0))
    np.testing.assert_array_equal(
        grid.cell_data["intensity"], volume.flatten(order="F")
    )


def test_build_volume_grid_rejects_unknown_array_order():
    with pytest.raises(ValueError, match="array_order"):
        build_volume_grid(np.ones((2, 3, 4)), {"array_order": "unknown"})


@pytest.mark.parametrize("mode, slice_enabled", [("volume", False), ("points", True)])
def test_slice_availability_is_data_mode_aware(mode, slice_enabled):
    workspace = SimpleNamespace(
        cloud_mesh_3d=object(),
        points_actor=object(),
        _data_mode=mode,
        cb_show_points=_Control(),
        cb_show_slice=_Control(),
        sb_min_intensity_3d=_Control(),
        sb_max_intensity_3d=_Control(),
    )

    Workspace3D._refresh_availability(workspace)

    assert workspace.cb_show_slice.enabled is slice_enabled


def test_volume_mode_never_accesses_image_data_points():
    class _Grid:
        @property
        def points(self):
            raise AssertionError("ImageData.points must not be materialized")

    workspace = SimpleNamespace(_data_mode="volume", cloud_mesh_3d=_Grid())
    Workspace3D.on_plane_update(workspace, np.array([0, 0, 1]), np.zeros(3))
    Workspace3D.view_slice_normal(workspace)


@pytest.fixture(scope="module")
def intensity_spinboxes():
    """The real Min/Max intensity widgets, loaded from the shipped .ui.

    Gridder3D voxels are per-bin means, so these must be QDoubleSpinBox --
    integer QSpinBox raises TypeError on a float range and would truncate an
    I0-normalized volume to an empty 0..0 range.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    uic = pytest.importorskip("PyQt5.uic")
    widgets = pytest.importorskip("PyQt5.QtWidgets")
    from dashpva.gui import ui_path

    app = widgets.QApplication.instance() or widgets.QApplication([])
    host = widgets.QWidget()
    uic.loadUi(ui_path("workbench", "workspace", "workspace_3d.ui"), host)
    yield host.sb_min_intensity_3d, host.sb_max_intensity_3d
    del app


def test_intensity_spinboxes_are_float_capable(intensity_spinboxes):
    from PyQt5.QtWidgets import QDoubleSpinBox

    for spinbox in intensity_spinboxes:
        assert isinstance(spinbox, QDoubleSpinBox)


@pytest.mark.parametrize(
    "data_min, data_max, decimals",
    [
        (0.0, 65535.0, 2),        # raw detector counts
        (0.5, 12.5678, 4),        # per-bin means
        (0.00341827, 0.523719, 6),  # I0-normalized means
    ],
)
def test_seeding_preserves_fractional_range(
    intensity_spinboxes, data_min, data_max, decimals
):
    sb_min, sb_max = intensity_spinboxes

    lo, hi = seed_intensity_spinboxes(sb_min, sb_max, data_min, data_max)

    assert (lo, hi) == pytest.approx((data_min, data_max))
    assert sb_min.decimals() == decimals
    assert sb_min.value() == pytest.approx(data_min, abs=10.0**-decimals)
    assert sb_max.value() == pytest.approx(data_max, abs=10.0**-decimals)


def test_seeding_handles_a_flat_volume(intensity_spinboxes):
    sb_min, sb_max = intensity_spinboxes

    lo, hi = seed_intensity_spinboxes(sb_min, sb_max, 2.5, 2.5)

    assert hi > lo
    assert sb_max.value() > sb_min.value()


def test_seeding_rejects_non_finite_range(intensity_spinboxes):
    sb_min, sb_max = intensity_spinboxes
    with pytest.raises(ValueError, match="finite"):
        seed_intensity_spinboxes(sb_min, sb_max, 0.0, np.nan)
