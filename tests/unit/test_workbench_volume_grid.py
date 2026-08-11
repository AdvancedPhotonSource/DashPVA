# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyvista")

from dashpva.viewer.workbench.workspace.workspace_3d import (
    Workspace3D,
    build_volume_grid,
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
