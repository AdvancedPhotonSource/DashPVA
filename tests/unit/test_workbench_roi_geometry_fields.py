# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Workbench ROI geometry fields reflect the ROI, not array-slice axis order."""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

sys.modules.setdefault("qtawesome", types.SimpleNamespace(icon=lambda *args, **kwargs: None))

import pyqtgraph as pg  # noqa: E402
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem  # noqa: E402

from dashpva.viewer.workbench.managers.roi_manager import (  # noqa: E402
    ContextRectROI,
    ROIManager,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _ImageView:
    def __init__(self, image):
        self.imageItem = pg.ImageItem(image)

    def getImage(self):
        return self.imageItem.image


class _Main:
    def __init__(self, image):
        self.image_view = _ImageView(image)
        self.roi_stats_table = QTableWidget(1, 13)

    def update_status(self, *args, **kwargs):
        pass


def test_stats_geometry_uses_roi_xywh_without_axis_swapping(qapp):
    image = np.arange(120 * 160, dtype=np.float32).reshape(120, 160)
    manager = ROIManager(_Main(image))
    roi = ContextRectROI(manager.main, [98, 37], [23, 41])

    stats = manager.compute_roi_stats(image, roi)

    assert (stats["x"], stats["y"]) == (98, 37)
    assert (stats["w"], stats["h"]) == (23, 41)


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [(9, 98, (98, 20, 30, 40)), (10, 99, (10, 99, 30, 21)),
     (11, 98, (10, 20, 98, 40)), (12, 99, (10, 20, 30, 99))],
)
def test_editing_one_geometry_field_does_not_change_another(
    qapp, column, value, expected
):
    image = np.arange(120 * 160, dtype=np.float32).reshape(120, 160)
    manager = ROIManager(_Main(image))
    roi = ContextRectROI(manager.main, [10, 20], [30, 40])
    manager.rois.append(roi)
    manager.roi_by_stats_row[0] = roi
    manager.stats_row_by_roi_id[id(roi)] = 0
    manager.update_stats_table_for_roi(roi, manager.compute_roi_stats(image, roi))
    roi.sigRegionChanged.connect(manager.show_roi_stats_for_roi)

    item = QTableWidgetItem(str(value))
    manager.main.roi_stats_table.setItem(0, column, item)
    manager.on_roi_stats_item_changed(item)

    pos = roi.pos()
    size = roi.size()
    actual = (round(pos.x()), round(pos.y()), round(size.x()), round(size.y()))
    assert actual == expected
