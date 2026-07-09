"""Interactive rectangular ROI with a right-click context menu.

Adapted from the workbench ``ContextRectROI`` for the area-detector live viewer.
The menu delegates to the viewer's ``AreaDetRoiManager`` and to the viewer's
own ``open_roi_plot_dock`` / ``open_roi_2d_plot_dock`` / ``save_software_rois_to_h5``.

Example:
    roi = ContextRectROI(viewer, [50, 50], [100, 100], pen=(224, 90, 90))
    viewer.image_view.addItem(roi)
"""

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QAction, QMenu


class ContextRectROI(pg.RectROI):
    """Rectangular ROI whose right-click menu drives the area-det ROI manager."""

    def __init__(self, parent_window, pos, size, pen=None):
        super().__init__(pos, size, pen=pen)
        self.parent_window = parent_window
        try:
            self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        except Exception:
            pass
        # Top-right rotate handle (rotates about the ROI center).
        try:
            self.addRotateHandle([1, 0], [0.5, 0.5])
        except Exception:
            pass

    def mouseClickEvent(self, ev):
        try:
            if ev.button() == Qt.RightButton:
                main = self.parent_window
                mgr = main.roi_manager
                menu = QMenu()
                action_stats = QAction("Show ROI Stats", menu)
                action_rename = QAction("Rename ROI", menu)
                action_color = QAction("Set Color…", menu)
                action_set_active = QAction("Set Active ROI", menu)
                action_plot = QAction("Open ROI Plot", menu)
                action_2d_plot = QAction("Open ROI 2D Plot", menu)
                action_hide = QAction("Hide ROI", menu)
                action_delete = QAction("Delete ROI", menu)
                action_save = QAction("Save ROIs", menu)
                action_save_json = QAction("Save to JSON…", menu)

                action_stats.triggered.connect(lambda: mgr.show_roi_stats_for_roi(self))
                action_rename.triggered.connect(lambda: mgr.rename_roi(self))
                action_color.triggered.connect(lambda: mgr.pick_roi_color(self))
                action_set_active.triggered.connect(lambda: mgr.set_active_roi(self))
                action_plot.triggered.connect(lambda: main.open_roi_plot_dock(self))
                action_2d_plot.triggered.connect(lambda: main.open_roi_2d_plot_dock(self))
                action_hide.triggered.connect(lambda: mgr.set_roi_visibility(self, False))
                action_delete.triggered.connect(lambda: mgr.delete_roi(self))
                action_save.triggered.connect(main.save_software_rois_to_h5)
                action_save_json.triggered.connect(lambda: mgr.save_roi_to_json(self))

                for act in (action_stats, action_rename, action_color, action_set_active,
                            action_plot, action_2d_plot, action_hide, action_delete):
                    menu.addAction(act)
                menu.addSeparator()
                menu.addAction(action_save)
                menu.addAction(action_save_json)
                menu.exec_(QCursor.pos())
                ev.accept()
                return
        except Exception:
            pass
        try:
            super().mouseClickEvent(ev)
        except Exception:
            pass


class ContextLineROI(pg.LineSegmentROI):
    """Two-point line cut whose right-click menu drives the area-det ROI manager.

    A line cut samples image intensity along the segment; its 1D profile is drawn
    in the viewer's Line Cuts dock while it lives in the same manager (and stats
    table) as the rectangular ROIs.

    Example:
        cut = ContextLineROI(viewer, [[50, 50], [200, 200]], pen=(224, 90, 90))
        viewer.image_view.addItem(cut)
    """

    def __init__(self, parent_window, positions, pen=None):
        super().__init__(positions, pen=pen)
        self.parent_window = parent_window
        try:
            self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        except Exception:
            pass

    def mouseClickEvent(self, ev):
        try:
            if ev.button() == Qt.RightButton:
                main = self.parent_window
                mgr = main.roi_manager
                menu = QMenu()
                action_rename = QAction("Rename Cut", menu)
                action_color = QAction("Set Color…", menu)
                action_set_active = QAction("Set Active Cut", menu)
                action_plot = QAction("Open Line Cut Plot", menu)
                action_delete = QAction("Delete Cut", menu)
                action_save_json = QAction("Save to JSON…", menu)
                action_save_all_json = QAction("Save All Cuts to JSON…", menu)

                action_rename.triggered.connect(lambda: mgr.rename_roi(self))
                action_color.triggered.connect(lambda: mgr.pick_roi_color(self))
                action_set_active.triggered.connect(lambda: mgr.set_active_roi(self))
                action_plot.triggered.connect(lambda: main.open_line_cut_dock(self))
                action_delete.triggered.connect(lambda: mgr.delete_roi(self))
                action_save_json.triggered.connect(lambda: mgr.save_roi_to_json(self))
                action_save_all_json.triggered.connect(mgr.save_cuts_to_json)

                for act in (action_rename, action_color, action_set_active,
                            action_plot, action_delete):
                    menu.addAction(act)
                menu.addSeparator()
                menu.addAction(action_save_json)
                menu.addAction(action_save_all_json)
                menu.exec_(QCursor.pos())
                ev.accept()
                return
        except Exception:
            pass
        try:
            super().mouseClickEvent(ev)
        except Exception:
            pass
