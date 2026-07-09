"""Live 1D intensity plot for the viewer's line cuts.

One curve per line cut, showing image intensity sampled along the segment
(X = position along the cut in pixels, Y = intensity). Cuts are owned by the
:class:`AreaDetRoiManager` (same list/table/colors as the rectangular ROIs);
this dock just draws the profiles the manager caches in ``cut_profiles`` and
refreshes on every ``'frame'``/``'moved'`` notification.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget

import dashpva.settings as settings
from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "controls"


class LineCutDock(BaseDock):
    """Live 1D graph of intensity along each line cut."""

    def __init__(self, main_window=None, show: bool = False):
        super().__init__(title="Line Cuts", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._curves: dict = {}
        self._build()
        self.main_window.roi_manager.add_listener(self._on_manager_event)

    def _build(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.plot_item = pg.PlotItem()
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        self.plot_item.setLabel('bottom', 'Position along cut [px]')
        self.plot_item.setLabel('left', 'Intensity')
        self.legend = self.plot_item.addLegend()
        layout.addWidget(self.plot_widget)
        self.setWidget(container)

    # ----- Manager callbacks -----
    def _on_manager_event(self, event: str, roi) -> None:
        if event in ('added', 'deleted', 'cleared', 'structure', 'recolor', 'renamed'):
            self._rebuild()
        elif event in ('frame', 'moved'):
            self._refresh()

    def _rebuild(self) -> None:
        self.plot_item.clear()
        try:
            if self.legend is not None:
                self.legend.clear()
        except Exception:
            pass
        self._curves.clear()
        mgr = self.main_window.roi_manager
        for cut in mgr.cuts:
            color = mgr.get_roi_color(cut)
            curve = self.plot_item.plot(pen=pg.mkPen(color, width=settings.LINE_CUT_PEN_WIDTH),
                                        name=mgr.get_roi_name(cut))
            self._curves[id(cut)] = curve
        self._refresh()

    def _refresh(self) -> None:
        mgr = self.main_window.roi_manager
        for cut in mgr.cuts:
            curve = self._curves.get(id(cut))
            if curve is None:
                continue
            profile = mgr.cut_profiles.get(id(cut))
            if profile is None or len(profile) == 0:
                curve.setData([], [])
                continue
            y = np.asarray(profile, dtype=float)
            curve.setData(np.arange(y.size), y)
