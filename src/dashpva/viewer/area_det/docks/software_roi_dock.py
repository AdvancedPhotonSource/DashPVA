"""Management + live-stats dock for the interactive software ROIs.

Provides Add / Clear / Save / Load controls and a per-ROI stats table that
refreshes every frame. Structural changes (add/delete/rename/clear) arrive via
the :class:`AreaDetRoiManager` listener bus; numeric cells are filled by the
manager calling :meth:`update_row` each frame.
"""

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "controls"

_COLUMNS = ["Name", "Sum", "Mean", "Min", "Max", "Std", "Count", "CoMX", "CoMY", "X", "Y", "W", "H"]


class SoftwareRoiDock(BaseDock):
    """Add/manage user-drawn ROIs and show their live per-frame stats."""

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Software ROIs", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._labels: dict = {}
        self._build()
        self.main_window.roi_manager.stats_dock = self
        self.main_window.roi_manager.add_listener(self._on_manager_event)

    def _build(self):
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        toolbar = QHBoxLayout()
        self.btn_add_roi = QPushButton("Add ROI")
        self.btn_add_roi.setObjectName("btn_add_roi")
        self.btn_add_cut = QPushButton("Add Cut")
        self.btn_add_cut.setObjectName("btn_add_cut")
        self.btn_add_cut.setToolTip("Add a line cut — samples intensity along the line into the Line Cuts plot")
        self.btn_clear_rois = QPushButton("Clear All")
        self.btn_clear_rois.setObjectName("btn_clear_rois")
        self.btn_save_rois = QPushButton("Save JSON…")
        self.btn_save_rois.setObjectName("btn_save_rois")
        self.btn_save_rois.setToolTip("Save the current ROIs to a JSON preset file")
        self.btn_load_rois = QPushButton("Load JSON…")
        self.btn_load_rois.setObjectName("btn_load_rois")
        self.btn_load_rois.setToolTip("Load ROIs from a JSON preset file")
        self.chk_show_names = QCheckBox("Show names")
        self.chk_show_names.setObjectName("chk_show_names")
        self.chk_show_names.setChecked(True)
        for w in (self.btn_add_roi, self.btn_add_cut, self.btn_clear_rois,
                  self.btn_save_rois, self.btn_load_rois):
            toolbar.addWidget(w)
        toolbar.addWidget(self.chk_show_names)
        toolbar.addStretch()
        outer.addLayout(toolbar)

        self.roi_stats_table = QTableWidget(0, len(_COLUMNS), container)
        self.roi_stats_table.setObjectName("roi_stats_table")
        self.roi_stats_table.setHorizontalHeaderLabels(_COLUMNS)
        self.roi_stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.roi_stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.roi_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.roi_stats_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_stats_table.customContextMenuRequested.connect(self._table_menu)
        outer.addWidget(self.roi_stats_table)

        self.setWidget(container)

        self.btn_add_roi.clicked.connect(lambda: self.main_window.roi_manager.create_and_add_roi())
        self.btn_add_cut.clicked.connect(self._add_cut)
        self.btn_clear_rois.clicked.connect(self.main_window.roi_manager.clear_all_rois)
        self.btn_save_rois.clicked.connect(self.main_window.save_rois_to_json)
        self.btn_load_rois.clicked.connect(self.main_window.load_rois_from_json)
        self.chk_show_names.toggled.connect(self._refresh_labels)

    def _add_cut(self) -> None:
        cut = self.main_window.roi_manager.create_and_add_cut()
        if cut is not None:
            self.main_window.open_line_cut_dock(cut)

    # ----- Manager callbacks -----
    def _on_manager_event(self, event: str, roi) -> None:
        if event in ('added', 'deleted', 'renamed', 'cleared', 'structure', 'recolor'):
            self._rebuild_table()
            self._refresh_labels()
        elif event in ('frame', 'moved'):
            self._refresh_label_positions()

    @staticmethod
    def _color_icon(hex_color: str) -> QIcon:
        pm = QPixmap(12, 12)
        pm.fill(QColor(hex_color))
        return QIcon(pm)

    def _rebuild_table(self) -> None:
        mgr = self.main_window.roi_manager
        rois = mgr.all_rois()
        self.roi_stats_table.setRowCount(len(rois))
        for row, roi in enumerate(rois):
            color = mgr.get_roi_color(roi)
            item = QTableWidgetItem(mgr.get_roi_name(roi))
            item.setForeground(QColor(color))
            item.setIcon(self._color_icon(color))
            if mgr.is_readonly(roi):
                item.setToolTip("Detector ROI (read-only) — geometry owned by EPICS")
            self.roi_stats_table.setItem(row, 0, item)

    def update_row(self, roi, stats) -> None:
        # Column 0 (name/color) is owned by _rebuild_table; only numeric cells
        # are refreshed per frame here.
        mgr = self.main_window.roi_manager
        rois = mgr.all_rois()
        if roi not in rois:
            return
        row = rois.index(roi)
        # Order matches _COLUMNS after the Name column: stats first, position last.
        values = [
            f"{stats['sum']:.3f}", f"{stats['mean']:.3f}", f"{stats['min']:.3f}",
            f"{stats['max']:.3f}", f"{stats['std']:.3f}", stats['count'],
            f"{stats['comx']:.2f}", f"{stats['comy']:.2f}",
            stats['x'], stats['y'], stats['w'], stats['h'],
        ]
        for offset, val in enumerate(values):
            self.roi_stats_table.setItem(row, offset + 1, QTableWidgetItem(str(val)))

    def _table_menu(self, pos) -> None:
        mgr = self.main_window.roi_manager
        row = self.roi_stats_table.rowAt(pos.y())
        rois = mgr.all_rois()
        if row < 0 or row >= len(rois):
            return
        roi = rois[row]
        menu = QMenu(self)
        if mgr.is_cut(roi):
            act_cut_plot = QAction("Open Line Cut Plot", menu)
            act_cut_plot.triggered.connect(lambda: self.main_window.open_line_cut_dock(roi))
            menu.addAction(act_cut_plot)
        else:
            act_plot = QAction("Open ROI Plot", menu)
            act_2d = QAction("Open ROI 2D Plot", menu)
            act_plot.triggered.connect(lambda: self.main_window.open_roi_plot_dock(roi))
            act_2d.triggered.connect(lambda: self.main_window.open_roi_2d_plot_dock(roi))
            menu.addAction(act_plot)
            menu.addAction(act_2d)
        if not mgr.is_readonly(roi):
            act_color = QAction("Set Color…", menu)
            act_color.triggered.connect(lambda: mgr.pick_roi_color(roi))
            selected = self._selected_rois() or [roi]
            editable = [r for r in selected if not mgr.is_readonly(r)]
            label = "Save Selected to JSON…" if len(editable) > 1 else "Save to JSON…"
            act_json = QAction(label, menu)
            act_json.triggered.connect(lambda: mgr.save_rois_to_json(editable))
            act_delete = QAction("Delete Cut" if mgr.is_cut(roi) else "Delete ROI", menu)
            act_delete.triggered.connect(lambda: mgr.delete_roi(roi))
            menu.addSeparator()
            menu.addAction(act_color)
            menu.addAction(act_json)
            menu.addAction(act_delete)
        menu.exec_(self.roi_stats_table.viewport().mapToGlobal(pos))

    def _selected_rois(self) -> list:
        """ROIs for the table's currently selected rows (in all_rois order)."""
        mgr = self.main_window.roi_manager
        rois = mgr.all_rois()
        rows = sorted({idx.row() for idx in self.roi_stats_table.selectionModel().selectedRows()})
        return [rois[r] for r in rows if 0 <= r < len(rois)]

    # ----- Name labels over ROIs -----
    def _refresh_labels(self, *_args) -> None:
        mgr = self.main_window.roi_manager
        # Only draw name labels while live view is running; clear them when idle
        # (start not on / stop clicked).
        show = self.chk_show_names.isChecked() and getattr(self.main_window, '_running', False)
        if not show:
            for label in self._labels.values():
                try:
                    self.main_window.image_view.removeItem(label)
                except Exception:
                    pass
            self._labels.clear()
            return
        live_ids = {id(r) for r in mgr.rois}
        for rid in list(self._labels):
            if rid not in live_ids:
                try:
                    self.main_window.image_view.removeItem(self._labels.pop(rid))
                except Exception:
                    self._labels.pop(rid, None)
        for roi in mgr.rois:
            if id(roi) not in self._labels:
                # Anchor the text's bottom-right at the ROI's top-right corner so
                # it sits just above the box, stays upright, and tracks moves/resizes.
                label = pg.TextItem(mgr.get_roi_name(roi), color=mgr.get_roi_color(roi), anchor=(1, 1))
                self.main_window.image_view.addItem(label)
                self._labels[id(roi)] = label
            else:
                self._labels[id(roi)].setText(mgr.get_roi_name(roi), color=mgr.get_roi_color(roi))
        self._refresh_label_positions()

    def _refresh_label_positions(self) -> None:
        if not self._labels:
            return
        mgr = self.main_window.roi_manager
        for roi in mgr.rois:
            label = self._labels.get(id(roi))
            if label is not None:
                pos, size = roi.pos(), roi.size()
                # Top-right corner (view Y is inverted, so top == pos.y()), so the
                # label sits at the upper-right of the box and follows moves/resizes.
                label.setPos(pos.x() + size.x(), pos.y())
