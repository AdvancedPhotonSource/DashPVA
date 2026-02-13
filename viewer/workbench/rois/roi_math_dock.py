#!/usr/bin/env python3
"""
ROIMathDock: A dockable window for ROI math expressions (1D view)

- Displays the ROI sub-image flattened to 1D (Index vs Value)
- Lets user define multiple math expressions using x (indices), y (ROI values), numpy (np),
  and common numpy functions (sin, cos, log, exp, sqrt, abs, clip, where)
- Each expression renders as a separate colored curve with a legend entry
- Updates automatically when the ROI region changes or when the Workbench frame changes
"""

from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg

class ROIMathDock(QDockWidget):
    def __init__(self, parent, title: str, main_window, roi):
        super().__init__(title, parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.main = main_window
        self.roi = roi

        # Container for plot + controls
        container = QWidget(self)
        layout = QVBoxLayout(container)
        try:
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(6)
        except Exception:
            pass

        # Plot setup
        self.plot_item = pg.PlotItem()
        self.plot_item.setLabel('bottom', 'Index')
        self.plot_item.setLabel('left', 'Value')
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        layout.addWidget(self.plot_widget)
        try:
            self.plot_item.addLegend()
        except Exception:
            pass

        # ROI Math panel
        self._setup_roi_math_panel(layout)

        # Install central widget
        self.setWidget(container)

        # Internal storage for equations and their plotted items
        self.math_items = {}  # name -> {'expr': str, 'curve': PlotDataItem}
        self._color_index = 0
        self._colors = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
            (188, 189, 34), (23, 190, 207)
        ]

        # Compute initial ROI vector and plot base curve
        self._update_base_curve()

        # Wire interactions to keep data in sync
        self._wire_interactions()

    # ----- UI setup -----
    def _setup_roi_math_panel(self, parent_layout: QVBoxLayout):
        gb = QGroupBox("ROI Math")
        v = QVBoxLayout(gb)

        # Instructions
        lbl = QLabel(
            "Define expressions using x (indices), y (ROI values), and numpy (np).\n"
            "Examples: y*2, np.log1p(y), (y-y.mean())/(y.std()+1e-9), clip(y,0,1000)"
        )
        lbl.setWordWrap(True)
        v.addWidget(lbl)

        # Input row: name + expression + add button
        row = QHBoxLayout()
        self.eq_name_edit = QLineEdit(); self.eq_name_edit.setPlaceholderText("Equation name (optional)")
        self.eq_edit = QLineEdit(); self.eq_edit.setPlaceholderText("Enter expression e.g., np.log1p(y)")
        self.btn_add = QPushButton("Add Equation")
        self.btn_add.clicked.connect(self._on_add_equation)
        row.addWidget(self.eq_name_edit)
        row.addWidget(self.eq_edit)
        row.addWidget(self.btn_add)
        v.addLayout(row)

        # Buttons: recompute all, remove selected, clear all
        btn_row = QHBoxLayout()
        self.btn_recompute = QPushButton("Recompute & Plot All")
        self.btn_recompute.clicked.connect(self._recompute_all)
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(self.btn_recompute)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear)
        v.addLayout(btn_row)

        # List of equations
        self.eq_list = QListWidget()
        self.eq_list.itemDoubleClicked.connect(self._edit_equation_item)
        v.addWidget(self.eq_list)

        parent_layout.addWidget(gb)

    # ----- Data extraction -----
    def _extract_roi_subimage(self):
        """Extract the ROI sub-image for the current frame, honoring transforms via getArrayRegion."""
        frame = None
        try:
            frame = self.main.get_current_frame_data()
        except Exception:
            frame = None
        if frame is None:
            return None
        sub = None
        try:
            image_item = getattr(self.main.image_view, 'imageItem', None) if hasattr(self.main, 'image_view') else None
            if image_item is not None:
                sub = self.roi.getArrayRegion(frame, image_item)
                if sub is not None and hasattr(sub, 'ndim') and sub.ndim > 2:
                    sub = np.squeeze(sub)
        except Exception:
            sub = None
        if sub is None or int(getattr(sub, 'size', 0)) == 0:
            # Fallback to axis-aligned bbox
            try:
                pos = self.roi.pos(); size = self.roi.size()
                x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
                w = max(1, int(size.x())); h = max(1, int(size.y()))
                hgt, wid = frame.shape
                x1 = min(wid, x0 + w); y1 = min(hgt, y0 + h)
                if x0 < x1 and y0 < y1:
                    sub = frame[y0:y1, x0:x1]
            except Exception:
                sub = None
        return sub

    def _update_base_curve(self):
        sub = self._extract_roi_subimage()
        if sub is None or int(getattr(sub, 'size', 0)) == 0:
            # Plot an empty placeholder
            self.x = np.array([0], dtype=int)
            self.y = np.array([0.0], dtype=float)
        else:
            self.y = np.asarray(sub, dtype=np.float32).ravel()
            self.x = np.arange(len(self.y))
        # Plot base ROI curve
        try:
            self.plot_item.clear()
            try:
                self.plot_item.addLegend()
            except Exception:
                pass
            self.base_curve = self.plot_item.plot(self.x, self.y, pen=pg.mkPen(color='y', width=1.5), name='ROI')
        except Exception:
            # Fallback
            try:
                self.base_curve = self.plot_widget.plot(self.x, self.y, pen='y', clear=True)
            except Exception:
                pass
        # Recompute math curves after base update
        self._recompute_all()

    # ----- Signal wiring -----
    def _wire_interactions(self):
        # ROI changes -> update base and math curves
        try:
            if hasattr(self.roi, 'sigRegionChanged'):
                self.roi.sigRegionChanged.connect(self._update_base_curve)
            if hasattr(self.roi, 'sigRegionChangeFinished'):
                self.roi.sigRegionChangeFinished.connect(self._update_base_curve)
        except Exception:
            pass
        # Frame spinbox -> update base and math curves
        try:
            if hasattr(self.main, 'frame_spinbox'):
                self.main.frame_spinbox.valueChanged.connect(lambda _: self._update_base_curve())
        except Exception:
            pass
        # Log scale toggle -> update curves based on what image shows (optional)
        try:
            if hasattr(self.main, 'cbLogScale'):
                self.main.cbLogScale.toggled.connect(lambda _: self._update_base_curve())
        except Exception:
            pass

    # ----- Math engine -----
    def _next_color(self):
        color = self._colors[self._color_index % len(self._colors)]
        self._color_index += 1
        return pg.mkPen(color=color, width=1.5)

    def _safe_eval(self, expr: str):
        """Safely evaluate an expression using restricted namespace.
        Returns a numpy array of shape (N,) or a scalar. Raises on error.
        """
        allowed = {
            'np': np,
            'x': self.x,
            'y': self.y,
            'sin': np.sin, 'cos': np.cos, 'log': np.log, 'exp': np.exp, 'sqrt': np.sqrt,
            'abs': np.abs, 'clip': np.clip, 'where': np.where,
        }
        globals_dict = {'__builtins__': {}}
        return eval(expr, globals_dict, allowed)

    def _plot_curve(self, name: str, y_curve):
        # Convert scalar to horizontal line
        if np.isscalar(y_curve):
            y_curve = np.full_like(self.x, float(y_curve), dtype=float)
        else:
            y_curve = np.asarray(y_curve, dtype=float)
        # Validate length
        if y_curve.shape[0] != self.x.shape[0]:
            raise ValueError(f"Expression result length {y_curve.shape[0]} does not match ROI length {self.x.shape[0]}")
        # Remove old curve if re-plotting
        if name in self.math_items and self.math_items[name]['curve'] is not None:
            try:
                self.plot_item.removeItem(self.math_items[name]['curve'])
            except Exception:
                pass
            self.math_items[name]['curve'] = None
        # Add new curve
        pen = self._next_color()
        curve = self.plot_item.plot(self.x, y_curve, pen=pen, name=name)
        return curve

    # ----- Handlers -----
    def _on_add_equation(self):
        expr = (self.eq_edit.text() or '').strip()
        if not expr:
            QMessageBox.information(self, "ROI Math", "Please enter an expression.")
            return
        name = (self.eq_name_edit.text() or '').strip()
        if not name:
            name = f"eq{len(self.math_items) + 1}"
        try:
            result = self._safe_eval(expr)
            curve = self._plot_curve(name, result)
        except Exception as e:
            QMessageBox.critical(self, "ROI Math Error", f"Could not evaluate expression:\n{expr}\n\n{e}")
            return
        self.math_items[name] = {'expr': expr, 'curve': curve}
        item = QListWidgetItem(f"{name}: {expr}")
        item.setData(32, name)
        self.eq_list.addItem(item)
        self.eq_name_edit.clear()
        self.eq_edit.clear()

    def _edit_equation_item(self, item: QListWidgetItem):
        name = item.data(32)
        if not name:
            return
        expr = self.math_items.get(name, {}).get('expr', '')
        self.eq_name_edit.setText(name)
        self.eq_edit.setText(expr)

    def _recompute_all(self):
        # Recompute and update curves for all equations
        for i in range(self.eq_list.count()):
            item = self.eq_list.item(i)
            name = item.data(32)
            if not name:
                continue
            expr = self.math_items.get(name, {}).get('expr')
            if not expr:
                continue
            try:
                result = self._safe_eval(expr)
                curve = self._plot_curve(name, result)
                self.math_items[name]['curve'] = curve
            except Exception as e:
                QMessageBox.critical(self, "ROI Math Error", f"Error recomputing '{name}':\n{expr}\n\n{e}")

    def _remove_selected(self):
        item = self.eq_list.currentItem()
        if not item:
            return
        name = item.data(32)
        try:
            curve = self.math_items.get(name, {}).get('curve')
            if curve is not None:
                self.plot_item.removeItem(curve)
        except Exception:
            pass
        self.math_items.pop(name, None)
        row = self.eq_list.row(item)
        self.eq_list.takeItem(row)

    def _clear_all(self):
        for name, rec in list(self.math_items.items()):
            try:
                if rec.get('curve') is not None:
                    self.plot_item.removeItem(rec['curve'])
            except Exception:
                pass
        self.math_items.clear()
        self.eq_list.clear()
