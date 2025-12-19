#!/usr/bin/env python3
"""
ROI Plot Dialog for Workbench (1D View)

Displays a 1D graph using the same approach as the Workbench 1D viewer:
- Flattens the ROI array to a 1D vector and plots Index vs Value
- Uses PyQtGraph PlotWidget with labeled axes
- Modeless dialog, does not block the main window

Adds a ROI Math panel to define and plot arbitrary math expressions
over the ROI data. You can add multiple equations; each renders as an
additional curve with a legend entry.

Variables available in expressions:
- x: index array (0..N-1)
- y: ROI values (flattened to 1D)
- np: numpy module
- Common numpy functions are also imported directly: sin, cos, log, exp, sqrt, abs, clip, where

Examples:
- y * 2
- np.log1p(y)
- (y - y.mean()) / (y.std() + 1e-9)
- clip(y, 0, 1000)
If an expression evaluates to a scalar, a horizontal line is plotted across x.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QGroupBox, QMessageBox
)
import numpy as np
import pyqtgraph as pg

class ROIPlotDialog(QDialog):
    def __init__(self, parent, roi_image: np.ndarray):
        super().__init__(parent)
        self.setWindowTitle("ROI 1D Plot")
        layout = QVBoxLayout(self)

        # Prepare 1D data: flatten ROI image into a vector
        self.y = np.asarray(roi_image, dtype=np.float32).ravel()
        self.x = np.arange(len(self.y))

        # Create a PlotItem and PlotWidget like the 1D viewer
        self.plot_item = pg.PlotItem()
        self.plot_item.setLabel('bottom', 'Index')
        self.plot_item.setLabel('left', 'Value')
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        layout.addWidget(self.plot_widget)

        # Add legend for multiple curves
        try:
            self.plot_item.addLegend()
        except Exception:
            pass

        # Plot the base ROI 1D data
        try:
            self.base_curve = self.plot_item.plot(self.x, self.y, pen=pg.mkPen(color='y', width=1.5), name='ROI')
        except Exception:
            # Fallback: simple PlotWidget plot
            self.base_curve = self.plot_widget.plot(self.x, self.y, pen='y')

        # ROI Math group
        self._setup_roi_math_panel(layout)

        # Internal storage for equations and their plotted items
        self.math_items = {}  # name -> {'expr': str, 'curve': PlotDataItem}
        self._color_index = 0
        self._colors = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
            (188, 189, 34), (23, 190, 207)
        ]

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

    def _next_color(self):
        color = self._colors[self._color_index % len(self._colors)]
        self._color_index += 1
        return pg.mkPen(color=color, width=1.5)

    def _safe_eval(self, expr: str):
        """Safely evaluate an expression using restricted namespace.
        Returns a numpy array of shape (N,) or a scalar. Raises on error.
        """
        # Allowed names
        allowed = {
            'np': np,
            'x': self.x,
            'y': self.y,
            # Common numpy functions directly for convenience
            'sin': np.sin, 'cos': np.cos, 'log': np.log, 'exp': np.exp, 'sqrt': np.sqrt,
            'abs': np.abs, 'clip': np.clip, 'where': np.where,
        }
        # No builtins
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

    def _on_add_equation(self):
        expr = (self.eq_edit.text() or '').strip()
        if not expr:
            QMessageBox.information(self, "ROI Math", "Please enter an expression.")
            return
        name = (self.eq_name_edit.text() or '').strip()
        if not name:
            # Derive a default name
            name = f"eq{len(self.math_items) + 1}"

        # Evaluate and plot
        try:
            result = self._safe_eval(expr)
            curve = self._plot_curve(name, result)
        except Exception as e:
            QMessageBox.critical(self, "ROI Math Error", f"Could not evaluate expression:\n{expr}\n\n{e}")
            return

        # Store and list
        self.math_items[name] = {'expr': expr, 'curve': curve}
        item = QListWidgetItem(f"{name}: {expr}")
        item.setData(32, name)  # store name for retrieval (Qt.UserRole=32)
        self.eq_list.addItem(item)
        # Clear inputs
        self.eq_name_edit.clear()
        self.eq_edit.clear()

    def _edit_equation_item(self, item: QListWidgetItem):
        # Simple inline edit via reusing input boxes: load into edits
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
        # Remove curve
        try:
            curve = self.math_items.get(name, {}).get('curve')
            if curve is not None:
                self.plot_item.removeItem(curve)
        except Exception:
            pass
        # Remove from storage and list
        self.math_items.pop(name, None)
        row = self.eq_list.row(item)
        self.eq_list.takeItem(row)

    def _clear_all(self):
        # Remove all curves
        for name, rec in list(self.math_items.items()):
            try:
                if rec.get('curve') is not None:
                    self.plot_item.removeItem(rec['curve'])
            except Exception:
                pass
        self.math_items.clear()
        self.eq_list.clear()

    # Optional: method to update ROI data and recompute curves (future use)
    def update_roi_data(self, roi_image: np.ndarray):
        self.y = np.asarray(roi_image, dtype=np.float32).ravel()
        self.x = np.arange(len(self.y))
        # Update base curve
        try:
            self.base_curve.setData(self.x, self.y)
        except Exception:
            pass
        # Recompute all math curves
        self._recompute_all()
