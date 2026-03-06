from pathlib import Path
from typing import Optional

from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import Qt

from viewer.workbench.docks.base_dock import BaseDock


class InformationDockBase(BaseDock):
    """Base information dock that loads a .ui and provides simple setters.

    This dock is UI-driven via gui/workbench/docks/information_dock.ui
    and exposes helpers to set points count and axis labels. It can be
    reused by dimension-specific subclasses (e.g., 2D, 3D).
    """

    def __init__(
        self,
        title: str = "Information",
        main_window=None,
        segment_name: Optional[str] = None,
        dock_area: Qt.DockWidgetArea = Qt.RightDockWidgetArea,
        show: bool = True,
    ):
        # BaseDock will perform docking and Windows-menu registration
        super().__init__(title=title, main_window=main_window, segment_name=segment_name, dock_area=dock_area, show=show)

        # Load the UI into a QWidget and set as the dock widget
        project_root = Path(__file__).resolve().parents[3]
        ui_path = project_root / "gui" / "workbench" / "docks" / "information_dock.ui"
        self._widget = QWidget(self)
        try:
            uic.loadUi(str(ui_path), self._widget)
        except Exception as e:
            # Fallback: create a minimal widget if UI load fails
            self._widget = QWidget(self)
            print(f"[InformationDockBase] Failed to load UI: {e}")
        self.setWidget(self._widget)

        # Cache label refs for fast updates
        try:
            self.lbl_points: QLabel = self._widget.findChild(QLabel, "lbl_points")
            self.lbl_axis_x: QLabel = self._widget.findChild(QLabel, "lbl_axis_x")
            self.lbl_axis_y: QLabel = self._widget.findChild(QLabel, "lbl_axis_y")
        except Exception:
            self.lbl_points = None
            self.lbl_axis_x = None
            self.lbl_axis_y = None
        # Intensity labels
        try:
            self.lbl_int_low: QLabel = self._widget.findChild(QLabel, "lbl_int_low")
            self.lbl_int_high: QLabel = self._widget.findChild(QLabel, "lbl_int_high")
        except Exception:
            self.lbl_int_low = None
            self.lbl_int_high = None
        # Mouse hover labels
        try:
            self.lbl_mouse_pos: QLabel = self._widget.findChild(QLabel, "lbl_mouse_pos")
            self.lbl_mouse_int: QLabel = self._widget.findChild(QLabel, "lbl_mouse_int")
            self.lbl_mouse_H: QLabel = self._widget.findChild(QLabel, "lbl_mouse_H")
            self.lbl_mouse_K: QLabel = self._widget.findChild(QLabel, "lbl_mouse_K")
            self.lbl_mouse_L: QLabel = self._widget.findChild(QLabel, "lbl_mouse_L")
        except Exception:
            self.lbl_mouse_pos = None
            self.lbl_mouse_int = None
            self.lbl_mouse_H = None
            self.lbl_mouse_K = None
            self.lbl_mouse_L = None

    # Helper setters
    def set_points(self, count: Optional[int]) -> None:
        try:
            if isinstance(count, int):
                txt = f"{count:,}"
            elif count is None:
                txt = "—"
            else:
                try:
                    txt = f"{int(count):,}"
                except Exception:
                    txt = str(count)
            if self.lbl_points is not None:
                self.lbl_points.setText(txt)
        except Exception:
            pass

    def set_axes(self, x_label: Optional[str], y_label: Optional[str]) -> None:
        try:
            if self.lbl_axis_x is not None:
                self.lbl_axis_x.setText(str(x_label) if x_label else "—")
            if self.lbl_axis_y is not None:
                self.lbl_axis_y.setText(str(y_label) if y_label else "—")
        except Exception:
            pass

    def set_intensity(self, low: Optional[float], high: Optional[float]) -> None:
        try:
            def fmt(val):
                if val is None:
                    return "—"
                try:
                    return f"{float(val):.6g}"
                except Exception:
                    return str(val)
            if self.lbl_int_low is not None:
                self.lbl_int_low.setText(fmt(low))
            if self.lbl_int_high is not None:
                self.lbl_int_high.setText(fmt(high))
        except Exception:
            pass

    def set_mouse_info(self, pos: Optional[tuple], intensity: Optional[float], H: Optional[float], K: Optional[float], L: Optional[float]) -> None:
        """Update the Mouse section with position, intensity, and HKL values, with HKL colors."""
        try:
            def fmtf(val):
                if val is None:
                    return "—"
                try:
                    return f"{float(val):.6g}"
                except Exception:
                    return str(val)
            def fmtpos(p):
                if not p or len(p) < 2:
                    return "—"
                try:
                    return f"({int(p[0])}, {int(p[1])})"
                except Exception:
                    return str(p)
            if getattr(self, 'lbl_mouse_pos', None) is not None:
                self.lbl_mouse_pos.setText(fmtpos(pos))
            if getattr(self, 'lbl_mouse_int', None) is not None:
                self.lbl_mouse_int.setText(fmtf(intensity))
            if getattr(self, 'lbl_mouse_H', None) is not None:
                self.lbl_mouse_H.setText(fmtf(H))
                try:
                    self.lbl_mouse_H.setStyleSheet("color: red;")
                except Exception:
                    pass
            if getattr(self, 'lbl_mouse_K', None) is not None:
                self.lbl_mouse_K.setText(fmtf(K))
                try:
                    self.lbl_mouse_K.setStyleSheet("color: green;")
                except Exception:
                    pass
            if getattr(self, 'lbl_mouse_L', None) is not None:
                self.lbl_mouse_L.setText(fmtf(L))
                try:
                    self.lbl_mouse_L.setStyleSheet("color: blue;")
                except Exception:
                    pass
        except Exception:
            pass
