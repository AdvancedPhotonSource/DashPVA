from typing import Optional
import numpy as np
from PyQt5.QtCore import Qt

from viewer.workbench.docks.information_dock_base import InformationDockBase


class Info2DDock(InformationDockBase):
    """Information dock specialized for 2D viewing state.

    Shows number of points in the current frame and X/Y axis variable labels.
    """

    def __init__(
        self,
        main_window=None,
        title: str = "2D Info",
        segment_name: Optional[str] = "2d",
        dock_area: Qt.DockWidgetArea = Qt.RightDockWidgetArea,
    ):
        super().__init__(title=title, main_window=main_window, segment_name=segment_name, dock_area=dock_area)

    def refresh(self) -> None:
        """Refresh the displayed information based on the main window's 2D state."""
        mw = getattr(self, 'main_window', None)
        if mw is None:
            return
        # Try to keep mouse info consistent when refresh occurs
        try:
            xy = getattr(mw, '_last_hover_xy', None)
            frame = mw.get_current_frame_data() if hasattr(mw, 'get_current_frame_data') else None
            intensity = None
            H_val = K_val = L_val = None
            pos = None
            # Populate Mouse HKL even if no hover yet by falling back to center pixel
            if frame is not None and frame.ndim == 2:
                h, w = frame.shape
                # Validate hover position
                if xy is not None:
                    try:
                        x_hover, y_hover = int(xy[0]), int(xy[1])
                        if 0 <= x_hover < w and 0 <= y_hover < h:
                            x, y = x_hover, y_hover
                            pos = (x, y)
                        else:
                            x, y = w // 2, h // 2
                            pos = (x, y)
                    except Exception:
                        x, y = w // 2, h // 2
                        pos = (x, y)
                else:
                    x, y = w // 2, h // 2
                    pos = (x, y)
                # Intensity at chosen position
                try:
                    intensity = float(frame[y, x])
                except Exception:
                    intensity = None
                # HKL from cached q-grids if present
                try:
                    qxg = getattr(mw, '_qx_grid', None)
                    qyg = getattr(mw, '_qy_grid', None)
                    qzg = getattr(mw, '_qz_grid', None)
                    if qxg is not None and qyg is not None and qzg is not None:
                        if qxg.ndim == 3:
                            idx = int(mw.frame_spinbox.value()) if hasattr(mw, 'frame_spinbox') and mw.frame_spinbox.isEnabled() else 0
                            if 0 <= idx < qxg.shape[0]:
                                H_val = float(qxg[idx, y, x]); K_val = float(qyg[idx, y, x]); L_val = float(qzg[idx, y, x])
                        elif qxg.ndim == 2:
                            H_val = float(qxg[y, x]); K_val = float(qyg[y, x]); L_val = float(qzg[y, x])
                except Exception:
                    H_val = K_val = L_val = None
                # Update Mouse section in the dock
                self.set_mouse_info(pos, intensity, H_val, K_val, L_val)
        except Exception:
            pass
        # Points: show total points across data dimensions (include product, e.g., FxHxW = N)
        points_str = None
        low_val = None
        high_val = None
        try:
            data = getattr(mw, 'current_2d_data', None)
            if isinstance(data, np.ndarray):
                # total points
                total = int(data.size)
                points_str = f"{total:,}"
                # intensity low/high across dataset
                try:
                    low_val = float(np.min(data))
                except Exception:
                    low_val = None
                try:
                    high_val = float(np.max(data))
                except Exception:
                    high_val = None
        except Exception:
            points_str = None
        self.set_points(points_str)
        try:
            self.set_intensity(low_val, high_val)
        except Exception:
            pass
        # Axes: from WorkbenchWindow axis variables; annotate default source axes
        try:
            xlab = getattr(mw, 'axis_2d_x', None)
            ylab = getattr(mw, 'axis_2d_y', None)
            dx = xlab
            dy = ylab
            try:
                if isinstance(xlab, str) and xlab.strip().lower() in ("columns", "column"):
                    dx = f"{xlab}(Source)"
                if isinstance(ylab, str) and ylab.strip().lower() in ("row", "rows"):
                    dy = f"{ylab}(Source)"
            except Exception:
                pass
        except Exception:
            dx = None; dy = None
        self.set_axes(dx, dy)
