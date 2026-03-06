from typing import Optional, Tuple
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QFormLayout

from viewer.workbench.docks.information_dock_base import InformationDockBase


class Info3DDock(InformationDockBase):
    """Information dock specialized for 3D slice state (HKL only).

    Programmatically augments the base InformationDock UI with 3D-specific rows:
    - Orientation (HK, KL, HL, or Custom)
    - Slice position (orthogonal HKL axis/value; e.g., L = 1.23456, or n·origin = value)
    - Origin (H,K,L) with 5 decimals
    - Normal (H,K,L) with 5 decimals
    - H/K/L ranges across current slice points (min..max)
    - Image size (HxW) for rasterization target

    Reuses base fields: Total Points, Intensity Low/High.
    """

    def __init__(
        self,
        main_window=None,
        title: str = "3D Info",
        segment_name: Optional[str] = "3d",
        dock_area: Qt.DockWidgetArea = Qt.RightDockWidgetArea,
        show: bool = False,
    ):
        super().__init__(title=title, main_window=main_window, segment_name=segment_name, dock_area=dock_area, show=show)
        self._setup_extra_rows()

    # UI augmentation
    def _setup_extra_rows(self) -> None:
        try:
            form: QFormLayout = self._widget.findChild(QFormLayout, "formLayout")
            if form is None:
                return
            # Helper to add a row and keep refs
            def add_row(caption: str, obj_name: str) -> QLabel:
                cap = QLabel(caption, self._widget)
                val = QLabel("—", self._widget)
                val.setObjectName(obj_name)
                form.addRow(cap, val)
                return val

            # Orientation
            self.lbl_orientation = add_row("Orientation:", "lbl_orientation")
            self.lbl_orientation.setToolTip("Slice plane orientation in HKL coordinates")
            # Slice position
            self.lbl_slice_pos = add_row("Slice Position:", "lbl_slice_pos")
            self.lbl_slice_pos.setToolTip("Orthogonal axis value for axis-aligned planes, or n·origin for custom")
            # Origin
            self.lbl_origin = add_row("Origin (H,K,L):", "lbl_origin")
            self.lbl_origin.setToolTip("Slice plane origin in HKL coordinates")
            # Normal
            self.lbl_normal = add_row("Normal (H,K,L):", "lbl_normal")
            self.lbl_normal.setToolTip("Slice plane normal in HKL coordinates")
            # Ranges
            self.lbl_H_range = add_row("H range:", "lbl_H_range")
            self.lbl_K_range = add_row("K range:", "lbl_K_range")
            self.lbl_L_range = add_row("L range:", "lbl_L_range")
            self.lbl_H_range.setToolTip("Min..Max over slice points H component")
            self.lbl_K_range.setToolTip("Min..Max over slice points K component")
            self.lbl_L_range.setToolTip("Min..Max over slice points L component")
            # Image size
            self.lbl_image_size = add_row("Image size:", "lbl_image_size")
            self.lbl_image_size.setToolTip("Rasterization target size (HxW)")
        except Exception:
            pass

    # Public API
    def update_from_slice(
        self,
        slice_mesh,
        normal: np.ndarray,
        origin: np.ndarray,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update all labels based on a PyVista slice mesh and plane definition.
        - HKL-only computations (no U/V).
        - Intensity low/high from slice_mesh['intensity'] if present.
        - Image size prefers provided target_shape; falls back to 512×512.
        """
        try:
            # Points
            try:
                npts = int(getattr(slice_mesh, 'n_points', 0))
            except Exception:
                npts = None
            self.set_points(npts)

            # Intensities
            low = high = None
            try:
                vals = np.asarray(slice_mesh["intensity"], dtype=float).ravel()
                if vals.size > 0:
                    low = float(np.min(vals))
                    high = float(np.max(vals))
            except Exception:
                pass
            self.set_intensity(low, high)

            # Origin, Normal formatting (5 decimals)
            def fmt5(x: float) -> str:
                try:
                    return f"{float(x):.5f}"
                except Exception:
                    return str(x)

            try:
                o = np.array(origin, dtype=float).reshape(3)
            except Exception:
                o = np.array([np.nan, np.nan, np.nan], dtype=float)
            try:
                n = np.array(normal, dtype=float).reshape(3)
            except Exception:
                n = np.array([0.0, 0.0, 1.0], dtype=float)

            try:
                self.lbl_origin.setText(f"({fmt5(o[0])}, {fmt5(o[1])}, {fmt5(o[2])})")
            except Exception:
                pass
            try:
                self.lbl_normal.setText(f"({fmt5(n[0])}, {fmt5(n[1])}, {fmt5(n[2])})")
            except Exception:
                pass

            # Orientation and orthogonal axis
            orientation, uv_idxs, orth_label = self._infer_orientation_and_axes(n)
            try:
                orient_txt = orientation if orientation == "Custom" else f"{orientation} plane"
                self.lbl_orientation.setText(orient_txt)
            except Exception:
                pass

            # Slice position
            try:
                orth_val = None
                if orth_label == "L":
                    orth_val = float(o[2])
                elif orth_label == "H":
                    orth_val = float(o[0])
                elif orth_label == "K":
                    orth_val = float(o[1])
                if orth_val is not None:
                    self.lbl_slice_pos.setText(f"{orth_label} = {fmt5(orth_val)}")
                else:
                    # Custom: n·origin
                    try:
                        n_unit = n / (np.linalg.norm(n) or 1.0)
                    except Exception:
                        n_unit = n
                    try:
                        val = float(np.dot(n_unit, o))
                    except Exception:
                        val = float('nan')
                    self.lbl_slice_pos.setText(f"n·origin = {fmt5(val)}")
            except Exception:
                pass

            # Ranges over slice points
            try:
                pts = np.asarray(getattr(slice_mesh, 'points', np.empty((0, 3))), dtype=float)
            except Exception:
                pts = np.empty((0, 3), dtype=float)
            def fmt_range(arr: np.ndarray) -> str:
                if arr.size == 0:
                    return "—"
                try:
                    amin = float(np.min(arr))
                    amax = float(np.max(arr))
                    if not np.isfinite(amin) or not np.isfinite(amax):
                        return "—"
                    return f"{amin:.6g}..{amax:.6g}"
                except Exception:
                    return "—"
            try:
                self.lbl_H_range.setText(fmt_range(pts[:, 0] if pts.shape[1] >= 1 else np.array([])))
                self.lbl_K_range.setText(fmt_range(pts[:, 1] if pts.shape[1] >= 2 else np.array([])))
                self.lbl_L_range.setText(fmt_range(pts[:, 2] if pts.shape[1] >= 3 else np.array([])))
            except Exception:
                pass

            # Image size (HxW)
            try:
                if not target_shape or not isinstance(target_shape, (tuple, list)) or len(target_shape) != 2:
                    target_shape = (0, 0)
                try:
                    H, W = int(target_shape[0]), int(target_shape[1])
                except Exception:
                    H, W = 0, 0
                self.lbl_image_size.setText(f"HxW = {H}×{W}")
            except Exception:
                pass

            # Optional: set base axes to match orientation
            try:
                if orientation == "HK":
                    self.set_axes("H", "K")
                elif orientation == "KL":
                    self.set_axes("K", "L")
                elif orientation == "HL":
                    self.set_axes("H", "L")
                else:
                    self.set_axes("U", "V")
            except Exception:
                pass
        except Exception:
            # Keep errors contained
            pass

    # Logic reuse (HKL only)
    def _infer_orientation_and_axes(self, normal: np.ndarray) -> Tuple[str, Optional[Tuple[int, int]], Optional[str]]:
        """Infer slice orientation from the plane normal in HKL coordinates.
        Returns (orientation, (u_idx, v_idx) for axis-aligned mapping or None, orth_label).
        orientation in {'HK','KL','HL','Custom'}; u_idx/v_idx map to columns of pts (0:H, 1:K, 2:L).
        orth_label is the axis perpendicular to the plane ('L' for HK, 'H' for KL, 'K' for HL).
        """
        try:
            n = np.array(normal, dtype=float).reshape(3)
            n_norm = float(np.linalg.norm(n))
            if not np.isfinite(n_norm) or n_norm <= 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = n / n_norm
            X = np.array([1.0, 0.0, 0.0], dtype=float)  # H
            Y = np.array([0.0, 1.0, 0.0], dtype=float)  # K
            Z = np.array([0.0, 0.0, 1.0], dtype=float)  # L
            tol = 0.95
            dX = abs(float(np.dot(n, X)))
            dY = abs(float(np.dot(n, Y)))
            dZ = abs(float(np.dot(n, Z)))
            if dZ >= tol:
                # Normal ~ L → HK plane
                return "HK", (0, 1), "L"
            if dX >= tol:
                # Normal ~ H → KL plane
                return "KL", (1, 2), "H"
            if dY >= tol:
                # Normal ~ K → HL plane
                return "HL", (0, 2), "K"
            return "Custom", None, None
        except Exception:
            return "Custom", None, None
