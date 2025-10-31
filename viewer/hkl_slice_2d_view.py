import numpy as np
from typing import Optional, Tuple

from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg


class HKLSlice2DView(QWidget):
    """
    Lightweight 2D slice view that mirrors the current slice from the parent HKL3DSliceWindow.
    - No file I/O, no extra controls.
    - Inherits min/max intensity and colormap directly from the parent.
    - Updates are throttled with a QTimer to avoid re-rasterizing on every drag event.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        # Load .ui and setup host layout for embedded 2D plot
        uic.loadUi('gui/hkl_slice_2d_view.ui', self)

        # Plot with ImageView + PlotItem for axis labeling
        self.plot_item = pg.PlotItem()
        self.image_view = pg.ImageView(view=self.plot_item)
        # Axis labels via PlotItem
        self.plot_item.setLabel('bottom', 'U')
        self.plot_item.setLabel('left', 'V')
        # Optional: lock aspect for square pixels
        try:
            self.image_view.view.setAspectLocked(True)
        except Exception:
            pass
        try:
            self.layoutPlotHost.addWidget(self.image_view)
        except Exception:
            # Fallback: attach directly if layout not found
            fallback_layout = QVBoxLayout(self)
            fallback_layout.setContentsMargins(6, 6, 6, 6)
            fallback_layout.addWidget(self.image_view)
        # Add a lightweight text overlay to indicate slice orientation and value of orthogonal axis
        try:
            self._slice_info_text = pg.TextItem("", color="w", anchor=(0, 1))
            self.plot_item.addItem(self._slice_info_text)
        except Exception:
            self._slice_info_text = None

        # Pending update queue + throttle timer
        self._pending = None  # type: Optional[Tuple[object, np.ndarray, np.ndarray]]
        self._timer = QTimer(self)
        self._timer.setInterval(100)  # ~10 fps coalesced updates
        self._timer.timeout.connect(self._flush_pending)

        # Store initial parent settings for consistency
        self._last_synced_levels = None
        self._last_synced_colormap = None
        
        # Initial sync of display settings from parent
        try:
            self.sync_levels()
        except Exception:
            pass
        try:
            self.sync_colormap()
        except Exception:
            pass

    def schedule_update(self, slice_mesh, normal: np.ndarray, origin: np.ndarray) -> None:
        """
        Called by the parent after updating the 3D slice. Stores latest data and starts the coalescing timer.
        """
        try:
            # Store latest references; slice_mesh is a PyVista dataset (PolyData)
            self._pending = (slice_mesh, np.array(normal, dtype=float), np.array(origin, dtype=float))
            if not self._timer.isActive():
                self._timer.start()
        except Exception:
            # Silently ignore to avoid impacting parent
            pass

    def _flush_pending(self) -> None:
        """
        Timer slot: perform a single rasterization from the most recent pending slice and update the ImageItem,
        with axes labeled to HK/KL/HL and plot ranges set to physical coordinates.
        """
        try:
            self._timer.stop()
            if not self._pending:
                return
            slice_mesh, normal, origin = self._pending
            self._pending = None

            # Extract points and intensities from the PyVista slice mesh
            try:
                pts = np.asarray(slice_mesh.points, dtype=float)  # (N,3)
            except Exception:
                pts = np.empty((0, 3), dtype=float)
            try:
                vals = np.asarray(slice_mesh["intensity"], dtype=float).reshape(-1)
            except Exception:
                vals = np.zeros((len(pts),), dtype=float)

            if pts.size == 0 or vals.size == 0 or pts.shape[0] != vals.shape[0]:
                # Nothing to render
                return

            # Target raster shape: prefer parent's curr_shape, then orig_shape, else fallback
            target_shape = self._get_target_shape()
            H = max(int(target_shape[0]), 1)
            W = max(int(target_shape[1]), 1)

            # Rasterize to 2D image + axis ranges/orientation
            result = self._rasterize_to_image(pts, vals, normal, origin, H, W)
            if result is None:
                return
            image, U_min, U_max, V_min, V_max, orientation, orth_label, orth_value = result
            if image is None or (hasattr(image, "size") and image.size == 0):
                return

            # Update the image content
            try:
                self.image_view.setImage(
                    image.astype(np.float32),
                    autoLevels=False,
                    autoRange=False,
                    autoHistogramRange=False
                )
            except Exception:
                # Fallback to underlying ImageItem
                try:
                    self.image_view.imageItem.setImage(image.astype(np.float32), autoLevels=False)
                except Exception:
                    pass

            # Apply item transform to map pixels to physical HKL coordinates
            try:
                it = self.image_view.imageItem
                try:
                    it.resetTransform()
                except Exception:
                    try:
                        it.setTransform(pg.QtGui.QTransform())  # identity
                    except Exception:
                        pass
                sx = float(U_max - U_min) / float(W if W != 0 else 1)
                sy = float(V_max - V_min) / float(H if H != 0 else 1)
                if not np.isfinite(sx) or sx == 0.0:
                    sx = 1.0
                if not np.isfinite(sy) or sy == 0.0:
                    sy = 1.0
                try:
                    it.scale(sx, sy)
                    it.setPos(U_min, V_min)
                except Exception:
                    pass
            except Exception:
                pass

            # Set axis ranges and labels based on orientation
            try:
                self.plot_item.setXRange(U_min, U_max, padding=0)
                self.plot_item.setYRange(V_min, V_max, padding=0)
            except Exception:
                pass
            try:
                if orientation == "HK":
                    self.plot_item.setLabel('bottom', 'H')
                    self.plot_item.setLabel('left', 'K')
                elif orientation == "KL":
                    self.plot_item.setLabel('bottom', 'K')
                    self.plot_item.setLabel('left', 'L')
                elif orientation == "HL":
                    self.plot_item.setLabel('bottom', 'H')
                    self.plot_item.setLabel('left', 'L')
                else:
                    self.plot_item.setLabel('bottom', 'U')
                    self.plot_item.setLabel('left', 'V')
            except Exception:
                pass

            # Update slice info text (e.g., "HK plane (L = 1.23)")
            try:
                if getattr(self, "_slice_info_text", None):
                    txt = str(orientation)
                    if txt and txt != "Custom":
                        txt += " plane"
                    if orth_label is not None and orth_value is not None and np.isfinite(orth_value):
                        txt += f" ({orth_label} = {orth_value:.2f})"
                    self._slice_info_text.setText(txt)
                    try:
                        # Place near top-left of current view
                        self._slice_info_text.setPos(U_min, V_max)
                    except Exception:
                        pass
            except Exception:
                pass

            # Inherit levels and colormap
            self.sync_levels()
            self.sync_colormap()
        except Exception:
            # Keep errors contained to avoid breaking parent interactions
            pass

    def sync_levels(self) -> None:
        """
        Inherit min/max intensity levels from the parent and apply them to the ImageItem.
        Only updates if levels have changed to avoid unnecessary operations.
        """
        try:
            vmin = float(self.parent.sbMinIntensity.value())
            vmax = float(self.parent.sbMaxIntensity.value())
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            
            # Check if levels have changed
            current_levels = (vmin, vmax)
            if self._last_synced_levels != current_levels:
                try:
                    # ImageView supports setLevels(min, max)
                    self.image_view.setLevels(vmin, vmax)
                except Exception:
                    try:
                        self.image_view.imageItem.setLevels((vmin, vmax))
                    except Exception:
                        pass
                self._last_synced_levels = current_levels
        except Exception:
            pass

    def sync_colormap(self) -> None:
        """
        Inherit the current colormap from the parent and apply it to the ImageItem.
        Only updates if colormap has changed to avoid unnecessary operations.
        """
        try:
            cmap_name = str(self.parent.cbColorMapSelect.currentText())
        except Exception:
            cmap_name = "viridis"

        # Check if colormap has changed
        if self._last_synced_colormap == cmap_name:
            return
            
        lut = None
        # Try pyqtgraph ColorMap
        try:
            if hasattr(pg, "colormap") and hasattr(pg.colormap, "get"):
                try:
                    cmap = pg.colormap.get(cmap_name)
                except Exception:
                    # Some names may be in matplotlib but not in pg
                    cmap = None
                if cmap is not None:
                    lut = cmap.getLookupTable(nPts=256)
        except Exception:
            lut = None

        # Fallback via matplotlib if needed
        if lut is None:
            try:
                import matplotlib.cm as cm
                mpl_cmap = cm.get_cmap(cmap_name)
                # Build LUT as uint8 Nx3
                xs = np.linspace(0.0, 1.0, 256, dtype=float)
                colors = mpl_cmap(xs, bytes=True)  # returns Nx4 uint8
                lut = colors[:, :3]
            except Exception:
                # Last resort: grayscale
                xs = (np.linspace(0, 255, 256)).astype(np.uint8)
                lut = np.column_stack([xs, xs, xs])

        try:
            self.image_view.imageItem.setLookupTable(lut)
            self._last_synced_colormap = cmap_name
        except Exception:
            pass

    def sync_all_settings(self) -> None:
        """
        Synchronize all rendering settings from parent (levels, colormap, etc.).
        Called when the 2D view is first opened or when major changes occur.
        """
        try:
            # Force sync by clearing cached values
            self._last_synced_levels = None
            self._last_synced_colormap = None
            
            # Sync all settings
            self.sync_levels()
            self.sync_colormap()
            
            # Apply any reduction factor settings if available
            try:
                if hasattr(self.parent, '_applied_reduction_factor'):
                    # The 2D view inherits the same reduction as applied to the 3D view
                    pass  # Already handled through target shape
            except Exception:
                pass
                
        except Exception:
            pass

    def _get_target_shape(self) -> Tuple[int, int]:
        """
        Determine target (H, W) for the raster image based on parent's known shapes.
        Defaults to 512x512.
        """
        # Prefer current shape if valid
        try:
            cs = getattr(self.parent, "curr_shape", None)
            if isinstance(cs, (tuple, list)) and len(cs) == 2 and int(cs[0]) > 0 and int(cs[1]) > 0:
                return int(cs[0]), int(cs[1])
        except Exception:
            pass
        # Fallback to original shape
        try:
            os_ = getattr(self.parent, "orig_shape", None)
            if isinstance(os_, (tuple, list)) and len(os_) == 2 and int(os_[0]) > 0 and int(os_[1]) > 0:
                return int(os_[0]), int(os_[1])
        except Exception:
            pass
        # Default
        return (512, 512)

    def _infer_orientation_and_axes(self, normal: np.ndarray) -> Tuple[str, Optional[Tuple[int, int]], Optional[str]]:
        """
        Infer slice orientation from the plane normal.
        Returns (orientation, (u_idx, v_idx) for axis-aligned mapping or None, orth_label).
        Orientation is one of 'HK', 'KL', 'HL', or 'Custom'.
        u_idx/v_idx map to columns of pts (0:H, 1:K, 2:L).
        orth_label is the axis perpendicular to the plane ('L' for HK, 'H' for KL, 'K' for HL).
        """
        try:
            n = np.array(normal, dtype=float)
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

    def _rasterize_to_image(
        self,
        pts: np.ndarray,
        vals: np.ndarray,
        normal: np.ndarray,
        origin: np.ndarray,
        H: int,
        W: int,
    ) -> Optional[Tuple[np.ndarray, float, float, float, float, str, Optional[str], Optional[float]]]:
        """
        Rasterize the slice to an HxW image and compute physical axis ranges and orientation.
        Returns a tuple: (image, U_min, U_max, V_min, V_max, orientation, orth_label, orth_value)
        - orientation in {'HK','KL','HL','Custom'}
        - U/V correspond to physical axes when orientation is axis-aligned; otherwise derived basis projection.
        - orth_label/orth_value represent the axis perpendicular to the slice plane (e.g., 'L' and origin[2] for HK).
        """
        try:
            n = np.array(normal, dtype=float)
            o = np.array(origin, dtype=float)

            # Normalize normal
            n_norm = float(np.linalg.norm(n))
            if not np.isfinite(n_norm) or n_norm <= 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = n / n_norm

            orientation, uv_idxs, orth_label = self._infer_orientation_and_axes(n)

            if uv_idxs is not None:
                # Axis-aligned planes: use absolute HKL coordinates directly
                u_idx, v_idx = uv_idxs
                U = pts[:, u_idx].astype(float)
                V = pts[:, v_idx].astype(float)
                U_min, U_max = float(np.min(U)), float(np.max(U))
                V_min, V_max = float(np.min(V)), float(np.max(V))
                # Handle degenerate ranges
                if (not np.isfinite(U_min)) or (not np.isfinite(U_max)) or (U_max == U_min):
                    U_min, U_max = -0.5, 0.5
                if (not np.isfinite(V_min)) or (not np.isfinite(V_max)) or (V_max == V_min):
                    V_min, V_max = -0.5, 0.5
                # Weighted histogram (average)
                sum_img, _, _ = np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]], weights=vals)
                cnt_img, _, _ = np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]])
                with np.errstate(invalid="ignore", divide="ignore"):
                    img = np.zeros_like(sum_img, dtype=np.float32)
                    nz = cnt_img > 0
                    img[nz] = (sum_img[nz] / cnt_img[nz]).astype(np.float32)
                    img[~nz] = 0.0
                # Orthogonal axis value from origin
                orth_value = None
                try:
                    if orth_label == "L":
                        orth_value = float(o[2])
                    elif orth_label == "H":
                        orth_value = float(o[0])
                    elif orth_label == "K":
                        orth_value = float(o[1])
                except Exception:
                    orth_value = None
                return img, U_min, U_max, V_min, V_max, orientation, orth_label, orth_value

            # Custom orientation: fall back to in-plane basis projection
            # Choose a reference axis not parallel to n to make in-plane basis
            world_axes = [
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            ]
            ref = world_axes[0]
            for ax in world_axes:
                if abs(float(np.dot(ax, n))) < 0.9:
                    ref = ax
                    break
            u = np.cross(n, ref)
            u_norm = float(np.linalg.norm(u))
            if not np.isfinite(u_norm) or u_norm <= 0.0:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
                u = np.cross(n, ref)
                u_norm = float(np.linalg.norm(u))
                if not np.isfinite(u_norm) or u_norm <= 0.0:
                    u = np.array([1.0, 0.0, 0.0], dtype=float)
                    u_norm = 1.0
            u = u / u_norm
            v = np.cross(n, u)
            v_norm = float(np.linalg.norm(v))
            if not np.isfinite(v_norm) or v_norm <= 0.0:
                v = np.array([0.0, 1.0, 0.0], dtype=float)

            # Project points into plane coordinates (origin-relative for custom)
            rel = pts - o[None, :]
            U = rel.dot(u)  # shape (N,)
            V = rel.dot(v)  # shape (N,)

            # Handle degenerate ranges
            U_min, U_max = float(np.min(U)), float(np.max(U))
            V_min, V_max = float(np.min(V)), float(np.max(V))
            if not np.isfinite(U_min) or not np.isfinite(U_max) or (U_max == U_min):
                U_min, U_max = -0.5, 0.5
            if not np.isfinite(V_min) or not np.isfinite(V_max) or (V_max == V_min):
                V_min, V_max = -0.5, 0.5

            # Histogram to image
            sum_img, _, _ = np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]], weights=vals)
            cnt_img, _, _ = np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]])
            with np.errstate(invalid="ignore", divide="ignore"):
                img = np.zeros_like(sum_img, dtype=np.float32)
                nz = cnt_img > 0
                img[nz] = (sum_img[nz] / cnt_img[nz]).astype(np.float32)
                img[~nz] = 0.0

            # Orthogonal scalar position for custom
            try:
                orth_value = float(np.dot(n, o))
            except Exception:
                orth_value = None

            return img, U_min, U_max, V_min, V_max, "Custom", None, orth_value
        except Exception:
            return None
