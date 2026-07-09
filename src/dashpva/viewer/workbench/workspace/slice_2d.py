from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QWidget

from dashpva.gui import ui_path
from dashpva.utils.slice_raster import rasterize_slab
from dashpva.viewer.core.docks.base_dock import BaseDock

_AXIS_LABELS = {
    "HK": ("H", "K"),
    "KL": ("K", "L"),
    "HL": ("H", "L"),
}


class SliceView2D(QWidget):
    """View that renders the current 3D slice as a live 2D image.

    Hosted by :class:`Slice2DDock`. Fed by ``Workspace3D.on_plane_update`` via
    :meth:`schedule_update`, it rasterizes the slab into an HxW image with
    per-pixel HKL grids (see :func:`dashpva.utils.slice_raster.rasterize_slab`)
    and displays it in a PyQtGraph ImageView. Color levels autoscale to the
    slice data by default (toggle off to use the 3D view's min/max); a log-scale
    toggle and an orientation readout are also provided. Updates are throttled so
    dragging the plane stays responsive. The last result is cached for saving.

    Example:
        >>> view = SliceView2D(main_window=mw)          # doctest: +SKIP
        >>> view.schedule_update(slab, [0, 0, 1], [0, 0, 0.5], (0.0, 100.0))  # doctest: +SKIP
        >>> result = view.get_last_slice()              # doctest: +SKIP
    """

    def __init__(self, parent=None, main_window=None, title="Slice 2D"):
        super().__init__(parent)
        self.main_window = main_window
        self.title = title
        uic.loadUi(ui_path("workbench", "workspace", "slice_2d.ui"), self)
        self.setObjectName("SliceView2D")
        self._last = None            # cached rasterize_slab result + clim
        self._pending = None         # (points, intensities, normal, origin, clim)
        self._last_colormap = None
        # (lo, hi, log) of the levels last written to the histogram. Used so we
        # only push levels when they actually change, leaving manual histogram
        # adjustments untouched across ordinary slice updates.
        self._applied_levels_key = None
        self._build_plot()

        for _cb in ("cb_autoscale", "cb_log_scale"):
            w = getattr(self, _cb, None)
            if w is not None:
                try:
                    w.toggled.connect(self._redraw_last)
                except Exception:
                    pass

        self._timer = QTimer(self)
        self._timer.setInterval(100)  # ~10 fps coalesced updates
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._flush_pending)

    def _build_plot(self):
        self.plot_item = pg.PlotItem()
        self.image_view = pg.ImageView(view=self.plot_item)
        self.plot_item.setLabel('bottom', 'U')
        self.plot_item.setLabel('left', 'V')
        try:
            self.image_view.view.setAspectLocked(True)
        except Exception:
            pass
        try:
            self.layoutPlotHost.addWidget(self.image_view)
        except Exception:
            pass
        try:
            self.plot_item.scene().sigMouseMoved.connect(self._on_mouse_moved)
        except Exception:
            pass

    # ── Public API ──────────────────────────────────────────────────────────
    def schedule_update(self, slab, normal, origin, clim) -> None:
        """Queue a slab for (throttled) rasterization and display."""
        try:
            pts = np.asarray(slab.points, dtype=float)
            vals = np.asarray(slab["intensity"], dtype=float).reshape(-1)
        except Exception:
            return
        self._pending = (pts, vals, np.asarray(normal, dtype=float),
                         np.asarray(origin, dtype=float), tuple(clim))
        if not self._timer.isActive():
            self._timer.start()

    def get_last_slice(self) -> Optional[dict]:
        """Return the last rasterized slice dict (incl. clim), or None."""
        return self._last

    # ── Rendering ───────────────────────────────────────────────────────────
    def _flush_pending(self) -> None:
        if not self._pending:
            return
        pts, vals, normal, origin, clim = self._pending
        self._pending = None
        vmin, vmax = (clim[0], clim[1]) if clim[0] <= clim[1] else (clim[1], clim[0])

        # Show all slab points; the current intensity range is applied as the
        # image's color levels (below vmin / above vmax are clamped), not as a
        # threshold that hides points.
        if pts.shape[0] == 0:
            self._last = None
            try:
                self.image_view.clear()
                self.lblSliceInfo.setText("No slice points to display.")
            except Exception:
                pass
            return

        shape = getattr(self.main_window.tab_3d, 'orig_shape', None) if self.main_window else None
        result = rasterize_slab(pts, vals, normal, origin, shape=shape)
        if result is None:
            return
        result["clim"] = (float(vmin), float(vmax))
        result["num_points"] = int(pts.shape[0])
        self._last = result

        self._display(result, (vmin, vmax))

    def _display(self, result: dict, clim) -> None:
        image = result["image"]
        U_min, U_max = result["u_range"]
        V_min, V_max = result["v_range"]
        H, W = image.shape

        log_on = bool(getattr(self, "cb_log_scale", None) is not None
                      and self.cb_log_scale.isChecked())
        disp = (np.log1p(np.maximum(image, 0.0)) if log_on else image).astype(np.float32)

        autoscale = bool(getattr(self, "cb_autoscale", None) is None
                         or self.cb_autoscale.isChecked())
        # Autoscale re-levels to the slice's own data each draw. Otherwise the
        # histogram range only rescales when the display domain flips
        # (linear <-> log) or on the first draw, so it stays put.
        prev = self._applied_levels_key
        domain_flip = prev is None or prev[2] != log_on
        try:
            self.image_view.setImage(disp, autoLevels=autoscale, autoRange=False,
                                     autoHistogramRange=(autoscale or domain_flip))
        except Exception:
            return

        # Map pixels to physical U/V coordinates.
        try:
            it = self.image_view.imageItem
            it.resetTransform()
            sx = (U_max - U_min) / float(W) if W else 1.0
            sy = (V_max - V_min) / float(H) if H else 1.0
            if not np.isfinite(sx) or sx == 0.0:
                sx = 1.0
            if not np.isfinite(sy) or sy == 0.0:
                sy = 1.0
            it.scale(sx, sy)
            it.setPos(U_min, V_min)
            self.plot_item.setXRange(U_min, U_max, padding=0)
            self.plot_item.setYRange(V_min, V_max, padding=0)
        except Exception:
            pass

        # Axis labels from orientation.
        xlabel, ylabel = _AXIS_LABELS.get(result["orientation"], ("U", "V"))
        try:
            self.plot_item.setLabel('bottom', xlabel)
            self.plot_item.setLabel('left', ylabel)
        except Exception:
            pass

        if autoscale:
            # setImage(autoLevels=True) already levelled to the data; drop the
            # cached key so manual levels re-apply cleanly when autoscale is off.
            self._applied_levels_key = None
        else:
            # Levels in the display domain (log1p when log scale is on).
            c0, c1 = float(clim[0]), float(clim[1])
            if log_on:
                lo = float(np.log1p(max(0.0, c0)))
                hi = float(np.log1p(max(0.0, c1)))
            else:
                lo, hi = c0, c1
            # Push levels only when they changed (control edit or log toggle); a
            # plain slice update reuses whatever is in the histogram so a manual
            # vmin/vmax stays set.
            levels_key = (round(lo, 9), round(hi, 9), log_on)
            if levels_key != self._applied_levels_key:
                try:
                    self.image_view.setLevels(lo, hi)
                    self._applied_levels_key = levels_key
                except Exception:
                    pass
        self._apply_colormap()

        # Orientation readout only.
        try:
            txt = result["orientation"]
            if txt != "Custom":
                txt += " plane"
            if result["orth_label"] and result["orth_value"] is not None:
                txt += f" ({result['orth_label']} = {result['orth_value']:.4f})"
            self.lblSliceInfo.setText(txt)
        except Exception:
            pass

    def _redraw_last(self, _checked: bool = False) -> None:
        """Re-render the cached slice (e.g. when a display toggle changes)."""
        if self._last is not None:
            self._display(self._last, self._last["clim"])

    def _apply_colormap(self) -> None:
        try:
            cmap_name = str(self.main_window.tab_3d.cb_colormap_3d.currentText())
        except Exception:
            cmap_name = "viridis"
        if cmap_name == self._last_colormap:
            return
        lut = None
        try:
            cmap = pg.colormap.get(cmap_name)
            if cmap is not None:
                lut = cmap.getLookupTable(nPts=256)
        except Exception:
            lut = None
        if lut is None:
            try:
                import matplotlib.cm as cm
                colors = cm.get_cmap(cmap_name)(np.linspace(0.0, 1.0, 256), bytes=True)
                lut = colors[:, :3]
            except Exception:
                xs = np.linspace(0, 255, 256).astype(np.uint8)
                lut = np.column_stack([xs, xs, xs])
        try:
            self.image_view.imageItem.setLookupTable(lut)
            self._last_colormap = cmap_name
        except Exception:
            pass

    # ── Hover HKL readout ───────────────────────────────────────────────────
    def _on_mouse_moved(self, pos) -> None:
        if self._last is None:
            return
        try:
            vb = self.plot_item.getViewBox()
            if not self.plot_item.sceneBoundingRect().contains(pos):
                return
            pt = vb.mapSceneToView(pos)
            u, v = float(pt.x()), float(pt.y())
            U_min, U_max = self._last["u_range"]
            V_min, V_max = self._last["v_range"]
            H, W = self._last["image"].shape
            col = int((u - U_min) / (U_max - U_min) * W) if U_max > U_min else 0
            row = int((v - V_min) / (V_max - V_min) * H) if V_max > V_min else 0
            if not (0 <= col < W and 0 <= row < H):
                return
            H_v = float(self._last["qx"][row, col])
            K_v = float(self._last["qy"][row, col])
            L_v = float(self._last["qz"][row, col])
            I_v = float(self._last["image"][row, col])
            base = self.lblSliceInfo.text().split("   |   ")[0]
            self.lblSliceInfo.setText(
                f"{base}   |   H={H_v:.4f} K={K_v:.4f} L={L_v:.4f}  I={I_v:.4g}")
        except Exception:
            pass


class Slice2DDock(BaseDock):
    """Dockable panel hosting the live 2D slice view (:class:`SliceView2D`).

    Replaces the former Slice 2D analysis tab; ``Workspace3D.show_slice_2d_tab``
    shows and raises this dock. The hosted view is exposed as ``.view`` so the
    3D workspace can push slabs into it via ``schedule_update``.

    Example:
        >>> dock = Slice2DDock(main_window=mw)          # doctest: +SKIP
        >>> dock.view.get_last_slice()                  # doctest: +SKIP
    """

    def __init__(self, main_window=None, title: str = "Slice 2D", segment_name: str = "3d",
                 dock_area=Qt.RightDockWidgetArea, show: bool = False):
        super().__init__(title=title, main_window=main_window, segment_name=segment_name,
                         dock_area=dock_area, show=show)
        self.view = SliceView2D(main_window=main_window)
        self.setWidget(self.view)
