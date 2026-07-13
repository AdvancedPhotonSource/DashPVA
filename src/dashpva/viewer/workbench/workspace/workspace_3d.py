import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from natsort import natsorted
from PyQt5.QtCore import QSettings, Qt, QThread, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import dashpva.settings as app_settings

# Import BaseTab using existing tabs package alias
from dashpva.gui import ui_path

from .base_tab import BaseTab
from .range_slider import RangeSlider

# Import 3D visualization components
try:
    import pyvista as pv
    import pyvista as pyv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    pv = None
    PYVISTA_AVAILABLE = False

# Worker for off-UI-thread 3D prep
from dashpva.viewer.workbench.workers import Render3D


class Workspace3D(BaseTab):
    """
    3D Tab encapsulating 3D viewer setup, loading, and plotting operations.
    Delegates UI widget access via main_window, but centralizes 3D actions here.
    """
    def __init__(self, parent=None, main_window=None, title="3D View"):
        pv.set_plot_theme('dark')
        try: 
            super().__init__(ui_file=ui_path("workbench", "workspace", "workspace_3d.ui"), parent=parent, main_window=main_window, title=title)
            self.title = title
            self.main_window = main_window 
            self.build()
            self.connect_all()
        except Exception as e:
            try:
                self.main_window.update_status(f"Error initializing 3D workspace: {e}")
            except Exception:
                pass

    def connect_all(self):
        """Wire up 3D controls to main window handlers."""
        try:
            self.btn_load_3d_data.clicked.connect(self.load_data)
            self.cb_colormap_3d.currentTextChanged.connect(self.on_3d_colormap_changed)
            self.cb_show_points.toggled.connect(self.toggle_3d_points)
            self.cb_show_slice.toggled.connect(self.toggle_3d_slice)
            self.sb_min_intensity_3d.editingFinished.connect(self.update_intensity)
            self.sb_max_intensity_3d.editingFinished.connect(self.update_intensity)
            for sl, da, db, lbl, sp_lo, sp_hi, btn in self._hkl_widgets():
                sl.range_changed.connect(self._make_slider_cb(sl, da, db, lbl, sp_lo, sp_hi))
                sp_lo.valueChanged.connect(self._make_spin_cb(sl, da, db, lbl, sp_lo, sp_hi))
                sp_hi.valueChanged.connect(self._make_spin_cb(sl, da, db, lbl, sp_lo, sp_hi))
                btn.clicked.connect(sl.setFullRange)
            self.rb_downsample_on.toggled.connect(self._on_downsample_changed)
            self.btn_save_3d_config.clicked.connect(self.save_3d_config)
            self.btn_load_3d_config.clicked.connect(self.load_3d_config)
            # Folder playback controls
            self.btn_load_3d_folder.clicked.connect(self.browse_folder)
            self.btn_prev_frame.clicked.connect(self.prev_frame)
            self.btn_next_frame.clicked.connect(self.next_frame)
            self.btn_play_pause.toggled.connect(self.toggle_play)
            self.sb_fps.valueChanged.connect(self.set_fps)
            self.slider_frame.valueChanged.connect(self._on_slider_changed)
            self.cb_playback_signal.currentIndexChanged.connect(lambda _=None: self._on_signal_changed())
        except Exception as e:
            try:
                self.main_window.update_status(f"Error setting up 3D connections: {e}")
            except Exception:
                pass
    
    def build(self):
        # Folder playback / frame session state (independent of the plotter)
        self._init_playback_state()

        # Try to create VTK QtInteractor; fall back if unavailable
        try:
            self.plotter = QtInteractor(self)
        except Exception:
            self.plotter = None

        self.hkl_info_label = None

        if self.plotter is None:
            placeholder = QLabel("3D (VTK) unavailable in tunnel mode.")
            try:
                placeholder.setAlignment(Qt.AlignCenter)
            except Exception:
                pass
            try:
                placeholder.setWordWrap(True)
            except Exception:
                pass
            try:
                self.container.insertWidget(1, placeholder, stretch=1)
            except Exception:
                pass
            # Disable 3D controls that would require the plotter
            for w in [getattr(self, "btn_load_3d_data", None),
                      getattr(self, "cb_show_points", None),
                      getattr(self, "cb_show_slice", None),
                      getattr(self, "sb_min_intensity_3d", None),
                      getattr(self, "sb_max_intensity_3d", None)]:
                try:
                    if w is not None:
                        w.setEnabled(False)
                except Exception:
                    pass
            # Initialize defaults
            self.cloud_mesh_3d = None
            self.slab_actor = None
            self.plane_widget = None
            self.lut = None
            self.lut2 = None
            self._h_min_data = self._h_max_data = None
            self._k_min_data = self._k_max_data = None
            self._l_min_data = self._l_max_data = None
            self._h_slider = RangeSlider()
            self._k_slider = RangeSlider()
            self._l_slider = RangeSlider()
            self._btn_reset_h = QPushButton()
            self._btn_reset_k = QPushButton()
            self._btn_reset_l = QPushButton()
            self._h_val_lbl = QLabel()
            self._k_val_lbl = QLabel()
            self._l_val_lbl = QLabel()
            self._h_min_spin = QDoubleSpinBox()
            self._h_max_spin = QDoubleSpinBox()
            self._k_min_spin = QDoubleSpinBox()
            self._k_max_spin = QDoubleSpinBox()
            self._l_min_spin = QDoubleSpinBox()
            self._l_max_spin = QDoubleSpinBox()
            # Default target raster shape (HxW) for slice rasterization
            self.orig_shape = (0, 0)
            self.curr_shape = (0, 0)
            # Slice & Camera defaults
            self._slice_translate_step = 0.01
            self._slice_rotate_step_deg = 1.0
            self._custom_normal = np.array([0.0, 0.0, 1.0], dtype=float)
            self._zoom_step = 1.5
            return

        # If plotter exists, proceed to embed and configure
        self.container.insertWidget(1, self.plotter, stretch=1)
        try:
            self.scrollArea_3d_controls.setMinimumWidth(280)
        except Exception:
            pass
        try:
            self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L', x_color='red', y_color='green', z_color='blue')
        except Exception:
            pass

        # Color axes: H=red (X), K=green (Y), L=blue (Z)
        try:
            ca = getattr(self.plotter.renderer, 'cube_axes_actor', None)
            if ca:
                ca.GetXAxesLinesProperty().SetColor(1.0, 0.0, 0.0)
                ca.GetYAxesLinesProperty().SetColor(0.0, 1.0, 0.0)
                ca.GetZAxesLinesProperty().SetColor(0.0, 0.0, 1.0)
                ca.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
                ca.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
                ca.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
                ca.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)
                ca.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)
                ca.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)
        except Exception:
            pass
        self.cloud_mesh_3d = None
        self.slab_actor = None
        self.plane_widget = None
        # Initialize LUTs similar to viewer/hkl_3d.py
        try:
            self.lut = pv.LookupTable(cmap='jet')
            self.lut.apply_opacity([0, 1])
            self.lut2 = pv.LookupTable(cmap='jet')
            self.lut2.apply_opacity([0, 1])
            # Sync initial LUTs to the UI-selected colormap and current intensity controls
            try:
                self.on_3d_colormap_changed()
            except Exception:
                pass
        except Exception:
            self.lut = None
            self.lut2 = None
        # Default target raster shape (HxW) for slice rasterization
        self.orig_shape = (0, 0)
        self.curr_shape = (0, 0)
        # Slice & Camera defaults
        self._slice_translate_step = 0.01
        self._slice_rotate_step_deg = 1.0
        self._custom_normal = np.array([0.0, 0.0, 1.0], dtype=float)
        self._zoom_step = 1.5
        # Cached true data intensity bounds (set on data load)
        self._data_intensity_min = None
        self._data_intensity_max = None
        # Full point cloud — source of truth for downsample toggle and HKL range filter
        self._raw_points = None
        self._raw_intensities = None
        # Point cloud passed to the renderer (strided or full)
        self._display_points = None
        self._display_intensities = None
        # Data extents for mapping slider 0-1000 to actual HKL float values
        self._h_min_data = self._h_max_data = None
        self._k_min_data = self._k_max_data = None
        self._l_min_data = self._l_max_data = None

        # Build the HKL Range tab widget (Slider tab + Manual tab)
        self._build_hkl_range_section()

    def _build_hkl_range_section(self):
        """Populate layout_hkl_range with a two-tab widget: Slider and Manual."""
        tab = QTabWidget()
        tab.setTabPosition(QTabWidget.North)

        # ── Slider tab ──────────────────────────────────────────────────────
        s_widget = QWidget()
        s_layout = QVBoxLayout(s_widget)
        s_layout.setSpacing(4)
        s_layout.setContentsMargins(4, 4, 4, 4)

        self._h_slider = RangeSlider()
        self._k_slider = RangeSlider()
        self._l_slider = RangeSlider()
        self._btn_reset_h = QPushButton('↺')
        self._btn_reset_k = QPushButton('↺')
        self._btn_reset_l = QPushButton('↺')
        self._h_val_lbl = QLabel('—')
        self._k_val_lbl = QLabel('—')
        self._l_val_lbl = QLabel('—')

        for axis, sl, val_lbl, btn in (
            ('H', self._h_slider, self._h_val_lbl, self._btn_reset_h),
            ('K', self._k_slider, self._k_val_lbl, self._btn_reset_k),
            ('L', self._l_slider, self._l_val_lbl, self._btn_reset_l),
        ):
            btn.setFixedWidth(26)
            btn.setToolTip(f'Reset {axis} to full data range')
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setObjectName("val_label")

            sl_col = QVBoxLayout()
            sl_col.setSpacing(0)
            sl_col.addWidget(sl)
            sl_col.addWidget(val_lbl)

            row = QHBoxLayout()
            row.setSpacing(4)
            lbl = QLabel(axis)
            lbl.setFixedWidth(14)
            row.addWidget(lbl)
            row.addLayout(sl_col, 1)
            row.addWidget(btn)
            s_layout.addLayout(row)

        tab.addTab(s_widget, 'Slider')

        # ── Manual tab ──────────────────────────────────────────────────────
        m_widget = QWidget()
        m_layout = QFormLayout(m_widget)
        m_layout.setSpacing(4)
        m_layout.setContentsMargins(4, 4, 4, 4)

        self._h_min_spin = QDoubleSpinBox()
        self._h_max_spin = QDoubleSpinBox()
        self._k_min_spin = QDoubleSpinBox()
        self._k_max_spin = QDoubleSpinBox()
        self._l_min_spin = QDoubleSpinBox()
        self._l_max_spin = QDoubleSpinBox()

        for axis, sp_lo, sp_hi in (
            ('H', self._h_min_spin, self._h_max_spin),
            ('K', self._k_min_spin, self._k_max_spin),
            ('L', self._l_min_spin, self._l_max_spin),
        ):
            for sp in (sp_lo, sp_hi):
                sp.setRange(-9999.0, 9999.0)
                sp.setDecimals(3)
                sp.setSingleStep(0.01)
            pair = QHBoxLayout()
            pair.setSpacing(2)
            pair.addWidget(sp_lo)
            pair.addWidget(QLabel('→'))
            pair.addWidget(sp_hi)
            m_layout.addRow(f'{axis}:', pair)

        tab.addTab(m_widget, 'Manual')

        try:
            self.layout_hkl_range.addWidget(tab)
        except Exception:
            pass

    # ── HKL range helpers ───────────────────────────────────────────────────
    def _hkl_widgets(self):
        """Yield (slider, d_min_attr, d_max_attr, val_lbl, sp_lo, sp_hi, btn) for H, K, L."""
        return [
            (self._h_slider, '_h_min_data', '_h_max_data', self._h_val_lbl, self._h_min_spin, self._h_max_spin, self._btn_reset_h),
            (self._k_slider, '_k_min_data', '_k_max_data', self._k_val_lbl, self._k_min_spin, self._k_max_spin, self._btn_reset_k),
            (self._l_slider, '_l_min_data', '_l_max_data', self._l_val_lbl, self._l_min_spin, self._l_max_spin, self._btn_reset_l),
        ]

    def _hkl_from_slider(self, sl_low, sl_high, d_min, d_max):
        if d_min is None or d_max is None:
            return 0.0, 0.0
        span = d_max - d_min
        return d_min + sl_low / 1000.0 * span, d_min + sl_high / 1000.0 * span

    def _slider_int(self, hkl_val, d_min, d_max):
        if d_min is None or d_max is None:
            return 0
        span = d_max - d_min
        if span == 0:
            return 0
        return max(0, min(1000, int(round((hkl_val - d_min) / span * 1000))))

    def _make_slider_cb(self, sl, da, db, lbl, sp_lo, sp_hi):
        def cb(low, high):
            f0, f1 = self._hkl_from_slider(low, high, getattr(self, da), getattr(self, db))
            lbl.setText(f'{f0:.3f} → {f1:.3f}')
            for sp, v in ((sp_lo, f0), (sp_hi, f1)):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
            self.update_hkl_range()
        return cb

    def _make_spin_cb(self, sl, da, db, lbl, sp_lo, sp_hi):
        def cb():
            d_min, d_max = getattr(self, da), getattr(self, db)
            f0, f1 = sp_lo.value(), sp_hi.value()
            sl.blockSignals(True)
            sl.setLow(self._slider_int(f0, d_min, d_max), emit=False)
            sl.setHigh(self._slider_int(f1, d_min, d_max), emit=False)
            sl.blockSignals(False)
            sl.update()
            lbl.setText(f'{f0:.3f} → {f1:.3f}')
            self.update_hkl_range()
        return cb

    def setup_plot_viewer(self):
        """
        Create and embed a PyVista QtInteractor into the 3D tab container.
        """
        mw = self.main_window
        try:
            if not PYVISTA_AVAILABLE:
                return
            pyv.set_plot_theme('dark')
            mw.plotter_3d = QtInteractor()
            mw.plotter_3d.add_axes(xlabel='H', ylabel='K', zlabel='L', x_color='red', y_color='green', z_color='blue')
            if hasattr(mw, 'layout3DPlotHost') and mw.layout3DPlotHost is not None:
                try:
                    # layout3DPlotHost may be a grid layout from the UI
                    mw.layout3DPlotHost.addWidget(mw.plotter_3d, 0, 0)
                except Exception:
                    mw.layout3DPlotHost.addWidget(mw.plotter_3d)
            else:
                try:
                    mw.update_status("Warning: layout3DPlotHost not found, 3D plot may not display correctly")
                except Exception:
                    pass
            # Clear initial state
            self.clear_plot()
        except Exception as e:
            try:
                mw.update_status(f"Error setting up 3D plot viewer: {e}")
            except Exception:
                pass

    def toggle_3d_points(self, checked: bool):
        """Shows/Hides the main HKL point cloud."""
        try:
            # Support either actor name used by different paths
            actor = None
            if "points" in getattr(self.plotter, 'actors', {}):
                actor = self.plotter.actors.get("points")
            elif "cloud_volume" in getattr(self.plotter, 'actors', {}):
                actor = self.plotter.actors.get("cloud_volume")
            if actor is not None:
                actor.SetVisibility(bool(checked))
                self.plotter.render()
        except Exception:
            pass

    def toggle_3d_slice(self, checked: bool):
        """Shows/Hides the interactive plane and the extracted slice points."""
        try:
            # Toggle the points extracted by the plane
            if "slab_points" in getattr(self.plotter, 'actors', {}):
                try:
                    self.plotter.actors["slab_points"].SetVisibility(bool(checked))
                except Exception:
                    try:
                        self.plotter.renderer._actors["slab_points"].SetVisibility(bool(checked))
                    except Exception:
                        pass

            # Toggle the interactive plane widget tool
            if self.plane_widget is not None:
                try:
                    if checked:
                        # Try both enable methods to support different versions
                        try:
                            self.plane_widget.EnabledOn()
                        except Exception:
                            self.plane_widget.On()
                    else:
                        try:
                            self.plane_widget.EnabledOff()
                        except Exception:
                            self.plane_widget.Off()
                except Exception:
                    pass
            else:
                # Fallback: use plotter.plane_widgets list if available
                widgets = getattr(self.plotter, "plane_widgets", [])
                for pw in widgets or []:
                    try:
                        if checked:
                            pw.EnabledOn()
                        else:
                            pw.EnabledOff()
                    except Exception:
                        pass

            self.plotter.render()
        except Exception:
            pass

    def on_3d_colormap_changed(self):
        """Apply selected colormap to the points (and slab if available)."""
        try:
            cmap_name = getattr(self.cb_colormap_3d, 'currentText', lambda: 'viridis')()
        except Exception:
            cmap_name = 'viridis'
        # Primary LUT used for the main cloud volume/points
        try:
            self.lut = pv.LookupTable(cmap=cmap_name)
            self.lut.apply_opacity([0, 1])
            self.lut2 = pv.LookupTable(cmap=cmap_name)
            self.lut2.apply_opacity([0, 1])
            # Keep LUT scalar ranges in sync with current controls or cached bounds
            try:
                # Prefer current UI intensity range
                vmin = float(self.sb_min_intensity_3d.value())
                vmax = float(self.sb_max_intensity_3d.value())
            except Exception:
                # Fallback to data bounds if UI unavailable
                vmin = getattr(self, '_data_intensity_min', None)
                vmax = getattr(self, '_data_intensity_max', None)
            try:
                if vmin is not None and vmax is not None:
                    # Ensure proper ordering and non-zero span
                    if vmin > vmax:
                        vmin, vmax = vmax, vmin
                    if vmin == vmax:
                        vmax = vmin + 1e-6
                    self.lut.scalar_range = (vmin, vmax)
                    self.lut2.scalar_range = (vmin, vmax)
            except Exception:
                pass
        except Exception:
            self.lut = None
            self.lut2 = None

        # Update points/cloud actor by changing the mapper's lookup table (no re-add)
        try:
            actors = getattr(self.plotter, 'actors', {}) or {}
            tgt_name = 'points' if 'points' in actors else ('cloud_volume' if 'cloud_volume' in actors else None)
            if tgt_name and self.lut is not None:
                actor = actors.get(tgt_name)
                try:
                    # Prefer direct mapper property
                    actor.mapper.lookup_table = self.lut
                except Exception:
                    try:
                        actor.GetMapper().SetLookupTable(self.lut)
                    except Exception:
                        pass
                # Maintain scalar range and visibility
                try:
                    rng = [self.sb_min_intensity_3d.value(), self.sb_max_intensity_3d.value()]
                    actor.mapper.scalar_range = rng
                except Exception:
                    pass
                try:
                    actor.SetVisibility(bool(self.cb_show_points.isChecked()))
                except Exception:
                    pass
        except Exception:
            pass

        # Attempt to update slab colormap (best-effort) using a separate LUT
        try:
            # Use the cached actor if present otherwise lookup by name
            slab_actor = self.slab_actor
            if slab_actor is None:
                actors = getattr(self.plotter, 'actors', {}) or {}
                slab_actor = actors.get('slab_points')
            if slab_actor is not None:
                try:
                    # Apply LUT to mapper
                    try:
                        slab_actor.mapper.lookup_table = (self.lut2 or self.lut)
                    except Exception:
                        slab_actor.GetMapper().SetLookupTable(self.lut2 or self.lut)
                except Exception:
                    pass
                # Apply scalar range from UI if available
                try:
                    rng = [self.sb_min_intensity_3d.value(), self.sb_max_intensity_3d.value()]
                    slab_actor.mapper.scalar_range = rng
                except Exception:
                    pass
                # Keep visibility consistent
                try:
                    slab_actor.SetVisibility(bool(self.cb_show_slice.isChecked()))
                except Exception:
                    pass
        except Exception:
            pass

        # Render and ensure ranges/visibility remain in sync
        try:
            # Update existing scalar bars with the new LUT
            if hasattr(self.plotter, 'scalar_bars') and self.lut is not None:
                for _, scalar_bar in self.plotter.scalar_bars.items():
                    try:
                        scalar_bar.SetLookupTable(self.lut)
                        scalar_bar.Modified()
                    except Exception:
                        pass
            self.plotter.render()
        except Exception:
            pass
        try:
            self.update_intensity()
        except Exception:
            pass


    # === Clear ===
    def clear_plot(self):
        try:
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.clear()
                self.current_3d_data = None
                self.mesh = None
        except Exception as e:
            try:
                self.main_window.update_status(f"Error clearing 3D plot: {e}")
            except Exception:
                pass

    # === Loading & Plotting ===
    def load_data(self):
        """Load dataset and render using the tab's local plotter."""
        mw = self.main_window
        try:
            mw.update_status("Loading data into 3D viewer...")
        except Exception:
            pass
        try:
            if not PYVISTA_AVAILABLE:
                QMessageBox.warning(self, "3D Viewer", "PyVista is not available.")
                return

            # 1. Get the file path
            file_path = getattr(mw, 'current_file_path', None) or getattr(mw, 'selected_dataset_path', None)
            if not file_path:
                file_name, _ = QFileDialog.getOpenFileName(
                    self, 'Select HDF5 or VTI File', '', 'HDF5 Files (*.h5 *.hdf5 *.vti);;All Files (*)'
                )
                if not file_name:
                    return
                file_path = file_name
            # Indeterminate progress indicator while loading + rendering. The raw
            # load runs inside the worker thread (below), so the bar stays live.
            self._load_progress = QProgressDialog("Loading 3D data…", None, 0, 0, self)
            self._load_progress.setWindowTitle("Loading")
            self._load_progress.setWindowModality(Qt.WindowModal)
            self._load_progress.setCancelButton(None)
            self._load_progress.setMinimumDuration(0)
            self._load_progress.show()
            QApplication.processEvents()

            try:
                downsample = self.rb_downsample_on.isChecked()
            except Exception:
                downsample = True

            # 3. Define what happens when the worker finishes processing
            def _on_ready():
                try:
                    # The worker loaded + prepared the arrays off the UI thread.
                    worker = self._render3d_worker
                    points = worker.points
                    intensities = worker.intensities
                    shape = worker.shape
                    # Always keep the full dataset for the downsampling toggle.
                    self._raw_points = worker.raw_points
                    self._raw_intensities = worker.raw_intensities
                    # IMPORTANT: Tell the worker to plot to THIS tab's plotter
                    # We pass 'self.plotter' instead of 'mw'
                    self._render3d_worker.plot_3d_points(self)
                    # Cache a reference to the main points/cloud actor for fast updates
                    try:
                        if "points" in self.plotter.actors:
                            self.points_actor = self.plotter.actors.get("points")
                        elif "cloud_volume" in self.plotter.actors:
                            self.points_actor = self.plotter.actors.get("cloud_volume")
                    except Exception:
                        self.points_actor = None

                    # Cache true data intensity bounds and set LUT scalar ranges
                    try:
                        self._data_intensity_min = float(np.min(intensities))
                        self._data_intensity_max = float(np.max(intensities))
                        if self.lut is not None:
                            self.lut.scalar_range = (self._data_intensity_min, self._data_intensity_max)
                        if self.lut2 is not None:
                            self.lut2.scalar_range = (self._data_intensity_min, self._data_intensity_max)
                    except Exception:
                        pass
                    # Ensure the currently selected colormap is applied immediately on load
                    try:
                        self.on_3d_colormap_changed()
                    except Exception:
                        pass
                    # Apply LUTs to actors
                    try:
                        if self.points_actor is not None and self.lut is not None:
                            self.points_actor.mapper.lookup_table = self.lut
                    except Exception:
                        pass
                    try:
                        if self.slab_actor is not None and (self.lut2 or self.lut) is not None:
                            self.slab_actor.mapper.lookup_table = (self.lut2 or self.lut)
                    except Exception:
                        pass
                    # Show bounds like in hkl_3d
                    try:
                        self.plotter.show_bounds(
                            mesh=self.points_actor.mapper.input if self.points_actor is not None else None,
                            xtitle='H Axis', ytitle='K Axis', ztitle='L Axis',
                            ticks='inside', minor_ticks=True,
                            n_xlabels=7, n_ylabels=7, n_zlabels=7,
                            x_color='red', y_color='green', z_color='blue',
                            font_size=20
                        )
                    except Exception:
                        pass
                    # Sync scalar bars to primary LUT
                    try:
                        if hasattr(self.plotter, 'scalar_bars') and self.lut is not None:
                            for _, sb in self.plotter.scalar_bars.items():
                                try:
                                    sb.SetLookupTable(self.lut)
                                    sb.Modified()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Ensure visibility respects checkboxes
                    try:
                        self.toggle_3d_points(self.cb_show_points.isChecked())
                        self.toggle_3d_slice(self.cb_show_slice.isChecked())
                    except Exception:
                        pass

                    # Align current intensity range with data bounds and reflect in UI
                    try:
                        self.update_intensity()
                    except Exception:
                        pass

                    # Match the slice raster resolution to the detector frame shape.
                    try:
                        if isinstance(shape, (tuple, list)) and len(shape) == 2:
                            self.orig_shape = (int(shape[0]), int(shape[1]))
                    except Exception:
                        pass

                    # Store display arrays and seed HKL range sliders + spinboxes
                    try:
                        self._display_points = points
                        self._display_intensities = intensities
                        if points is not None and len(points) > 0:
                            self._h_min_data = float(np.min(points[:, 0]))
                            self._h_max_data = float(np.max(points[:, 0]))
                            self._k_min_data = float(np.min(points[:, 1]))
                            self._k_max_data = float(np.max(points[:, 1]))
                            self._l_min_data = float(np.min(points[:, 2]))
                            self._l_max_data = float(np.max(points[:, 2]))
                            for sl, da, db, lbl, sp_lo, sp_hi, _btn in self._hkl_widgets():
                                d_min, d_max = getattr(self, da), getattr(self, db)
                                sl.blockSignals(True)
                                sl.setFullRange()
                                sl.blockSignals(False)
                                for sp, v in ((sp_lo, d_min), (sp_hi, d_max)):
                                    sp.blockSignals(True)
                                    sp.setValue(v)
                                    sp.blockSignals(False)
                                lbl.setText(f'{d_min:.3f} → {d_max:.3f}')
                    except Exception:
                        pass

                    # Switch to this tab automatically
                    if hasattr(mw, 'tabWidget_analysis'):
                        idx = mw.tabWidget_analysis.indexOf(self)
                        mw.tabWidget_analysis.setCurrentIndex(idx)
                    
                    self.main_window.update_status("3D Rendering Complete")
                except Exception as e:
                    try:
                        self.main_window.update_status(f"Render Error: {e}")
                    except Exception:
                        pass
                finally:
                    self._close_load_progress()

            # 4. Threaded Execution — the worker loads the file off the UI thread.
            def _on_load_failed(msg):
                self._close_load_progress()
                QMessageBox.warning(
                    self, '3D Load Failed',
                    f'Could not load 3D data from:\n{file_path}\n\n'
                    f'The file may not contain HKL metadata or precomputed Q-space data.\n\n'
                    f'Error: {msg}'
                )

            self._render_thread = QThread(self)
            self._render3d_worker = Render3D(
                file_path=file_path,
                downsample=downsample,
            )

            self._render3d_worker.moveToThread(self._render_thread)

            # Connect signals
            self._render_thread.started.connect(self._render3d_worker.run)
            self._render3d_worker.render_ready.connect(_on_ready) # Use the local plotter
            self._render3d_worker.failed.connect(_on_load_failed)

            # Cleanup
            self._render3d_worker.finished.connect(self._render_thread.quit)
            self._render3d_worker.finished.connect(self._render3d_worker.deleteLater)
            self._render3d_worker.finished.connect(self._close_load_progress)
            self._render_thread.finished.connect(self._render_thread.deleteLater)

            self._render_thread.start()

        except Exception as e:
            self._close_load_progress()
            QMessageBox.critical(self, "3D Viewer Error", f"Error: {str(e)}")
        finally:
            pass

    def _close_load_progress(self):
        """Close and discard the 3D loading progress dialog if present."""
        try:
            dlg = getattr(self, "_load_progress", None)
            if dlg is not None:
                dlg.close()
                self._load_progress = None
        except Exception:
            pass

    def on_plane_update(self, normal, origin):
        """Extracts points near the plane to simulate a 3D slice."""
        if self.cloud_mesh_3d is None:
            return

        # Plane math: (Point - Origin) ⋅ Normal
        vec = self.cloud_mesh_3d.points - origin
        dist = np.dot(vec, normal)
        
        # Thickness of the slice in HKL units (set via the Slice Controls dock).
        thickness = getattr(self, '_slice_thickness', 0.002)
        mask = np.abs(dist) < thickness
        
        slab = self.cloud_mesh_3d.extract_points(mask)

        # Cache latest slice state for extraction/saving to 2D.
        self._last_slab = slab
        self._last_normal = np.asarray(normal, dtype=float)
        self._last_origin = np.asarray(origin, dtype=float)

        # Clean up any existing slab actor before adding a new one to keep references current
        try:
            actors = getattr(self.plotter, 'actors', {}) or {}
            if 'slab_points' in actors:
                try:
                    self.plotter.remove_actor('slab_points', reset_camera=False)
                except Exception:
                    try:
                        self.plotter.remove_actor(actors.get('slab_points'), reset_camera=False)
                    except Exception:
                        pass
        except Exception:
            pass
        self.slab_actor = None

        if slab.n_points > 0:
            # Add the slab without passing cmap; set mapper.lookup_table explicitly afterwards
            self.slab_actor = self.plotter.add_mesh(
                slab,
                name="slab_points",
                render_points_as_spheres=True,
                point_size=8,
                scalars='intensity',
                show_scalar_bar=False,
                reset_camera=False,
            )
            # Apply current LUT and scalar range to the new slab actor
            try:
                if (self.lut2 or self.lut) is not None:
                    try:
                        self.slab_actor.mapper.lookup_table = (self.lut2 or self.lut)
                    except Exception:
                        self.slab_actor.GetMapper().SetLookupTable(self.lut2 or self.lut)
            except Exception:
                pass
            # Match current intensity/clim
            try:
                clim = [self.sb_min_intensity_3d.value(), self.sb_max_intensity_3d.value()]
                self.slab_actor.mapper.scalar_range = clim
            except Exception:
                pass
            # Ensure the new slab respects the current checkbox state
            try:
                self.slab_actor.SetVisibility(self.cb_show_slice.isChecked())
            except Exception:
                pass
        
        self.plotter.render()
        # Respect slice toggle state after update
        try:
            self.toggle_3d_slice(self.cb_show_slice.isChecked())
        except Exception:
            pass
        
        # Update the 3D Info dock with HKL slice information
        try:
            info_dock = getattr(self.main_window, 'info_3d_dock', None)
            if info_dock is not None:
                shape = None
                try:
                    shape = tuple(getattr(self, 'curr_shape', None) or ())
                    if not (isinstance(shape, tuple) and len(shape) == 2):
                        shape = (0, 0)
                except Exception:
                    shape = (0, 0)
                info_dock.update_from_slice(
                    slab,
                    np.asarray(normal, dtype=float),
                    np.asarray(origin, dtype=float),
                    target_shape=shape
                )
        except Exception:
            pass

        # Live-update the Slice 2D tab with the masked/rasterized slice.
        try:
            tab_2d = getattr(self.main_window, 'tab_slice_2d', None)
            if tab_2d is not None:
                clim = (float(self.sb_min_intensity_3d.value()),
                        float(self.sb_max_intensity_3d.value()))
                tab_2d.schedule_update(slab, normal, origin, clim)
        except Exception:
            pass

    def update_intensity(self):
        """Updates the min/max intensity levels and scalar bar range"""
        if not self.plotter:
            return

        # Read requested values from UI
        try:
            requested_min = float(self.sb_min_intensity_3d.value())
            requested_max = float(self.sb_max_intensity_3d.value())
        except Exception:
            # Fallback to current mapper range if spinboxes unavailable
            requested_min, requested_max = 0.0, 1.0

        # Clamp to true data range if available
        data_min = getattr(self, '_data_intensity_min', None)
        data_max = getattr(self, '_data_intensity_max', None)
        vmin = requested_min
        vmax = requested_max
        if data_min is not None and data_max is not None:
            vmin = max(requested_min, data_min)
            vmax = min(requested_max, data_max)

        # Enforce ordering and non-zero span
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmax = vmin + 1e-6

        # Reflect applied values back to the UI
        try:
            self.sb_min_intensity_3d.setValue(vmin)
            self.sb_max_intensity_3d.setValue(vmax)
        except Exception:
            pass

        # Define the new scalar range
        new_range = [vmin, vmax]

        # Update main cloud/points actor scalar range
        try:
            actors = getattr(self.plotter, 'actors', {}) or {}
            if "points" in actors:
                actors["points"].mapper.scalar_range = (new_range[0], new_range[1])
            if "cloud_volume" in actors:
                actors["cloud_volume"].mapper.scalar_range = (new_range[0], new_range[1])
        except Exception:
            pass

        if "slab_points" in self.plotter.actors:
            self.plotter.actors["slab_points"].mapper.scalar_range = (new_range[0], new_range[1])

        # Keep LUTs' internal ranges consistent as well
        try:
            if self.lut is not None:
                self.lut.scalar_range = (new_range[0], new_range[1])
            if self.lut2 is not None:
                self.lut2.scalar_range = (new_range[0], new_range[1])
        except Exception:
            pass
            
        # Update the volume actor by re-adding with new clim range
        if hasattr(self.plotter, 'scalar_bars'):
            for bar in self.plotter.scalar_bars.values():
                try:
                    bar.GetLookupTable().SetTableRange(new_range[0], new_range[1])
                except Exception:
                    pass

        # Force update of all scalar bars with the new range
        if hasattr(self.plotter, 'scalar_bars'):
            for name, scalar_bar in self.plotter.scalar_bars.items():
                if scalar_bar:
                    try:
                        scalar_bar.GetLookupTable().SetTableRange(new_range[0], new_range[1])
                        scalar_bar.Modified()
                    except Exception:
                        pass

        # Update slice actor scalar range if it exists
        if "slice" in self.plotter.actors:
            slice_actor = self.plotter.actors["slice"]
            if hasattr(slice_actor, 'mapper'):
                try:
                    slice_actor.mapper.scalar_range = (new_range[0], new_range[1])
                except Exception:
                    pass

        # Force a re-render to apply the changes
        self.plotter.render()
        # Respect checkbox states after intensity update
        try:
            self.toggle_3d_points(self.cb_show_points.isChecked())
            self.toggle_3d_slice(self.cb_show_slice.isChecked())
        except Exception:
            pass

        # Update Info labels and availability after intensity changes (best-effort)
        try:
            self.update_info_slice_labels()
            self._refresh_availability()
        except Exception:
            pass

        # Keep the extracted 2D slice in sync with the current intensity range.
        self.refresh_slice_2d()

    def refresh_slice_2d(self):
        """Re-push the current slab to the Slice 2D tab using the current controls.

        Keeps the extracted 2D slice consistent with the 3D view's intensity range
        (and colormap) even when the slice plane itself has not moved.
        """
        try:
            slab = getattr(self, '_last_slab', None)
            if slab is None:
                return
            tab_2d = getattr(self.main_window, 'tab_slice_2d', None)
            if tab_2d is None:
                return
            clim = (float(self.sb_min_intensity_3d.value()),
                    float(self.sb_max_intensity_3d.value()))
            tab_2d.schedule_update(slab, self._last_normal, self._last_origin, clim)
        except Exception:
            pass

    def save_3d_config(self):
        """Save the current HKL range + intensity min/max as a named config.

        Writes into the currently-loaded HDF5 file under /entry/view_configs so
        several named configs can live in the same file and be reloaded later.
        """
        from PyQt5.QtWidgets import QInputDialog, QMessageBox

        mw = self.main_window
        file_path = getattr(mw, 'current_file_path', None) or getattr(mw, 'selected_dataset_path', None)
        if not file_path:
            QMessageBox.warning(self, '3D Config',
                                'No loaded file to save into. Load a 3D dataset first.')
            return
        name, ok = QInputDialog.getText(self, 'Save 3D Config', 'Configuration name:')
        if not ok or not name.strip():
            return
        name = name.strip()

        from dashpva.utils.hdf5_loader import HDF5Loader
        loader = HDF5Loader()
        if name in loader.list_view_configs(file_path):
            if QMessageBox.question(
                self, '3D Config', f"Configuration '{name}' exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return

        config = {
            'hkl_range': [
                float(self._h_min_spin.value()), float(self._h_max_spin.value()),
                float(self._k_min_spin.value()), float(self._k_max_spin.value()),
                float(self._l_min_spin.value()), float(self._l_max_spin.value()),
            ],
            'intensity_min': float(self.sb_min_intensity_3d.value()),
            'intensity_max': float(self.sb_max_intensity_3d.value()),
        }
        if loader.save_view_config(file_path, name, config):
            try:
                mw.update_status(f"Saved 3D config '{name}' to {file_path}")
            except Exception:
                pass
            QMessageBox.information(self, '3D Config', f"Saved configuration '{name}'.")
        else:
            QMessageBox.critical(self, '3D Config',
                                 f'Failed to save configuration:\n{loader.get_last_error()}')

    def load_3d_config(self):
        """Load a saved HKL range + intensity config from the loaded file and apply it."""
        from PyQt5.QtWidgets import QInputDialog, QMessageBox

        mw = self.main_window
        file_path = getattr(mw, 'current_file_path', None) or getattr(mw, 'selected_dataset_path', None)
        if not file_path:
            QMessageBox.warning(self, '3D Config', 'No loaded file to read configurations from.')
            return

        from dashpva.utils.hdf5_loader import HDF5Loader
        loader = HDF5Loader()
        names = loader.list_view_configs(file_path)
        if not names:
            QMessageBox.information(self, '3D Config', 'No saved configurations in this file.')
            return
        if len(names) == 1:
            name = names[0]
        else:
            name, ok = QInputDialog.getItem(
                self, 'Load 3D Config', 'Configuration:', names, 0, False)
            if not ok:
                return

        config = loader.load_view_config(file_path, name)
        if not config:
            QMessageBox.warning(self, '3D Config',
                                f"Could not load '{name}':\n{loader.get_last_error()}")
            return
        self._apply_view_config(config)
        try:
            mw.update_status(f"Loaded 3D config '{name}'")
        except Exception:
            pass

    def _apply_view_config(self, config: dict) -> None:
        """Apply a loaded HKL range + intensity config to the 3D controls."""
        hr = config.get('hkl_range')
        if hr and len(hr) == 6:
            try:
                pairs = [(hr[0], hr[1]), (hr[2], hr[3]), (hr[4], hr[5])]
                for (sl, da, db, lbl, sp_lo, sp_hi, _btn), (lo, hi) in zip(self._hkl_widgets(), pairs):
                    d_min, d_max = getattr(self, da), getattr(self, db)
                    for sp, v in ((sp_lo, lo), (sp_hi, hi)):
                        sp.blockSignals(True)
                        sp.setValue(v)
                        sp.blockSignals(False)
                    try:
                        sl.blockSignals(True)
                        sl.setLow(self._slider_int(lo, d_min, d_max), emit=False)
                        sl.setHigh(self._slider_int(hi, d_min, d_max), emit=False)
                        sl.blockSignals(False)
                        sl.update()
                    except Exception:
                        pass
                    lbl.setText(f'{lo:.3f} → {hi:.3f}')
                self.update_hkl_range()
            except Exception:
                pass
        try:
            imin = config.get('intensity_min')
            imax = config.get('intensity_max')
            if imin is not None:
                self.sb_min_intensity_3d.blockSignals(True)
                self.sb_min_intensity_3d.setValue(int(round(imin)))
                self.sb_min_intensity_3d.blockSignals(False)
            if imax is not None:
                self.sb_max_intensity_3d.blockSignals(True)
                self.sb_max_intensity_3d.setValue(int(round(imax)))
                self.sb_max_intensity_3d.blockSignals(False)
            self.update_intensity()
        except Exception:
            pass

    def update_hkl_range(self):
        """Filter the displayed cloud to the H/K/L min/max range and re-render."""
        if self._display_points is None or self.plotter is None:
            return
        try:
            h_min = self._h_min_spin.value()
            h_max = self._h_max_spin.value()
            k_min = self._k_min_spin.value()
            k_max = self._k_max_spin.value()
            l_min = self._l_min_spin.value()
            l_max = self._l_max_spin.value()

            pts = self._display_points
            mask = (
                (pts[:, 0] >= h_min) & (pts[:, 0] <= h_max) &
                (pts[:, 1] >= k_min) & (pts[:, 1] <= k_max) &
                (pts[:, 2] >= l_min) & (pts[:, 2] <= l_max)
            )
            filtered_pts = pts[mask]
            filtered_int = self._display_intensities[mask]

            if filtered_pts.shape[0] == 0:
                return

            try:
                actors = getattr(self.plotter, 'actors', {}) or {}
                if 'points' in actors:
                    self.plotter.remove_actor('points', reset_camera=False)
            except Exception:
                pass

            mesh = pv.PolyData(filtered_pts)
            mesh['intensity'] = filtered_int
            self.cloud_mesh_3d = mesh

            self.plotter.add_mesh(
                mesh,
                scalars='intensity',
                cmap=self.lut,
                point_size=5.0,
                name='points',
                show_scalar_bar=True,
                nan_opacity=0.0,
                show_edges=False,
            )
            self.plotter.show_bounds(
                mesh=mesh,
                xtitle='H Axis', ytitle='K Axis', ztitle='L Axis',
                bounds=mesh.bounds,
            )
            self.plotter.reset_camera()

            try:
                current_normal, current_origin = self.get_plane_state()
                bounds = mesh.bounds
                clamped_origin = np.array([
                    np.clip(current_origin[0], bounds[0], bounds[1]),
                    np.clip(current_origin[1], bounds[2], bounds[3]),
                    np.clip(current_origin[2], bounds[4], bounds[5]),
                ], dtype=float)
                self._remove_plane_widget()
                self.plane_widget = self.plotter.add_plane_widget(
                    callback=self.on_plane_update,
                    normal=current_normal,
                    origin=clamped_origin,
                    bounds=bounds,
                    factor=1.0,
                    implicit=True,
                    assign_to_axis=None,
                    tubing=False,
                    origin_translation=True,
                    outline_opacity=0,
                )
            except Exception:
                pass

            try:
                self.update_intensity()
            except Exception:
                pass
            self.plotter.render()
        except Exception as e:
            try:
                self.main_window.update_status(f"HKL range filter error: {e}")
            except Exception:
                pass

    def _on_downsample_changed(self) -> None:
        """Recompute the display point cloud when the downsampling toggle changes."""
        if self._raw_points is None:
            return
        try:
            if self.rb_downsample_on.isChecked():
                _MAX_VIEWER_PTS = 2_000_000
                _stride = max(1, len(self._raw_intensities) // _MAX_VIEWER_PTS)
                if _stride > 1:
                    self._display_points = self._raw_points[::_stride]
                    self._display_intensities = self._raw_intensities[::_stride]
                else:
                    self._display_points = self._raw_points
                    self._display_intensities = self._raw_intensities
            else:
                self._display_points = self._raw_points
                self._display_intensities = self._raw_intensities
            self.update_hkl_range()
        except Exception as e:
            try:
                self.main_window.update_status(f"Downsampling toggle error: {e}")
            except Exception:
                pass

    # === Folder playback ===
    def _init_playback_state(self):
        """Initialise folder-playback state + the per-frame 1D signal plot."""
        self._frame_files = []
        self._current_frame = 0
        self._frame_cache = OrderedDict()
        self._signal_series = {}
        self._playback_seeded = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        self._signal_plot = None
        self._signal_marker = None
        self._build_playback_plot()

    def _build_playback_plot(self):
        try:
            self._signal_plot = pg.PlotWidget()
            self._signal_marker = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen(color=(0, 200, 255), width=1))
            self._signal_plot.addItem(self._signal_marker)
            self._signal_marker.hide()
            self.layout_playback_signal.addWidget(self._signal_plot)
        except Exception:
            self._signal_plot = None

    def browse_folder(self):
        """Pick a folder of per-frame .h5 files and start folder playback."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox

        s = QSettings("DashPVA", "Workbench")
        start_dir = s.value("last_3d_folder", "", type=str)
        folder = QFileDialog.getExistingDirectory(self, "Select folder of 3D frame files", start_dir)
        if not folder:
            return
        s.setValue("last_3d_folder", folder)
        files = natsorted({str(p) for p in Path(folder).glob('*.h5')}
                          | {str(p) for p in Path(folder).glob('*.hdf5')})
        if not files:
            QMessageBox.warning(self, "Folder Playback", "No .h5/.hdf5 files found in that folder.")
            return
        self._frame_files = list(files)
        self._frame_cache.clear()
        self._current_frame = 0
        self._playback_seeded = False
        try:
            self.gb_playback.setEnabled(True)
            self.slider_frame.blockSignals(True)
            self.slider_frame.setRange(0, len(files) - 1)
            self.slider_frame.setValue(0)
            self.slider_frame.blockSignals(False)
        except Exception:
            pass
        self._populate_signal_dropdown(files[0])
        self._load_frame_signals(files)
        self._render_frame(0, reset_camera=True)
        self._on_signal_changed()

    def _populate_signal_dropdown(self, sample_file):
        from dashpva.utils.hdf5_loader import HDF5Loader
        try:
            names = HDF5Loader().list_frame_signals(sample_file)
            self.cb_playback_signal.blockSignals(True)
            self.cb_playback_signal.clear()
            for n in names:
                self.cb_playback_signal.addItem(n, n)
            self.cb_playback_signal.blockSignals(False)
        except Exception:
            pass

    def _load_frame_signals(self, files):
        """One pass over the folder reading each frame's scalar signals."""
        from PyQt5.QtWidgets import QApplication, QProgressDialog

        from dashpva.utils.hdf5_loader import HDF5Loader
        names = [self.cb_playback_signal.itemData(i) for i in range(self.cb_playback_signal.count())]
        self._signal_series = {n: [] for n in names}
        if not names:
            return
        loader = HDF5Loader()
        progress = QProgressDialog("Reading frame signals…", "Cancel", 0, len(files), self)
        progress.setWindowTitle("Folder Playback")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        for i, path in enumerate(files):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            for n in names:
                self._signal_series[n].append(loader.read_frame_scalar(path, n))
        progress.setValue(len(files))
        for n in list(self._signal_series.keys()):
            self._signal_series[n] = np.array(
                [np.nan if v is None else v for v in self._signal_series[n]], dtype=float)

    def _load_frame_cloud(self, path):
        if path in self._frame_cache:
            self._frame_cache.move_to_end(path)
            return self._frame_cache[path]
        from dashpva.utils.rsm_converter import RSMConverter
        points, intensities, _num, _shape = RSMConverter().load_h5_to_3d(path)
        pts = np.asarray(points, dtype=float)
        ints = np.asarray(intensities, dtype=float).reshape(-1)
        self._frame_cache[path] = (pts, ints)
        cap = max(1, int(getattr(app_settings, 'SLICE_FRAME_FILE_CACHE', 2)))
        while len(self._frame_cache) > cap:
            self._frame_cache.popitem(last=False)
        return pts, ints

    def _render_frame(self, index, reset_camera=False):
        if not self._frame_files:
            return
        index = max(0, min(int(index), len(self._frame_files) - 1))
        self._current_frame = index
        path = self._frame_files[index]
        try:
            pts, ints = self._load_frame_cloud(path)
        except Exception as e:
            try:
                self.main_window.update_status(f"Frame load error: {e}")
            except Exception:
                pass
            return
        if not self._playback_seeded:
            self._seed_playback_from(pts, ints)
            self._playback_seeded = True
            reset_camera = True
        self._render_cloud(pts, ints, reset_camera=reset_camera)
        try:
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(index)
            self.slider_frame.blockSignals(False)
        except Exception:
            pass
        try:
            self.lbl_frame_counter.setText(f"{index + 1} / {len(self._frame_files)}")
            self.lbl_frame_counter.setToolTip(os.path.basename(path))
        except Exception:
            pass
        try:
            if self._signal_marker is not None:
                self._signal_marker.setValue(index)
                self._signal_marker.show()
        except Exception:
            pass

    def _seed_playback_from(self, pts, ints):
        """First-frame setup: intensity bounds, LUT, and HKL range from the data."""
        try:
            self._data_intensity_min = float(np.min(ints))
            self._data_intensity_max = float(np.max(ints))
            lo, hi = int(self._data_intensity_min), int(self._data_intensity_max)
            if hi <= lo:
                hi = lo + 1
            for sb, v in ((self.sb_min_intensity_3d, lo), (self.sb_max_intensity_3d, hi)):
                sb.blockSignals(True)
                sb.setRange(lo, hi)
                sb.setValue(v)
                sb.blockSignals(False)
        except Exception:
            pass
        try:
            self.on_3d_colormap_changed()
        except Exception:
            pass
        try:
            self._h_min_data = float(np.min(pts[:, 0]))
            self._h_max_data = float(np.max(pts[:, 0]))
            self._k_min_data = float(np.min(pts[:, 1]))
            self._k_max_data = float(np.max(pts[:, 1]))
            self._l_min_data = float(np.min(pts[:, 2]))
            self._l_max_data = float(np.max(pts[:, 2]))
            for sl, da, db, lbl, sp_lo, sp_hi, _btn in self._hkl_widgets():
                d_min, d_max = getattr(self, da), getattr(self, db)
                sl.blockSignals(True)
                sl.setFullRange()
                sl.blockSignals(False)
                for sp, v in ((sp_lo, d_min), (sp_hi, d_max)):
                    sp.blockSignals(True)
                    sp.setValue(v)
                    sp.blockSignals(False)
                lbl.setText(f'{d_min:.3f} → {d_max:.3f}')
        except Exception:
            pass

    def _render_cloud(self, points, intensities, reset_camera=False):
        """Swap the point cloud shown in the plotter (lightweight per-frame render)."""
        if self.plotter is None:
            return
        pts = np.asarray(points, dtype=float)
        ints = np.asarray(intensities, dtype=float).reshape(-1)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            return
        self._raw_points = pts
        self._raw_intensities = ints
        self._display_points = pts
        self._display_intensities = ints
        try:
            actors = getattr(self.plotter, 'actors', {}) or {}
            if 'points' in actors:
                self.plotter.remove_actor('points', reset_camera=False)
        except Exception:
            pass
        mesh = pv.PolyData(pts)
        mesh['intensity'] = ints
        self.cloud_mesh_3d = mesh
        try:
            self.plotter.add_mesh(
                mesh, scalars='intensity',
                cmap=self.lut if getattr(self, 'lut', None) is not None else 'viridis',
                point_size=5.0, name='points', show_scalar_bar=True,
                nan_opacity=0.0, show_edges=False)
        except Exception:
            return
        try:
            self.plotter.show_bounds(mesh=mesh, xtitle='H Axis', ytitle='K Axis',
                                     ztitle='L Axis', bounds=mesh.bounds)
        except Exception:
            pass
        if reset_camera:
            try:
                self.plotter.reset_camera()
            except Exception:
                pass
        try:
            self.update_intensity()
        except Exception:
            pass
        try:
            self.plotter.render()
        except Exception:
            pass

    def prev_frame(self):
        self._render_frame(self._current_frame - 1)

    def next_frame(self):
        self._render_frame(self._current_frame + 1)

    def _advance_frame(self):
        if not self._frame_files:
            return
        nxt = self._current_frame + 1
        if nxt >= len(self._frame_files):
            nxt = 0
        self._render_frame(nxt)

    def toggle_play(self, checked):
        if checked and self._frame_files:
            fps = max(1, int(self.sb_fps.value()))
            self._play_timer.start(int(1000 / fps))
            try:
                self.btn_play_pause.setText('⏸')
            except Exception:
                pass
        else:
            self._play_timer.stop()
            try:
                self.btn_play_pause.setText('▶')
            except Exception:
                pass

    def set_fps(self, value):
        if self._play_timer.isActive():
            self._play_timer.start(int(1000 / max(1, int(value))))

    def _on_slider_changed(self, value):
        self._render_frame(int(value))

    def _on_signal_changed(self):
        if self._signal_plot is None:
            return
        try:
            name = self.cb_playback_signal.currentData()
            series = self._signal_series.get(name) if name else None
            self._signal_plot.clear()
            self._signal_plot.addItem(self._signal_marker)
            if series is not None and len(series) > 0:
                self._signal_plot.plot(np.arange(len(series)), series,
                                       pen='y', symbol='o', symbolSize=4)
                self._signal_plot.setLabel('bottom', 'Frame')
                self._signal_plot.setLabel('left', str(name))
            self._signal_marker.setValue(self._current_frame)
            self._signal_marker.show()
        except Exception:
            pass

    def reset_slice(self):
        """Reset slice to HK (xy) preset at the data center."""
        try:
            # Determine a reasonable center
            origin = None
            try:
                if self.cloud_mesh_3d is not None and hasattr(self.cloud_mesh_3d, 'center'):
                    origin = np.array(self.cloud_mesh_3d.center, dtype=float)
                elif self.mesh is not None and hasattr(self.mesh, 'center'):
                    origin = np.array(self.mesh.center, dtype=float)
            except Exception:
                origin = None
            if origin is None:
                origin = np.array([0.0, 0.0, 0.0], dtype=float)
            # Normal along L for HK plane
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
            self.set_plane_state(normal, origin)
        except Exception as e:
            try:
                self.main_window.update_status(f"Error resetting 3D slice: {e}")
            except Exception:
                pass

    def _remove_plane_widget(self):
        """Safely remove existing plane widget (if any)."""
        try:
            # Use the same attribute name that is set by the Render3D worker
            if self.plane_widget is not None:
                try:
                    self.plane_widget.EnabledOff()
                except Exception:
                    pass

                try:
                    self.plotter.clear_plane_widgets()
                except Exception:
                    pass

                self.plane_widget = None
        except Exception:
            pass

    def toggle_pointer(self, checked: bool):
        """Enable/Disable the interactive plane widget and show/hide slab points."""
        try:
            # Plane widget visibility
            if self.plane_widget is not None:
                try:
                    if checked:
                        self.plane_widget.On()
                    else:
                        self.plane_widget.Off()
                except Exception:
                    pass
            else:
                # Fallback: use plotter.plane_widgets list if available
                widgets = getattr(self.plotter, "plane_widgets", [])
                for pw in widgets or []:
                    try:
                        if checked:
                            pw.EnabledOn()
                        else:
                            pw.EnabledOff()
                    except Exception:
                        pass
            # Slab points actor visibility
            if "slab_points" in self.plotter.actors:
                try:
                    self.plotter.actors["slab_points"].SetVisibility(bool(checked))
                except Exception:
                    try:
                        self.plotter.renderer._actors["slab_points"].SetVisibility(bool(checked))
                    except Exception:
                        pass
            self.plotter.render()
        except Exception:
            pass

    # ===== Info/Availability (align with hkl_3d patterns) =====

    def _refresh_availability(self):
        """Enable/disable controls depending on plotter/data availability."""
        try:
            has_data = bool(self.cloud_mesh_3d is not None or getattr(self, 'points_actor', None) is not None)
            for w in [getattr(self, "cb_show_points", None),
                      getattr(self, "cb_show_slice", None),
                      getattr(self, "sb_min_intensity_3d", None),
                      getattr(self, "sb_max_intensity_3d", None)]:
                try:
                    if w is not None:
                        w.setEnabled(has_data)
                except Exception:
                    pass
        except Exception:
            pass

    # ===== Slice Plane helpers =====
    def get_plane_state(self):
        """Return (normal, origin) for current plane; defaults to Z-axis and mesh center."""
        try:
            if self.plane_widget is not None:
                try:
                    normal = np.array(self.plane_widget.GetNormal(), dtype=float)
                    origin = np.array(self.plane_widget.GetOrigin(), dtype=float)
                    return normal, origin
                except Exception:
                    pass
            # Fallback to first plane widget if present
            widgets = getattr(self.plotter, 'plane_widgets', [])
            if widgets:
                pw = widgets[0]
                try:
                    normal = np.array(pw.GetNormal(), dtype=float)
                    origin = np.array(pw.GetOrigin(), dtype=float)
                    return normal, origin
                except Exception:
                    pass
        except Exception:
            pass
        # Defaults
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        try:
            if self.cloud_mesh_3d is not None and hasattr(self.cloud_mesh_3d, 'center'):
                origin = np.array(self.cloud_mesh_3d.center, dtype=float)
            elif self.mesh is not None and hasattr(self.mesh, 'center'):
                origin = np.array(self.mesh.center, dtype=float)
            else:
                origin = np.array([0.0, 0.0, 0.0], dtype=float)
        except Exception:
            origin = np.array([0.0, 0.0, 0.0], dtype=float)
        return normal, origin

    def set_plane_state(self, normal, origin):
        """Programmatically set plane state and trigger slice update."""
        try:
            n = self.normalize_vector(np.array(normal, dtype=float))
            o = np.array(origin, dtype=float)
            # Update widget if available
            if self.plane_widget is not None:
                try:
                    self.plane_widget.SetNormal(n)
                    self.plane_widget.SetOrigin(o)
                except Exception:
                    pass
            else:
                widgets = getattr(self.plotter, 'plane_widgets', [])
                if widgets:
                    try:
                        widgets[0].SetNormal(n)
                        widgets[0].SetOrigin(o)
                    except Exception:
                        pass
            # Refresh slice
            try:
                self.on_plane_update(n, o)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def normalize_vector(v):
        try:
            v = np.array(v, dtype=float)
            norm = float(np.linalg.norm(v))
            if norm <= 0.0:
                return np.array([0.0, 0.0, 1.0], dtype=float)
            return v / norm
        except Exception:
            return np.array([0.0, 0.0, 1.0], dtype=float)

    def set_custom_normal(self, n):
        try:
            self._custom_normal = np.array(n, dtype=float)
        except Exception:
            self._custom_normal = np.array([0.0, 0.0, 1.0], dtype=float)

    def set_plane_preset(self, preset_text: str):
        """Set plane normal to preset HK/KL/HL or custom vector."""
        try:
            preset = (preset_text or '').lower()
        except Exception:
            preset = ''
        if ('xy' in preset) or ('hk' in preset):
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        elif ('yz' in preset) or ('kl' in preset):
            n = np.array([1.0, 0.0, 0.0], dtype=float)
        elif ('xz' in preset) or ('hl' in preset):
            n = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            # Custom
            n = self.normalize_vector(getattr(self, '_custom_normal', np.array([0.0, 0.0, 1.0], dtype=float)))
        _, origin = self.get_plane_state()
        self.set_plane_state(n, origin)

    # ===== Translation =====
    def nudge_along_normal(self, sign: int):
        try:
            normal, origin = self.get_plane_state()
            step = float(getattr(self, '_slice_translate_step', 0.01))
            origin_new = origin + float(sign) * step * normal
            self.set_plane_state(normal, origin_new)
        except Exception:
            pass

    def nudge_along_axis(self, axis: str, sign: int):
        try:
            axis = (axis or 'H').upper()
            if axis == 'H':
                d = np.array([1.0, 0.0, 0.0], dtype=float)
            elif axis == 'K':
                d = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                d = np.array([0.0, 0.0, 1.0], dtype=float)
            normal, origin = self.get_plane_state()
            step = float(getattr(self, '_slice_translate_step', 0.01))
            origin_new = origin + float(sign) * step * d
            self.set_plane_state(normal, origin_new)
        except Exception:
            pass

    # ===== Rotation =====
    def rotate_about_axis(self, axis: str, deg: float):
        try:
            axis = (axis or 'H').upper()
            if axis == 'H':
                u = np.array([1.0, 0.0, 0.0], dtype=float)
            elif axis == 'K':
                u = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                u = np.array([0.0, 0.0, 1.0], dtype=float)
            normal, origin = self.get_plane_state()
            theta = float(np.deg2rad(deg))
            ux, uy, uz = u
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([
                [c+ux*ux*(1-c),    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c),    uy*uz*(1-c)-ux*s],
                [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]
            ], dtype=float)
            new_normal = R @ normal
            new_normal = self.normalize_vector(new_normal)
            self.set_plane_state(new_normal, origin)
        except Exception:
            pass

    # ===== Camera =====
    def zoom_in(self):
        try:
            step = float(getattr(self, '_zoom_step', 1.5))
            if step <= 1.0:
                step = 1.5
            self.plotter.camera.zoom(step)
            self.plotter.render()
        except Exception:
            pass

    def zoom_out(self):
        try:
            step = float(getattr(self, '_zoom_step', 1.5))
            if step <= 1.0:
                step = 1.5
            self.plotter.camera.zoom(1.0 / step)
            self.plotter.render()
        except Exception:
            pass

    def reset_camera(self):
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def set_camera_position(self, preset: str):
        try:
            txt = (preset or '').strip().lower()
            p = self.plotter
            cam = getattr(p, 'camera', None)
            # center focus
            try:
                if self.cloud_mesh_3d is not None and hasattr(self.cloud_mesh_3d, 'center'):
                    p.set_focus(self.cloud_mesh_3d.center)
            except Exception:
                pass
            if txt in ('hk', 'xy'):
                p.view_xy()
            elif txt in ('kl', 'yz'):
                p.view_yz()
            elif txt in ('hl', 'xz'):
                p.view_xz()
            elif 'iso' in txt:
                try:
                    p.view_isometric()
                except Exception:
                    try:
                        p.view_vector((1.0, 1.0, 1.0))
                        if cam is not None:
                            cam.view_up = (0.0, 0.0, 1.0)
                    except Exception:
                        pass
            else:
                # Axis-aligned
                if 'h+' in txt:
                    p.view_vector((1.0, 0.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif 'h-' in txt:
                    p.view_vector((-1.0, 0.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif 'k+' in txt:
                    p.view_vector((0.0, 1.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif 'k-' in txt:
                    p.view_vector((0.0, -1.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif 'l+' in txt:
                    p.view_vector((0.0, 0.0, 1.0))
                    if cam is not None:
                        cam.view_up = (0.0, 1.0, 0.0)
                elif 'l-' in txt:
                    p.view_vector((0.0, 0.0, -1.0))
                    if cam is not None:
                        cam.view_up = (0.0, 1.0, 0.0)
            try:
                if cam is not None and hasattr(cam, 'orthogonalize_view_up'):
                    cam.orthogonalize_view_up()
            except Exception:
                pass
            try:
                p.render()
            except Exception:
                pass
        except Exception:
            pass

    def view_slice_normal(self):
        try:
            normal, origin = self.get_plane_state()
            normal = self.normalize_vector(normal)
            origin = np.array(origin, dtype=float)
            cam = getattr(self.plotter, 'camera', None)
            if cam is None:
                return
            # distance heuristic
            try:
                rng = None
                if self.cloud_mesh_3d is not None and hasattr(self.cloud_mesh_3d, 'points'):
                    rng = self.cloud_mesh_3d.points.max(axis=0) - self.cloud_mesh_3d.points.min(axis=0)
                d = float(np.linalg.norm(rng)) * 0.5 if rng is not None else 1.0
            except Exception:
                d = 1.0
            try:
                cam.focal_point = origin.tolist()
            except Exception:
                pass
            try:
                cam.position = (origin + normal * d).tolist()
            except Exception:
                pass
            # adjust view up if parallel
            try:
                up = np.array(getattr(cam, 'view_up', [0.0, 1.0, 0.0]), dtype=float)
                upn = self.normalize_vector(up)
                if abs(float(np.dot(upn, normal))) > 0.99:
                    new_up = np.array([0.0, 1.0, 0.0], dtype=float) if abs(normal[1]) < 0.99 else np.array([1.0, 0.0, 0.0], dtype=float)
                    cam.view_up = new_up.tolist()
            except Exception:
                pass
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception:
            pass
