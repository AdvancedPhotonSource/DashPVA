"""
2D Workspace tab for the Workbench.

Encapsulates the 2D image viewer (PyQtGraph ImageView), its controls, frame
playback, intensity/levels handling, hover readout, ROI hooks and per-frame
annotation display. Loaded from ``workspace_2d.ui`` and registered as its own
tab in ``tabWidget_analysis`` — mirroring :class:`Workspace3D`.

Usage:
    self.tab_2d = Workspace2D(parent=self, main_window=self)
    self.tab_2d.display_2d_data(numpy_array)   # HxW or FxHxW
"""

import os

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyle,
)

from dashpva.gui import ui_path

from .base_tab import BaseTab


class Workspace2D(BaseTab):
    """
    2D viewer tab encapsulating 2D image display, controls, playback and ROI hooks.

    Delegates shared workbench state (ROI collections, docks, status bar) to
    ``self.main_window``; owns its own image widgets and 2D display state.
    """

    def __init__(self, parent=None, main_window=None, title="2D Viewer"):
        try:
            super().__init__(ui_file=ui_path("workbench", "workspace", "workspace_2d.ui"),
                             parent=parent, main_window=main_window, title=title)
            self.title = title
            self.main_window = main_window
            self.current_2d_data = None
            self.axis_2d_x = "Columns"
            self.axis_2d_y = "Row"
            self._roi_stats_label_added = False
            self._vmin_vmax_hints_inserted = False
            self.setup_2d_plot_viewer()
            self.connect_all()
        except Exception as e:
            try:
                self.main_window.update_status(f"Error initializing 2D workspace: {e}")
            except Exception:
                pass

    def connect_all(self):
        """Wire 2D controls (colormap, levels, playback, ROI) to this workspace's handlers."""
        try:
            if hasattr(self, 'cbColorMapSelect_2d'):
                self.cbColorMapSelect_2d.currentTextChanged.connect(self.on_colormap_changed)
            if hasattr(self, 'cbAutoLevels'):
                self.cbAutoLevels.toggled.connect(self.on_auto_levels_toggled)
            if hasattr(self, 'btn_prev_frame'):
                self.btn_prev_frame.clicked.connect(self.previous_frame)
            if hasattr(self, 'btn_next_frame'):
                self.btn_next_frame.clicked.connect(self.next_frame)
            if hasattr(self, 'frame_spinbox'):
                self.frame_spinbox.valueChanged.connect(self.on_frame_spinbox_changed)
            if hasattr(self, 'cbLogScale'):
                self.cbLogScale.toggled.connect(self.on_log_scale_toggled)
            if hasattr(self, 'sbVmin'):
                self.sbVmin.valueChanged.connect(self.on_vmin_changed)
                try:
                    self.sbVmin.setFixedWidth(120)
                except Exception:
                    pass
            if hasattr(self, 'sbVmax'):
                self.sbVmax.valueChanged.connect(self.on_vmax_changed)
                try:
                    self.sbVmax.setFixedWidth(120)
                except Exception:
                    pass

            # Vmin/Vmax data-domain hint labels (reuse the static UI labels)
            try:
                if hasattr(self, 'label_vmin') and hasattr(self, 'label_vmax'):
                    self.lblVminHint = self.label_vmin
                    self.lblVmaxHint = self.label_vmax
                    self.lblVminHint.setText("Vmin(...):")
                    self.lblVmaxHint.setText("Vmax(...):")
                    self._vmin_vmax_hints_inserted = True
            except Exception:
                pass

            if hasattr(self, 'btnDrawROI'):
                self.btnDrawROI.clicked.connect(self.on_draw_roi_clicked)
            if hasattr(self, 'sbRefFrame'):
                self.sbRefFrame.valueChanged.connect(self.on_ref_frame_changed)
            if hasattr(self, 'sbOtherFrame'):
                self.sbOtherFrame.valueChanged.connect(self.on_other_frame_changed)

            # Playback: timer + play/pause (with media icons) + fps
            try:
                if not getattr(self, 'play_timer', None):
                    self.play_timer = QTimer(self)
                    self.play_timer.timeout.connect(self._advance_frame_playback)
                style = self.style()
                if hasattr(self, 'btn_play'):
                    self.btn_play.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
                    self.btn_play.clicked.connect(self.start_playback)
                if hasattr(self, 'btn_pause'):
                    self.btn_pause.setIcon(style.standardIcon(QStyle.SP_MediaPause))
                    self.btn_pause.clicked.connect(self.pause_playback)
                if hasattr(self, 'sb_fps'):
                    self.sb_fps.valueChanged.connect(self.on_fps_changed)
            except Exception as e:
                print(f"[PLAYBACK][Workspace2D] wiring error: {e}")

            # ROI stats readout at the bottom of the controls panel
            if hasattr(self, 'layout_2d_controls_main') and not getattr(self, 'roi_stats_label', None):
                try:
                    self.roi_stats_label = QLabel("ROI Stats: -")
                    self.roi_stats_label.setObjectName("roi_stats_label")
                    hbox = QHBoxLayout()
                    hbox.addWidget(self.roi_stats_label)
                    self.layout_2d_controls_main.addLayout(hbox)
                    self._roi_stats_label_added = True
                except Exception:
                    pass

            self._update_vmin_vmax_hints()
        except Exception as e:
            try:
                self.main_window.update_status(f"Error setting up 2D connections: {e}")
            except Exception:
                pass

    def open_roi_2d_plot_dock(self, roi):
        """Open a dockable 2D scatter plot of ROI metrics with color scale."""
        try:
            frame_data = self.get_current_frame_data()
            if frame_data is None:
                QMessageBox.information(self, "ROI 2D Plot", "No image data available.")
                return
            try:
                from dashpva.viewer.workbench.docks.rois.roi_2d_plot_dock import (
                    ROI2DPlotDock,
                )
            except Exception:
                ROI2DPlotDock = None
            if ROI2DPlotDock is None:
                QMessageBox.warning(self, "ROI 2D Plot", "ROI2DPlotDock not available.")
                return
            dock_title = f"ROI 2D: {self.main_window.get_roi_name(roi)}"
            dock = ROI2DPlotDock(self.main_window, dock_title, self.main_window, roi)
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
            try:
                self.main_window.add_dock_toggle_action(dock, dock_title, segment_name="2d")
            except Exception:
                pass
            dock.show()
            if not hasattr(self, '_roi_2d_plot_dock_widgets'):
                self._roi_2d_plot_dock_widgets = []
            self._roi_2d_plot_dock_widgets.append(dock)
            try:
                if not hasattr(self, 'roi_2d_plot_docks_by_roi_id') or self.roi_2d_plot_docks_by_roi_id is None:
                    self.roi_2d_plot_docks_by_roi_id = {}
                self.roi_2d_plot_docks_by_roi_id.setdefault(id(roi), []).append(dock)
            except Exception:
                pass
        except Exception as e:
            self.main_window.update_status(f"Error opening ROI 2D plot dock: {e}")

    def setup_2d_plot_viewer(self):
        """Set up the 2D plot viewer with PyQtGraph PlotItem and ImageView."""
        try:
            # Create the plot item and image view similar to HKL slice 2D viewer
            self.plot_item = pg.PlotItem()
            self.image_view = pg.ImageView(view=self.plot_item)

            # Manual levels override state for histogram-driven levels
            try:
                self._manual_levels_override = False
                self._manual_vmin = None
                self._manual_vmax = None
                # Cache display-domain (ImageItem) levels to preserve precision in log scale
                self._manual_vmin_display = None
                self._manual_vmax_display = None
                # Guards to avoid feedback loops between histogram and spinboxes
                self._suppress_spinbox_update = False
                self._suppress_spinbox_handlers = False
                self._suppress_histogram_update = False
            except Exception:
                pass

            # Set axis labels
            self.plot_item.setLabel('bottom', 'Columns [pixels]')
            self.plot_item.setLabel('left', 'Row [pixels]')

            # Lock aspect ratio for square pixels
            try:
                self.image_view.view.setAspectLocked(True)
            except Exception:
                pass

            # Add the image view directly to the plot host
            if hasattr(self, 'layoutPlotHost'):
                self.layoutPlotHost.addWidget(self.image_view)

                # Per-frame annotation note row beneath the image
                try:
                    note_row = QHBoxLayout()
                    self.annotation_note_label = QLabel("No annotation for this frame")
                    self.annotation_note_label.setObjectName("annotationNoteLabel")
                    self.annotation_note_label.setWordWrap(True)
                    self.annotation_edit_btn = QPushButton("Edit note")
                    self.annotation_edit_btn.clicked.connect(self._on_edit_frame_annotation)
                    note_row.addWidget(self.annotation_note_label, 1)
                    note_row.addWidget(self.annotation_edit_btn)
                    self.layoutPlotHost.addLayout(note_row)
                except Exception:
                    pass
            else:
                print("Warning: layoutPlotHost not found, 2D plot may not display correctly")

            # Overlay toggle for ROI axis mode — shown only when a saved ROI is displayed
            try:
                self._current_roi_origin = None
                self._roi_axis_toggle = QCheckBox("Dataset coordinates", parent=self.image_view)
                self._roi_axis_toggle.setVisible(False)
                self._roi_axis_toggle.move(8, 8)
                self._roi_axis_toggle.setObjectName("roiAxisToggle")
                self._roi_axis_toggle.stateChanged.connect(self._on_roi_axis_toggle)
            except Exception:
                pass


            # Initialize with empty data
            self.clear_2d_plot()

            # Setup hover overlays and mouse tracking
            self._setup_2d_hover()

            # Set default hover enabled and preserve default context menu
            try:
                self._hover_enabled = True
                if hasattr(self, 'image_view') and self.image_view is not None:
                    # Restore default context menu (do not override with custom)
                    self.image_view.setContextMenuPolicy(Qt.DefaultContextMenu)
            except Exception:
                pass

            # Wire histogram-level signals so dragging the sidebar becomes authoritative in manual mode
            try:
                self._wire_histogram_levels_signals()
                self._debug_histogram_types(prefix="[DEBUG] initial wiring")
            except Exception:
                pass

            # Start a lightweight poller to detect histogram level changes on versions that don't emit signals
            try:
                if not hasattr(self, '_histogram_poll_timer') or self._histogram_poll_timer is None:
                    self._histogram_poll_timer = QTimer(self)
                    self._histogram_poll_timer.setInterval(100)  # 10 Hz polling
                    self._histogram_poll_timer.timeout.connect(self._poll_histogram_levels)
                    self._last_hist_levels = None
                    self._histogram_poll_timer.start()
                    try:
                        print("[DEBUG] Histogram polling timer started (10 Hz)")
                        # Reduce noise: debug log instead of user-facing status
                        if hasattr(self.main_window, 'logger'):
                            self.main_window.logger.debug("Histogram polling enabled (10 Hz)")
                    except Exception:
                        pass
            except Exception:
                pass

        except Exception as e:
            self.main_window.update_status(f"Error setting up 2D plot viewer: {e}")

    def set_2d_axes(self, x_axis, y_axis):
        try:
            self.axis_2d_x = str(x_axis) if x_axis else None
            self.axis_2d_y = str(y_axis) if y_axis else None
            if hasattr(self.main_window, 'info_2d_dock') and self.main_window.info_2d_dock is not None:
                try:
                    self.main_window.info_2d_dock.refresh()
                except Exception:
                    pass
        except Exception:
            pass

    def clear_2d_plot(self):
        """Clear the 2D plot and show placeholder."""
        try:
            if hasattr(self, 'image_view'):
                # Create a small placeholder image
                placeholder = np.zeros((100, 100), dtype=np.float32)
                self.image_view.setImage(placeholder, autoLevels=False, autoRange=True)

                # Remove any existing ROIs
                if hasattr(self.main_window, 'rois') and isinstance(self.main_window.rois, list):
                    for roi in self.main_window.rois:
                        try:
                            self.image_view.removeItem(roi)
                        except Exception:
                            pass
                    self.main_window.rois.clear()
                self.main_window.current_roi = None
                # Clear docked ROI list
                if hasattr(self.main_window, 'roi_list') and self.main_window.roi_list is not None:
                    try:
                        self.main_window.roi_list.clear()
                        self.main_window.roi_by_item = {}
                        self.main_window.item_by_roi_id = {}
                    except Exception:
                        pass
                # Clear ROI stats dock
                if hasattr(self.main_window, 'roi_stats_table') and self.main_window.roi_stats_table is not None:
                    try:
                        self.main_window.roi_stats_table.setRowCount(0)
                        self.main_window.stats_row_by_roi_id = {}
                    except Exception:
                        pass

                # Set default axis labels
                self.plot_item.setLabel('bottom', 'X')
                self.plot_item.setLabel('left', 'Y')
                try:
                    self.set_2d_axes("Columns", "Row")
                except Exception:
                    pass
                try:
                    if hasattr(self.main_window, 'info_2d_dock') and self.main_window.info_2d_dock is not None:
                        self.main_window.info_2d_dock.refresh()
                except Exception:
                    pass

                # Update Vmin/Vmax hint labels to placeholders when plot is cleared
                try:
                    self._update_vmin_vmax_hints()
                except Exception:
                    pass

                # Update above-image info label with placeholder dimensions
                if hasattr(self, 'image_info_label'):
                    try:
                        self.image_info_label.setText("Image Dimensions: 100x100 pixels")
                    except Exception:
                        pass

                # Remove hover overlays and clear HKL caches
                try:
                    view = self.image_view.getView() if hasattr(self.image_view, 'getView') else None
                    if view is not None:
                        if hasattr(self, '_hover_hline') and self._hover_hline is not None:
                            try:
                                view.removeItem(self._hover_hline)
                            except Exception:
                                pass
                            self._hover_hline = None
                        if hasattr(self, '_hover_vline') and self._hover_vline is not None:
                            try:
                                view.removeItem(self._hover_vline)
                            except Exception:
                                pass
                            self._hover_vline = None
                        if hasattr(self, '_hover_text') and self._hover_text is not None:
                            try:
                                view.removeItem(self._hover_text)
                            except Exception:
                                pass
                            self._hover_text = None
                    self._mouse_proxy = None
                    self._qx_grid = None
                    self._qy_grid = None
                    self._qz_grid = None
                except Exception:
                    pass

        except Exception as e:
            self.main_window.update_status(f"Error clearing 2D plot: {e}")

    def update_overlay_text(self, width, height, frame_info=None):
        """Update the label above the image with dimensions and optional frame info.
        Augmented to append current motor position (if available) for the selected frame.
        """
        try:
            text = f"Image Dimensions: {width}x{height} pixels"
            info = frame_info or ""
            # Append current data-domain Vmin/Vmax so they are visible in the 2D space
            try:
                frame = self.get_current_frame_data()
            except Exception:
                frame = None
            try:
                vmin_txt = "Vmin(...):"
                vmax_txt = "Vmax(...):"
                if frame is not None:
                    arr = np.asarray(frame)
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        dmin = int(np.min(finite))
                        dmax = int(np.max(finite))
                        vmin_txt = f"Vmin({dmin}):"
                        vmax_txt = f"Vmax({dmax}):"
                vv = f"{vmin_txt} {vmax_txt}"
                info = f"{info} | {vv}" if info else vv
            except Exception:
                pass
            # Try to append motor position for current frame if 3D data
            try:
                if hasattr(self, 'current_2d_data') and self.current_2d_data is not None and self.current_2d_data.ndim == 3:
                    num_frames = int(self.current_2d_data.shape[0])
                    idx = 0
                    if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                        try:
                            idx = int(self.frame_spinbox.value())
                        except Exception:
                            idx = 0
                    motor_val = None
                    fp = getattr(self.main_window, 'current_file_path', None)
                    if fp and os.path.exists(fp):
                        try:
                            with h5py.File(fp, 'r') as h5f:
                                arr = self._find_motor_positions(h5f, num_frames)
                                if arr is not None and 0 <= idx < arr.size:
                                    motor_val = float(arr[idx])
                        except Exception:
                            motor_val = None
                    if motor_val is not None:
                        if info:
                            info = f"{info} | Motor {motor_val:.6f}"
                        else:
                            info = f"Motor {motor_val:.6f}"
            except Exception:
                pass
            if info:
                text = f"{text} ({info})"
            if hasattr(self, 'image_info_label'):
                self.image_info_label.setText(text)
        except Exception as e:
            self.main_window.update_status(f"Error updating image info label: {e}")

    def _find_motor_positions(self, h5f, num_frames: int):
        """Return a 1-D array of motor positions whose length matches num_frames.

        Searches entry/data/metadata/motor_positions for axis-labeled datasets
        (no ':' in name). Returns the first one whose length equals num_frames,
        or None if none is found.
        """
        try:
            motor_group_path = 'entry/data/metadata/motor_positions'
            if motor_group_path not in h5f:
                return None
            grp = h5f[motor_group_path]
            for key in grp.keys():
                if ':' in key:
                    continue
                try:
                    arr = np.asarray(grp[key], dtype=float).ravel()
                    if arr.size == num_frames:
                        return arr
                except Exception:
                    pass
        except Exception:
            pass
        return None

    def _get_histogram_widget(self):
        try:
            if hasattr(self, 'image_view') and self.image_view is not None:
                if hasattr(self.image_view, 'getHistogramWidget'):
                    return self.image_view.getHistogramWidget()
                # Fallback for ImageView internals
                ui = getattr(self.image_view, 'ui', None)
                if ui is not None and hasattr(ui, 'histogram'):
                    return ui.histogram
        except Exception:
            pass
        return None

    def _wire_histogram_levels_signals(self):
        """Connect HistogramLUTWidget signals so that user drags update the manual vmin/vmax.

        Preferred: hist.sigLevelChange
        Fallback: hist.region.sigRegionChanged / sigRegionChangeFinished.
        """
        try:
            hist = self._get_histogram_widget()
            if hist is None:
                try:
                    self.main_window.update_status("Histogram widget not found; levels sync disabled")
                except Exception:
                    pass
                return

            connected = False
            # Prefer dedicated level-change signal on the item
            try:
                item = getattr(hist, 'item', None)
                if item is not None:
                    if hasattr(item, 'sigLevelsChanged'):
                        item.sigLevelsChanged.connect(self._on_histogram_levels_changed)
                        connected = True
                    elif hasattr(item, 'sigLevelChange'):
                        item.sigLevelChange.connect(self._on_histogram_levels_changed)
                        connected = True
                    # Also listen to region signals attached to the item (some versions)
                    try:
                        iregion = getattr(item, 'region', None)
                        if iregion is not None:
                            if hasattr(iregion, 'sigRegionChanged'):
                                iregion.sigRegionChanged.connect(lambda _=None: self._on_histogram_levels_changed())
                                connected = True
                            if hasattr(iregion, 'sigRegionChangeFinished'):
                                iregion.sigRegionChangeFinished.connect(lambda _=None: self._on_histogram_levels_changed())
                                connected = True
                    except Exception:
                        pass
            except Exception:
                item = None

            # Some versions may emit on the widget directly
            try:
                if not connected and hasattr(hist, 'sigLevelsChanged'):
                    hist.sigLevelsChanged.connect(self._on_histogram_levels_changed)
                    connected = True
                if not connected and hasattr(hist, 'sigLevelChange'):
                    hist.sigLevelChange.connect(self._on_histogram_levels_changed)
                    connected = True
            except Exception:
                pass

            # Fallback: listen to the draggable region signals on the widget
            try:
                if not connected:
                    region = getattr(hist, 'region', None)
                    if region is not None:
                        if hasattr(region, 'sigRegionChanged'):
                            region.sigRegionChanged.connect(lambda _=None: self._on_histogram_levels_changed())
                            connected = True
                        if hasattr(region, 'sigRegionChangeFinished'):
                            region.sigRegionChangeFinished.connect(lambda _=None: self._on_histogram_levels_changed())
                            connected = True
            except Exception:
                pass

            # Log connection status and types to aid runtime diagnosis
            try:
                if connected:
                    # Reduce noise: debug log instead of user-facing status
                    msg = "Histogram signals wired: "
                    srcs = []
                    try:
                        if item is not None:
                            srcs.append(f"item={type(item).__name__}")
                    except Exception:
                        pass
                    try:
                        srcs.append(f"widget={type(hist).__name__}")
                    except Exception:
                        pass
                    try:
                        if hasattr(self.main_window, 'logger'):
                            self.main_window.logger.debug(msg + ", ".join(srcs))
                    except Exception:
                        pass
                else:
                    itype = type(getattr(hist, 'item', None)).__name__ if hasattr(hist, 'item') else 'None'
                    wtype = type(hist).__name__
                    self.main_window.update_status(f"Could not wire histogram signals; widget={wtype}, item={itype}")
            except Exception:
                pass
        except Exception:
            pass

    def _on_histogram_levels_changed(self):
        """Handle histogram level drag: make histogram authoritative and update spinboxes.

        Reads histogram's current display-domain levels, converts to data domain if log scale,
        sets manual override, updates sbVmin/sbVmax (guarded), and refreshes image.
        """
        try:
            # Avoid feedback loop if we are programmatically syncing histogram
            if bool(getattr(self, '_suppress_histogram_update', False)):
                return
            if not hasattr(self, 'image_view') or self.image_view is None:
                return
            hist = self._get_histogram_widget()
            if hist is None:
                return
            # Get current histogram levels in display (ImageItem) domain
            vmin_d = vmax_d = None
            try:
                # Try widget first
                if hasattr(hist, 'getLevels'):
                    vmin_d, vmax_d = hist.getLevels()
                else:
                    raise AttributeError
            except Exception:
                # Try item
                try:
                    item = getattr(hist, 'item', None)
                    if item is not None and hasattr(item, 'getLevels'):
                        vmin_d, vmax_d = item.getLevels()
                except Exception:
                    # Fallback: read region positions
                    try:
                        region = getattr(hist, 'region', None)
                        if region is None:
                            region = getattr(getattr(hist, 'item', None), 'region', None)
                        if region is not None and hasattr(region, 'getRegion'):
                            vmin_d, vmax_d = region.getRegion()
                    except Exception:
                        vmin_d = vmax_d = None
            if vmin_d is None or vmax_d is None:
                return
            # Cache last polled levels to avoid redundant updates
            try:
                self._last_hist_levels = (float(vmin_d), float(vmax_d))
            except Exception:
                pass
            # Convert to data domain if log scale is enabled
            try:
                if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                    vmin_data = float(np.expm1(max(0.0, float(vmin_d))))
                    vmax_data = float(np.expm1(max(0.0, float(vmax_d))))
                else:
                    vmin_data = float(vmin_d)
                    vmax_data = float(vmax_d)
            except Exception:
                vmin_data = vmin_d
                vmax_data = vmax_d
            if not (np.isfinite(vmin_data) and np.isfinite(vmax_data)):
                return
            if vmax_data <= vmin_data:
                # Enforce ordering; small epsilon to avoid equality
                vmax_data = vmin_data + 1e-6

            # Set manual override and cache
            try:
                self._manual_levels_override = True
                self._manual_vmin = float(vmin_data)
                self._manual_vmax = float(vmax_data)
                # Cache display-domain levels to reuse exactly during refresh in log scale
                self._manual_vmin_display = float(vmin_d)
                self._manual_vmax_display = float(vmax_d)
            except Exception:
                pass

            # Update spinboxes to reflect histogram-driven levels without triggering handlers
            try:
                self._suppress_spinbox_update = True
                self._suppress_spinbox_handlers = True
                if hasattr(self, 'sbVmin') and self.sbVmin is not None:
                    # Keep ranges wide enough to include manual values to avoid clamping
                    try:
                        current_data = self.get_current_frame_data()
                        if current_data is not None:
                            dmin = int(np.min(current_data))
                            dmax = int(np.max(current_data))
                        else:
                            dmin = 0
                            dmax = 1
                        lower = min(dmin, int(self._manual_vmin))
                        upper = max(int(dmax * 2), int(self._manual_vmax))
                        self.sbVmin.setRange(lower, upper)
                    except Exception:
                        pass
                    self.sbVmin.setValue(self._manual_vmin)
                if hasattr(self, 'sbVmax') and self.sbVmax is not None:
                    try:
                        current_data = self.get_current_frame_data()
                        if current_data is not None:
                            dmin = int(np.min(current_data))
                            dmax = int(np.max(current_data))
                        else:
                            dmin = 0
                            dmax = 1
                        lower = min(dmin + 1, int(self._manual_vmin) + 1)
                        upper = max(int(dmax * 2), int(self._manual_vmax))
                        self.sbVmax.setRange(lower, upper)
                    except Exception:
                        pass
                    self.sbVmax.setValue(self._manual_vmax)
                # Also reflect histogram-driven values into the hint labels near the controls
                try:
                    if hasattr(self, 'lblVminHint') and self.lblVminHint is not None:
                        self.lblVminHint.setText(f"Vmin({int(self._manual_vmin)}):")
                    if hasattr(self, 'lblVmaxHint') and self.lblVmaxHint is not None:
                        self.lblVmaxHint.setText(f"Vmax({int(self._manual_vmax)}):")
                except Exception:
                    pass
            finally:
                try:
                    self._suppress_spinbox_update = False
                    self._suppress_spinbox_handlers = False
                except Exception:
                    pass

            # Refresh current frame to apply manual levels persistently
            try:
                self._refresh_current_frame_image()
            except Exception:
                pass

            # Debug logging to help verify runtime behavior without spamming INFO logs
            try:
                if hasattr(self.main_window, 'logger'):
                    self.main_window.logger.debug(
                        f"Histogram drag -> display [{vmin_d:.3f}, {vmax_d:.3f}] -> data [{vmin_data:.3f}, {vmax_data:.3f}]"
                    )
            except Exception:
                pass
        except Exception:
            pass

    def _poll_histogram_levels(self):
        """Fallback polling for histogram levels to ensure UI sync when signals are not emitted.

        Checks at ~10 Hz during manual levels mode. If levels change compared to the last seen,
        triggers _on_histogram_levels_changed()."""
        try:
            # Skip when auto levels are on or when we are programmatically updating histogram
            if bool(getattr(self, '_suppress_histogram_update', False)):
                return
            auto_levels = hasattr(self, 'cbAutoLevels') and self.cbAutoLevels.isChecked()
            if auto_levels:
                return
            hist = self._get_histogram_widget()
            if hist is None:
                return
            vmin_d = vmax_d = None
            try:
                if hasattr(hist, 'getLevels'):
                    vmin_d, vmax_d = hist.getLevels()
                else:
                    item = getattr(hist, 'item', None)
                    if item is not None and hasattr(item, 'getLevels'):
                        vmin_d, vmax_d = item.getLevels()
            except Exception:
                # Last fallback to region
                try:
                    region = getattr(hist, 'region', None)
                    if region is None:
                        region = getattr(getattr(hist, 'item', None), 'region', None)
                    if region is not None and hasattr(region, 'getRegion'):
                        vmin_d, vmax_d = region.getRegion()
                except Exception:
                    vmin_d = vmax_d = None
            if vmin_d is None or vmax_d is None:
                return
            try:
                cur = (float(vmin_d), float(vmax_d))
            except Exception:
                cur = (vmin_d, vmax_d)
            prev = getattr(self, '_last_hist_levels', None)
            changed = False
            try:
                if prev is None:
                    changed = True
                else:
                    changed = (abs(cur[0] - prev[0]) > 1e-9) or (abs(cur[1] - prev[1]) > 1e-9)
            except Exception:
                changed = True
            if changed:
                self._last_hist_levels = cur
                # Delegate to the authoritative handler
                self._on_histogram_levels_changed()
        except Exception:
            pass

    def _apply_display_transform(self, frame: np.ndarray) -> np.ndarray:
        """Return frame transformed for display (log1p if log scale is enabled)."""
        try:
            arr = np.asarray(frame, dtype=np.float32)
            log_enabled = hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked()
            if log_enabled:
                # log1p on non-negative domain for stable display
                return np.log1p(np.maximum(arr, 0))
            return arr
        except Exception:
            # Fallback to input frame
            return frame

    def _apply_levels_for_current_mode(self, display_data: np.ndarray):
        """Apply levels to image and sync HistogramLUT widget based on auto/manual and display domain.

        - Auto Levels ON: keep auto levels
        sync histogram range to display_data min/max
        - Auto Levels OFF: map vmin/vmax from controls into display domain and set both ImageItem and Histogram
        """
        try:
            if not hasattr(self, 'image_view') or self.image_view is None:
                return
            auto_levels = hasattr(self, 'cbAutoLevels') and self.cbAutoLevels.isChecked()
            hist = self._get_histogram_widget()

            if auto_levels:
                # Sync histogram range with displayed domain for consistent sidebar
                try:
                    finite_vals = display_data[np.isfinite(display_data)] if isinstance(display_data, np.ndarray) else None
                    if finite_vals is not None and finite_vals.size > 0:
                        dmin = float(np.min(finite_vals))
                        dmax = float(np.max(finite_vals))
                    else:
                        dmin = float(np.nanmin(display_data))
                        dmax = float(np.nanmax(display_data))
                    if hist is not None and np.isfinite(dmin) and np.isfinite(dmax) and dmax > dmin:
                        # Update histogram range and levels to reflect the new dataset/frame
                        hist.setHistogramRange(dmin, dmax)
                        try:
                            self._suppress_histogram_update = True
                            hist.setLevels(dmin, dmax)
                        finally:
                            self._suppress_histogram_update = False
                except Exception:
                    pass
                return

            # Manual levels: compute display-domain levels from caches if available (preserve precision and avoid spinbox rounding)
            vmin_d = vmax_d = None
            try:
                if bool(getattr(self, '_manual_levels_override', False)):
                    d0 = getattr(self, '_manual_vmin_display', None)
                    d1 = getattr(self, '_manual_vmax_display', None)
                    if d0 is not None and d1 is not None:
                        vmin_d = float(d0)
                        vmax_d = float(d1)
            except Exception:
                vmin_d = vmax_d = None
            if vmin_d is None or vmax_d is None:
                # Fall back to spinbox -> display mapping
                vmin = None
                vmax = None
                try:
                    vmin = float(self.sbVmin.value()) if hasattr(self, 'sbVmin') else None
                    vmax = float(self.sbVmax.value()) if hasattr(self, 'sbVmax') else None
                except Exception:
                    vmin = vmax = None
                if vmin is None or vmax is None or vmax <= vmin:
                    return

                try:
                    if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                        vmin_d = float(np.log1p(max(0.0, vmin)))
                        vmax_d = float(np.log1p(max(0.0, vmax)))
                    else:
                        vmin_d = float(vmin)
                        vmax_d = float(vmax)
                except Exception:
                    vmin_d = vmin
                    vmax_d = vmax

            # Apply to image item
            try:
                self.image_view.setLevels(min=vmin_d, max=vmax_d)
            except Exception:
                pass

            # Sync histogram widget levels for visual alignment
            try:
                if hist is not None:
                    # Guard to avoid triggering histogram-level signal feedback
                    try:
                        self._suppress_histogram_update = True
                        hist.setLevels(vmin_d, vmax_d)
                    finally:
                        self._suppress_histogram_update = False
                    # Keep spinboxes in sync with the histogram levels even when set programmatically
                    try:
                        # Map display-domain -> data-domain for spinboxes
                        if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                            vmin_data = float(np.expm1(max(0.0, float(vmin_d))))
                            vmax_data = float(np.expm1(max(0.0, float(vmax_d))))
                        else:
                            vmin_data = float(vmin_d)
                            vmax_data = float(vmax_d)
                        # Update without firing handlers
                        self._suppress_spinbox_handlers = True
                        if hasattr(self, 'sbVmin') and self.sbVmin is not None:
                            try:
                                self.sbVmin.setValue(vmin_data)
                            except Exception:
                                pass
                        if hasattr(self, 'sbVmax') and self.sbVmax is not None:
                            try:
                                self.sbVmax.setValue(vmax_data)
                            except Exception:
                                pass
                        # Also reflect into the hint labels near the spinboxes
                        try:
                            if hasattr(self, 'lblVminHint') and self.lblVminHint is not None:
                                self.lblVminHint.setText(f"Vmin({int(vmin_data)}):")
                            if hasattr(self, 'lblVmaxHint') and self.lblVmaxHint is not None:
                                self.lblVmaxHint.setText(f"Vmax({int(vmax_data)}):")
                        except Exception:
                            pass
                    finally:
                        try:
                            self._suppress_spinbox_handlers = False
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            pass

    def _refresh_current_frame_image(self, frame_data: np.ndarray = None):
        """Centralized refresh for current frame image: transform, setImage, and sync levels/histogram.

        Args:
            frame_data: Optional pre-sliced frame. If None, uses get_current_frame_data().
        """
        try:
            if not hasattr(self, 'image_view') or self.image_view is None:
                return
            # Resolve frame
            try:
                frame = frame_data if frame_data is not None else self.get_current_frame_data()
            except Exception:
                frame = frame_data
            if frame is None:
                return
            # Transform to display domain
            display_data = self._apply_display_transform(frame)

            auto_levels = hasattr(self, 'cbAutoLevels') and self.cbAutoLevels.isChecked()
            # Auto-range behavior: only force on first display when requested
            auto_range = bool(getattr(self, '_force_auto_range_next', False))

            # Compute explicit levels for manual mode and pass into setImage to prevent auto-scaling during playback
            levels_arg = None
            if not auto_levels:
                # Prefer display-domain manual caches when manual override is active (both linear and log)
                try:
                    use_cached_display = bool(getattr(self, '_manual_levels_override', False))
                except Exception:
                    use_cached_display = False
                if use_cached_display:
                    d0 = getattr(self, '_manual_vmin_display', None)
                    d1 = getattr(self, '_manual_vmax_display', None)
                    if d0 is not None and d1 is not None and np.isfinite(d0) and np.isfinite(d1) and d1 > d0:
                        levels_arg = (float(d0), float(d1))
                if levels_arg is None:
                    # Fallback to spinbox mapping
                    try:
                        vmin = float(self.sbVmin.value()) if hasattr(self, 'sbVmin') else None
                        vmax = float(self.sbVmax.value()) if hasattr(self, 'sbVmax') else None
                    except Exception:
                        vmin = vmax = None
                    if vmin is not None and vmax is not None and vmax > vmin:
                        try:
                            if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                                vmin_d = float(np.log1p(max(0.0, vmin)))
                                vmax_d = float(np.log1p(max(0.0, vmax)))
                            else:
                                vmin_d = float(vmin)
                                vmax_d = float(vmax)
                            if np.isfinite(vmin_d) and np.isfinite(vmax_d) and vmax_d > vmin_d:
                                levels_arg = (vmin_d, vmax_d)
                        except Exception:
                            levels_arg = None

            try:
                if levels_arg is not None:
                    self.image_view.setImage(
                        display_data,
                        autoLevels=False,
                        autoRange=auto_range,
                        autoHistogramRange=False,
                        levels=levels_arg
                    )
                else:
                    self.image_view.setImage(
                        display_data,
                        autoLevels=auto_levels,
                        autoRange=auto_range,
                        autoHistogramRange=auto_levels
                    )
            finally:
                try:
                    self._force_auto_range_next = False
                except Exception:
                    pass

            # Apply/Sync levels and histogram bar based on current mode
            self._apply_levels_for_current_mode(display_data)
        except Exception:
            pass

    def display_2d_data(self, data):
        """Display 2D or 3D numeric data in the PyQtGraph ImageView."""
        try:
            if not hasattr(self, 'image_view'):
                print("Warning: ImageView not initialized")
                return

            # New dataset: clear manual override so initial seeding from data makes sense
            try:
                self._manual_levels_override = False
                self._manual_vmin = None
                self._manual_vmax = None
                self._manual_vmin_display = None
                self._manual_vmax_display = None
            except Exception:
                pass

            # Store the original data for frame navigation
            self.current_2d_data = data
            try:
                print(f"[DISPLAY] data ndim={getattr(data,'ndim',None)}, shape={getattr(data,'shape',None)}")
            except Exception:
                pass

            # Handle different data dimensions
            if data.ndim == 2:
                # 2D data - display directly
                image_data = np.asarray(data, dtype=np.float32)

                # Update frame controls for 2D data
                self.update_frame_controls_for_2d_data()

                height, width = image_data.shape
                if hasattr(self, 'frame_info_label'):
                    self.frame_info_label.setText(f"Image Dimensions: {width}x{height} pixels")
                self._notify_annotation_frame_changed()
                # Update overlay text
                self.update_overlay_text(width, height, None)

            elif data.ndim == 3:
                # 3D data - display first frame and set up navigation
                image_data = np.asarray(data[0], dtype=np.float32)

                # Update frame controls for 3D data
                num_frames = data.shape[0]
                self.update_frame_controls_for_3d_data(num_frames)

                height, width = image_data.shape
                if hasattr(self, 'frame_info_label'):
                    self.frame_info_label.setText(f"Image Dimensions: {width}x{height} pixels (frame 0 of {num_frames})")
                self._notify_annotation_frame_changed()
                # Update overlay text
                self.update_overlay_text(width, height, f"Frame 0 of {num_frames}")

            else:
                print(f"Unsupported data dimensions: {data.ndim}")
                return

            # Set the image data through centralized refresh pipeline
            # Force auto range on first display of new data
            try:
                self._force_auto_range_next = True
            except Exception:
                pass
            self._refresh_current_frame_image(image_data)
            # Ensure hover overlays exist after any prior clear
            try:
                self._setup_2d_hover()
            except Exception:
                pass

            # Re-wire histogram after first image is set to ensure item/region exist
            try:
                self._wire_histogram_levels_signals()
                self._debug_histogram_types(prefix="[DEBUG] post-image wiring")
            except Exception:
                pass

            # Update axis labels based on data shape
            height, width = image_data.shape
            self.plot_item.setLabel('bottom', f'Columns [pixels] (0 to {width-1})')
            self.plot_item.setLabel('left', f'Row [pixels] (0 to {height-1})')
            try:
                self.set_2d_axes("Columns", "Row")
            except Exception:
                pass

            # Apply current colormap
            if hasattr(self, 'cbColorMapSelect_2d'):
                current_colormap = self.cbColorMapSelect_2d.currentText()
                self.apply_colormap(current_colormap)

            # Update speckle analysis controls programmatically
            self.update_speckle_controls_for_data(data)

            # Update vmin/vmax controls based on data (ranges only if user override is active)
            self.update_vmin_vmax_controls_for_data(image_data)

            # Refresh ROI stats for current frame/data
            try:
                self.main_window.roi_manager.update_all_roi_stats()
            except Exception:
                pass
            try:
                if hasattr(self.main_window, 'info_2d_dock') and self.main_window.info_2d_dock is not None:
                    self.main_window.info_2d_dock.refresh()
            except Exception:
                pass
            # Refresh any open ROI Plot docks to reflect dataset change (axes and series)
            try:
                docks = []
                if hasattr(self.main_window, '_roi_plot_dock_widgets') and self.main_window._roi_plot_dock_widgets:
                    docks.extend(list(self.main_window._roi_plot_dock_widgets))
                if hasattr(self.main_window, 'roi_plot_docks_by_roi_id') and self.main_window.roi_plot_docks_by_roi_id:
                    for lst in self.main_window.roi_plot_docks_by_roi_id.values():
                        docks.extend(list(lst))
                for d in docks:
                    try:
                        if hasattr(d, 'refresh_for_dataset_change'):
                            d.refresh_for_dataset_change()
                    except Exception:
                        continue
            except Exception:
                pass

        except Exception as e:
            self.main_window.update_status(f"Error displaying 2D data: {e}")

    def _show_image_context_menu(self, pos):
        """Show right-click menu for the 2D image with hover toggle and HKL plotting."""
        try:
            menu = QMenu(self)
            # Enable/Disable Hover
            action_hover = QAction("Enable Hover", self)
            action_hover.setCheckable(True)
            action_hover.setChecked(bool(getattr(self, '_hover_enabled', True)))
            action_hover.toggled.connect(self._toggle_hover_enabled)
            menu.addAction(action_hover)
            # Show current hover state explicitly (do not remove original options)
            try:
                state_text = "Hover: ON" if bool(getattr(self, '_hover_enabled', True)) else "Hover: OFF"
            except Exception:
                state_text = "Hover: ON"
            action_state = QAction(state_text, self)
            action_state.setEnabled(False)
            menu.addAction(action_state)
            # Show last HKL value if available (disabled info item)
            hkl_label = "HKL: N/A"
            try:
                xy = getattr(self, '_last_hover_xy', None)
                qxg = getattr(self, '_qx_grid', None)
                qyg = getattr(self, '_qy_grid', None)
                qzg = getattr(self, '_qz_grid', None)
                if xy and qxg is not None and qyg is not None and qzg is not None:
                    x, y = int(xy[0]), int(xy[1])
                    if qxg.ndim == 3 and qyg.ndim == 3 and qzg.ndim == 3:
                        idx = 0
                        try:
                            idx = int(self.frame_spinbox.value()) if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled() else 0
                        except Exception:
                            idx = 0
                        if 0 <= idx < qxg.shape[0]:
                            H = float(qxg[idx, y, x])
                            K = float(qyg[idx, y, x])
                            L = float(qzg[idx, y, x])
                            hkl_label = f"HKL: H={H:.6f}, K={K:.6f}, L={L:.6f}"
                    elif qxg.ndim == 2 and qyg.ndim == 2 and qzg.ndim == 2:
                        H = float(qxg[y, x])
                        K = float(qyg[y, x])
                        L = float(qzg[y, x])
                        hkl_label = f"HKL: H={H:.6f}, K={K:.6f}, L={L:.6f}"
            except Exception:
                pass
            action_hkl_info = QAction(hkl_label, self)
            action_hkl_info.setEnabled(False)
            menu.addAction(action_hkl_info)
            # Plot HKL in 3D
            action_plot_hkl = QAction("Plot HKL (3D)", self)
            action_plot_hkl.setToolTip("Plot current frame intensity at HKL (qx,qy,qz) points")
            action_plot_hkl.triggered.connect(self._plot_current_hkl_points)
            menu.addAction(action_plot_hkl)
            # Show menu at global position
            try:
                gpos = self.image_view.mapToGlobal(pos)
            except Exception:
                gpos = QCursor.pos()
            menu.exec_(gpos)
        except Exception as e:
            self.main_window.update_status(f"Error showing image context menu: {e}")

    def _toggle_hover_enabled(self, enabled: bool):
        try:
            self._hover_enabled = bool(enabled)
            self._update_hover_visibility()
        except Exception:
            pass

    def _update_hover_visibility(self):
        try:
            visible = bool(getattr(self, '_hover_enabled', True))
            for item_name in ['_hover_hline', '_hover_vline', '_hover_text']:
                it = getattr(self, item_name, None)
                try:
                    if it is not None:
                        it.setVisible(visible)
                except Exception:
                    pass
        except Exception:
            pass

    def mark_pixel(self, row: int, col: int, frame=None):
        """Navigate to ``frame`` (if given) and circle the pixel at (row, col).

        Row/col index the frame array (axis0/axis1). The circle is drawn in the
        same plot coordinates the hover crosshair uses (x=row, y=col).
        """
        try:
            if (frame is not None and getattr(self, 'frame_spinbox', None) is not None
                    and self.frame_spinbox.isEnabled()):
                lo, hi = self.frame_spinbox.minimum(), self.frame_spinbox.maximum()
                self.frame_spinbox.setValue(max(lo, min(hi, int(frame))))
            if getattr(self, '_peak_marker', None) is None:
                self._peak_marker = pg.ScatterPlotItem(
                    size=26, symbol='o', pen=pg.mkPen((255, 60, 60), width=2),
                    brush=pg.mkBrush(None))
                self._peak_marker.setZValue(100)
                self.plot_item.addItem(self._peak_marker)
            self._peak_marker.setData([float(row)], [float(col)])
            self._peak_marker.setVisible(True)
        except Exception:
            pass

    def clear_peak_marker(self):
        """Hide the intensity-navigation circle (e.g. when the dataset changes)."""
        try:
            if getattr(self, '_peak_marker', None) is not None:
                self._peak_marker.setVisible(False)
                self._peak_marker.clear()
        except Exception:
            pass

    def _hkl_at_pixel(self, x: int, y: int):
        """Return (H, K, L) at pixel (x, y) on the current frame, or (None, None, None)."""
        try:
            qxg = getattr(self, '_qx_grid', None)
            qyg = getattr(self, '_qy_grid', None)
            qzg = getattr(self, '_qz_grid', None)
            if qxg is None or qyg is None or qzg is None:
                return None, None, None
            if qxg.ndim == 3 and qyg.ndim == 3 and qzg.ndim == 3:
                idx = int(self.frame_spinbox.value()) if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled() else 0
                if 0 <= idx < qxg.shape[0]:
                    return float(qxg[idx, y, x]), float(qyg[idx, y, x]), float(qzg[idx, y, x])
            elif qxg.ndim == 2 and qyg.ndim == 2 and qzg.ndim == 2:
                return float(qxg[y, x]), float(qyg[y, x]), float(qzg[y, x])
        except Exception:
            pass
        return None, None, None

    def _update_hover_readout(self, x: int, y: int, intensity: float):
        """Push (pixel, intensity, HKL) to the 2D Info dock Mouse section."""
        H_val, K_val, L_val = self._hkl_at_pixel(x, y)
        try:
            dock = getattr(self.main_window, 'info_2d_dock', None)
            if dock is not None:
                dock.set_mouse_info((x, y), intensity, H_val, K_val, L_val)
        except Exception:
            pass

    def _update_hover_text_at(self, x: int, y: int):
        """Update hover crosshair and tooltip for given pixel coordinates on current frame."""
        try:
            frame = self.get_current_frame_data()
            if frame is None or frame.ndim != 2:
                return
            height, width = frame.shape
            if x < 0 or y < 0 or x >= width or y >= height:
                return
            # Update crosshair positions
            try:
                if hasattr(self, '_hover_hline') and self._hover_hline is not None:
                    self._hover_hline.setPos(float(y))
                if hasattr(self, '_hover_vline') and self._hover_vline is not None:
                    self._hover_vline.setPos(float(x))
            except Exception:
                pass
            # Intensity
            try:
                intensity = float(frame[x, y])
            except Exception:
                intensity = float('nan')
            # Update the 2D Info dock at this pixel
            self._update_hover_readout(x, y, intensity)
        except Exception:
            pass

    def _plot_current_hkl_points(self):
        """Plot current frame intensities at HKL positions in an HKL 3D Plot Dock."""
        try:
            # Ensure q-grids are available
            if getattr(self, '_qx_grid', None) is None or getattr(self, '_qy_grid', None) is None or getattr(self, '_qz_grid', None) is None:
                try:
                    self._try_load_hkl_grids()
                except Exception:
                    pass
            qxg = getattr(self, '_qx_grid', None)
            qyg = getattr(self, '_qy_grid', None)
            qzg = getattr(self, '_qz_grid', None)
            frame = self.get_current_frame_data()
            if qxg is None or qyg is None or qzg is None or frame is None:
                self.main_window.update_status("HKL grids or frame not available for plotting")
                return
            # Select frame index if 3D
            idx = 0
            try:
                idx = int(self.frame_spinbox.value()) if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled() else 0
            except Exception:
                idx = 0
            # Extract H,K,L arrays matching frame
            try:
                if qxg.ndim == 3:
                    qx = qxg[idx]
                    qy = qyg[idx]
                    qz = qzg[idx]
                else:
                    qx = qxg
                    qy = qyg
                    qz = qzg
            except Exception:
                self.main_window.update_status("Error extracting HKL arrays for current frame")
                return
            # Build points and intensities
            try:
                H = np.asarray(qx, dtype=np.float32).ravel()
                K = np.asarray(qy, dtype=np.float32).ravel()
                L = np.asarray(qz, dtype=np.float32).ravel()
                points = np.column_stack([H, K, L])
                intens = np.asarray(frame, dtype=np.float32).ravel()
            except Exception:
                self.main_window.update_status("Error building HKL points")
                return
            # Create or reuse HKL 3D Plot Dock
            try:
                from dashpva.viewer.workbench.hkl_3d_plot_dock import HKL3DPlotDock
            except Exception:
                HKL3DPlotDock = None
            if HKL3DPlotDock is None:
                self.main_window.update_status("HKL3DPlotDock not available")
                return
            if not hasattr(self, '_hkl3d_plot_dock') or self._hkl3d_plot_dock is None:
                dock_title = "HKL 3D Plot"
                dock = HKL3DPlotDock(self, dock_title, self)
                self.main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
                try:
                    self.main_window.add_dock_toggle_action(dock, dock_title, segment_name="2d")
                except Exception:
                    pass
                dock.show()
                self._hkl3d_plot_dock = dock
            # Plot points
            try:
                self._hkl3d_plot_dock._plot_points(points, intens)
                self.main_window.update_status("Plotted HKL points for current frame")
            except Exception as e:
                self.main_window.update_status(f"Error plotting HKL points: {e}")
        except Exception as e:
            self.main_window.update_status(f"Error in HKL plot: {e}")

    def _update_hkl3d_plot_for_current_frame(self):
        """If HKL 3D plot dock is open, update it to current frame."""
        try:
            if hasattr(self, '_hkl3d_plot_dock') and self._hkl3d_plot_dock is not None:
                self._plot_current_hkl_points()
        except Exception:
            pass

    def _on_roi_axis_toggle(self, state: int) -> None:
        """Switch the image axes between the ROI's own pixel space and the source dataset coordinates."""
        try:
            origin = getattr(self, '_current_roi_origin', None)
            if origin is None:
                return
            x, y, w, h = origin
            img_item = getattr(self.image_view, 'imageItem', None)
            if img_item is None:
                return
            checked = bool(state)
            if checked:
                img_item.setRect(QRectF(x, y, w, h))
                self.plot_item.setLabel('bottom', f'Columns [pixels]  origin={x}')
                self.plot_item.setLabel('left',   f'Row [pixels]  origin={y}')
            else:
                img_item.resetTransform()
                self.plot_item.setLabel('bottom', 'Columns [pixels]')
                self.plot_item.setLabel('left',   'Row [pixels]')
        except Exception:
            pass

    def _notify_annotation_frame_changed(self) -> None:
        """Coordination only: tell the annotation dock the frame changed so it updates its overlays + note label."""
        try:
            dock = getattr(self.main_window, 'annotation_dock', None)
            if dock is not None:
                dock.on_frame_changed()
        except Exception:
            pass

    def reset_roi_axis(self) -> None:
        """Clear ROI-origin overlay state and hide the dataset-coordinates toggle (non-ROI datasets)."""
        try:
            self._current_roi_origin = None
            if hasattr(self, '_roi_axis_toggle'):
                self._roi_axis_toggle.setChecked(False)
                self._roi_axis_toggle.setVisible(False)
        except Exception:
            pass

    def set_roi_axis_origin(self, origin) -> None:
        """Set the ROI-origin overlay and reveal the dataset-coordinates toggle when an origin is available."""
        try:
            self._current_roi_origin = origin
            if hasattr(self, '_roi_axis_toggle') and origin is not None:
                self._roi_axis_toggle.setChecked(False)
                self._roi_axis_toggle.setVisible(True)
                self._roi_axis_toggle.raise_()
        except Exception:
            pass

    def _current_frame_index(self) -> int:
        """Current frame index from the spinbox, or 0 when frame navigation is inactive."""
        try:
            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                return int(self.frame_spinbox.value())
        except Exception:
            pass
        return 0

    def _on_edit_frame_annotation(self) -> None:
        """Open the annotation editor prefilled for the current frame."""
        try:
            dock = getattr(self.main_window, 'annotation_dock', None)
            if dock is not None:
                dock.open_for_frame(self._current_frame_index())
        except Exception:
            pass

    def _setup_2d_hover(self):
        """Create crosshair and tooltip overlays, and connect mouse move events via SignalProxy."""
        try:
            if not hasattr(self, 'image_view') or self.image_view is None:
                return
            view = self.image_view.getView() if hasattr(self.image_view, 'getView') else None
            if view is None:
                return
            # Create overlays only once
            if not hasattr(self, '_hover_hline') or self._hover_hline is None:
                try:
                    self._hover_hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=(255, 255, 0, 150), width=1))
                    self.plot_item.addItem(self._hover_hline)
                    try:
                        self._hover_hline.setZValue(1000)
                    except Exception:
                        pass
                except Exception:
                    self._hover_hline = None
            if not hasattr(self, '_hover_vline') or self._hover_vline is None:
                try:
                    self._hover_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(255, 255, 0, 150), width=1))
                    self.plot_item.addItem(self._hover_vline)
                    try:
                        self._hover_vline.setZValue(1000)
                    except Exception:
                        pass
                except Exception:
                    self._hover_vline = None
            if not hasattr(self, '_hover_text') or self._hover_text is None:
                try:
                    self._hover_text = pg.TextItem("", color=(255, 255, 255))
                    try:
                        self._hover_text.setAnchor((0, 1))
                    except Exception:
                        pass
                    self.plot_item.addItem(self._hover_text)
                    try:
                        self._hover_text.setZValue(1000)
                    except Exception:
                        pass
                except Exception:
                    self._hover_text = None
            # Connect mouse move via SignalProxy to throttle updates
            try:
                vb = getattr(self.plot_item, 'vb', None)
                scene = vb.scene() if vb is not None else self.plot_item.scene()
                self._mouse_proxy = pg.SignalProxy(scene.sigMouseMoved, rateLimit=60, slot=self._on_2d_mouse_moved)
            except Exception:
                self._mouse_proxy = None
        except Exception as e:
            try:
                self.main_window.update_status(f"Error setting up 2D hover: {e}")
            except Exception:
                pass

    def _on_2d_mouse_moved(self, evt):
        """Map scene coordinates to pixel indices; update crosshair and tooltip with intensity and HKL if available."""
        try:
            # evt may be (QPointF,) from SignalProxy
            pos = evt[0] if isinstance(evt, (tuple, list)) and len(evt) > 0 else evt
            if not hasattr(self, 'image_view') or self.image_view is None:
                return
            view = self.image_view.getView() if hasattr(self.image_view, 'getView') else None
            image_item = getattr(self.image_view, 'imageItem', None)
            if view is None or image_item is None:
                return
            # Map to data coordinates
            try:
                vb = getattr(self.plot_item, 'vb', None)
                if vb is not None:
                    mouse_point = vb.mapSceneToView(pos)
                else:
                    mouse_point = view.mapSceneToView(pos)
            except Exception:
                return
            # Respect hover enabled flag
            if not bool(getattr(self, '_hover_enabled', True)):
                return
            x = int(round(float(mouse_point.x())))
            y = int(round(float(mouse_point.y())))
            frame = self.get_current_frame_data()
            if frame is None or frame.ndim != 2:
                return
            height, width = frame.shape
            # Move crosshairs regardless, using float positions
            try:
                if hasattr(self, '_hover_hline') and self._hover_hline is not None:
                    self._hover_hline.setPos(mouse_point.y())
                if hasattr(self, '_hover_vline') and self._hover_vline is not None:
                    self._hover_vline.setPos(mouse_point.x())
            except Exception:
                pass
            if x < 0 or y < 0 or x >= width or y >= height:
                return
            # Remember last valid hover position
            try:
                self._last_hover_xy = (x, y)
            except Exception:
                pass
            # Intensity at pixel
            try:
                intensity = float(frame[x, y])
            except Exception:
                intensity = float('nan')
            # Update the 2D Info dock at this pixel
            self._update_hover_readout(x, y, intensity)
        except Exception:
            pass

    def _try_load_hkl_grids(self):
        """Load and cache qx/qy/qz grids (supports 2D HxW and 3D FxHxW). Called after display_2d_data."""
        try:
            # Reset caches by default
            self._qx_grid = None
            self._qy_grid = None
            self._qz_grid = None
            if not getattr(self.main_window, 'current_file_path', None) or not getattr(self.main_window, 'selected_dataset_path', None):
                return
            with h5py.File(self.main_window.current_file_path, 'r') as h5f:
                sel_path = str(self.main_window.selected_dataset_path)
                parent_path = sel_path.rsplit('/', 1)[0] if '/' in sel_path else '/'
                candidates = []
                try:
                    if parent_path in h5f:
                        candidates.append(h5f[parent_path])
                except Exception:
                    pass
                try:
                    if '/entry/data' in h5f:
                        candidates.append(h5f['/entry/data'])
                except Exception:
                    pass
                qx = qy = qz = None
                def find_in_group(g, name):
                    for key in g.keys():
                        try:
                            if isinstance(g[key], h5py.Dataset) and key.lower() == name:
                                return g[key]
                        except Exception:
                            pass
                    return None
                # Try strict names first
                for g in candidates:
                    if g is None:
                        continue
                    try:
                        qx = find_in_group(g, 'qx')
                        qy = find_in_group(g, 'qy')
                        qz = find_in_group(g, 'qz')
                    except Exception:
                        qx = qy = qz = None
                    if qx is not None and qy is not None and qz is not None:
                        break
                # Fallback: case-insensitive suffix match within parent group
                if (qx is None or qy is None or qz is None) and parent_path in h5f:
                    g = h5f[parent_path]
                    for key in g.keys():
                        try:
                            if not isinstance(g[key], h5py.Dataset):
                                continue
                        except Exception:
                            continue
                        lk = key.lower()
                        # Support additional naming conventions for HKL/q grids
                        if lk.endswith('qx') or lk == 'qx' or lk in ('q_x', 'qx_grid', 'qgrid_x', 'h', 'QX'.lower()):
                            qx = g[key]
                        elif lk.endswith('qy') or lk == 'qy' or lk in ('q_y', 'qy_grid', 'qgrid_y', 'k', 'QY'.lower()):
                            qy = g[key]
                        elif lk.endswith('qz') or lk == 'qz' or lk in ('q_z', 'qz_grid', 'qgrid_z', 'l', 'QZ'.lower()):
                            qz = g[key]
                # Last resort: search entire file for datasets named like qx/qy/qz
                if qx is None or qy is None or qz is None:
                    for group in [h5f]:
                        for key in group.keys():
                            try:
                                item = group[key]
                                if not isinstance(item, h5py.Dataset):
                                    continue
                                lk = key.lower()
                                if qx is None and (lk.endswith('qx') or lk == 'qx' or lk in ('q_x', 'h')):
                                    qx = item
                                elif qy is None and (lk.endswith('qy') or lk == 'qy' or lk in ('q_y', 'k')):
                                    qy = item
                                elif qz is None and (lk.endswith('qz') or lk == 'qz' or lk in ('q_z', 'l')):
                                    qz = item
                            except Exception:
                                continue
                if qx is None or qy is None or qz is None:
                    return
                # Read arrays
                try:
                    qx_arr = np.asarray(qx[...], dtype=np.float32)
                    qy_arr = np.asarray(qy[...], dtype=np.float32)
                    qz_arr = np.asarray(qz[...], dtype=np.float32)
                except Exception:
                    return
                frame = self.get_current_frame_data()
                if frame is None or frame.ndim != 2:
                    return
                h, w = frame.shape
                # Normalize shapes: transpose 2D grids if (w,h)
                if qx_arr.ndim == 2 and qy_arr.ndim == 2 and qz_arr.ndim == 2:
                    if qx_arr.shape == (w, h) and qy_arr.shape == (w, h) and qz_arr.shape == (w, h):
                        try:
                            qx_arr = qx_arr.T
                            qy_arr = qy_arr.T
                            qz_arr = qz_arr.T
                        except Exception:
                            pass
                    if qx_arr.shape == (h, w) and qy_arr.shape == (h, w) and qz_arr.shape == (h, w):
                        self._qx_grid = qx_arr
                        self._qy_grid = qy_arr
                        self._qz_grid = qz_arr
                    else:
                        return
                elif qx_arr.ndim == 3 and qy_arr.ndim == 3 and qz_arr.ndim == 3:
                    # Expect (F, H, W), but reorder axes if needed
                    def reorder_to_fhw(arr, h, w):
                        try:
                            shp = arr.shape
                            if len(shp) != 3:
                                return None
                            # Identify axes matching h and w
                            idx_h = None
                            idx_w = None
                            for i, d in enumerate(shp):
                                if d == h and idx_h is None:
                                    idx_h = i
                            for i, d in enumerate(shp):
                                if d == w and i != idx_h and idx_w is None:
                                    idx_w = i
                            if idx_h is None or idx_w is None:
                                return None
                            idx_f = [0, 1, 2]
                            idx_f.remove(idx_h)
                            idx_f.remove(idx_w)
                            idx_f = idx_f[0]
                            order = [idx_f, idx_h, idx_w]
                            return np.transpose(arr, axes=order)
                        except Exception:
                            return None
                    if not (qx_arr.shape[1:] == (h, w) and qy_arr.shape[1:] == (h, w) and qz_arr.shape[1:] == (h, w)):
                        rqx = reorder_to_fhw(qx_arr, h, w)
                        rqy = reorder_to_fhw(qy_arr, h, w)
                        rqz = reorder_to_fhw(qz_arr, h, w)
                        if rqx is not None and rqy is not None and rqz is not None:
                            qx_arr, qy_arr, qz_arr = rqx, rqy, rqz
                    if qx_arr.shape[1:] == (h, w) and qy_arr.shape[1:] == (h, w) and qz_arr.shape[1:] == (h, w):
                        self._qx_grid = qx_arr
                        self._qy_grid = qy_arr
                        self._qz_grid = qz_arr
                    else:
                        return
                else:
                    return
                try:
                    self.main_window.update_status("HKL q-grids loaded for hover")
                except Exception:
                    pass
                try:
                    self.set_2d_axes("h", "k")
                except Exception:
                    pass
                try:
                    if hasattr(self.main_window, 'info_2d_dock') and self.main_window.info_2d_dock is not None:
                        self.main_window.info_2d_dock.refresh()
                except Exception:
                    pass
        except Exception as e:
            try:
                self.main_window.update_status(f"HKL q-grids load failed: {e}")
            except Exception:
                pass

    def on_colormap_changed(self, colormap_name):
        """Handle colormap changes."""
        try:
            self.apply_colormap(colormap_name)
        except Exception as e:
            self.main_window.update_status(f"Error changing colormap: {e}")

    def on_auto_levels_toggled(self, enabled):
        """Handle auto levels toggle."""
        try:
            if hasattr(self, 'image_view') and hasattr(self.image_view, 'imageItem'):
                if enabled:
                    # Enable auto levels
                    self.image_view.autoLevels()
                    # Clear manual override when switching to Auto
                    try:
                        self._manual_levels_override = False
                        self._manual_vmin = None
                        self._manual_vmax = None
                        self._manual_vmin_display = None
                        self._manual_vmax_display = None
                    except Exception:
                        pass
                # If disabled, keep current levels
        except Exception as e:
            self.main_window.update_status(f"Error toggling auto levels: {e}")

    def apply_colormap(self, colormap_name):
        """Apply a colormap to the image view."""
        try:
            if not hasattr(self, 'image_view'):
                return

            lut = None
            # Try pyqtgraph ColorMap first
            try:
                if hasattr(pg, "colormap") and hasattr(pg.colormap, "get"):
                    try:
                        cmap = pg.colormap.get(colormap_name)
                    except Exception:
                        cmap = None
                    if cmap is not None:
                        lut = cmap.getLookupTable(nPts=256)
            except Exception:
                lut = None

            # Fallback to matplotlib if needed
            if lut is None:
                try:
                    import matplotlib.pyplot as plt
                    mpl_cmap = plt.get_cmap(colormap_name)
                    # Build LUT as uint8 Nx3
                    xs = np.linspace(0.0, 1.0, 256, dtype=float)
                    colors = mpl_cmap(xs, bytes=True)  # returns Nx4 uint8
                    lut = colors[:, :3]
                except Exception:
                    # Last resort: grayscale
                    xs = (np.linspace(0, 255, 256)).astype(np.uint8)
                    lut = np.column_stack([xs, xs, xs])

            # Apply the lookup table
            try:
                self.image_view.imageItem.setLookupTable(lut)
            except Exception:
                pass

        except Exception as e:
            self.main_window.update_status(f"Error applying colormap: {e}")

    def previous_frame(self):
        """Navigate to the previous frame."""
        try:
            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                current_frame = self.frame_spinbox.value()
                if current_frame > 0:
                    self.frame_spinbox.setValue(current_frame - 1)
        except Exception as e:
            self.main_window.update_status(f"Error navigating to previous frame: {e}")

    def next_frame(self):
        """Navigate to the next frame."""
        try:
            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                current_frame = self.frame_spinbox.value()
                max_frame = self.frame_spinbox.maximum()
                if current_frame < max_frame:
                    self.frame_spinbox.setValue(current_frame + 1)
        except Exception as e:
            self.main_window.update_status(f"Error navigating to next frame: {e}")

    def on_frame_spinbox_changed(self, frame_index):
        """Handle frame spinbox changes for 3D data navigation."""
        try:
            if not hasattr(self, 'current_2d_data') or self.current_2d_data is None:
                return

            if self.current_2d_data.ndim != 3:
                return

            # Get the selected frame
            if frame_index < 0 or frame_index >= self.current_2d_data.shape[0]:
                frame_index = 0

            frame_data = np.asarray(self.current_2d_data[frame_index], dtype=np.float32)

            # Update the image view with the new frame via centralized pipeline
            self._refresh_current_frame_image(frame_data)

            # Update frame info label and button states
            num_frames = self.current_2d_data.shape[0]
            print(f"[FRAME] on_frame_spinbox_changed: frame_index={frame_index}, num_frames={num_frames}")
            height, width = frame_data.shape
            if hasattr(self, 'frame_info_label'):
                self.frame_info_label.setText(f"Image Dimensions: {width}x{height} pixels (frame {frame_index} of {num_frames})")
            # Let the annotation dock surface this frame's note + per-frame overlays
            self._notify_annotation_frame_changed()
            # Update overlay text
            self.update_overlay_text(width, height, f"Frame {frame_index} of {num_frames}")

            # Update per-frame Vmin/Vmax controls and data-domain hint labels
            try:
                self._update_vmin_vmax_for_frame(frame_data)
            except Exception:
                pass

            # Update hover tooltip/crosshair at last position during playback
            try:
                xy = getattr(self, '_last_hover_xy', None)
                if xy and bool(getattr(self, '_hover_enabled', True)):
                    self._update_hover_text_at(int(xy[0]), int(xy[1]))
            except Exception:
                pass

            # Update HKL 3D plot if open
            try:
                self._update_hkl3d_plot_for_current_frame()
            except Exception:
                pass

            # Update button states
            if hasattr(self, 'btn_prev_frame'):
                self.btn_prev_frame.setEnabled(frame_index > 0)
            if hasattr(self, 'btn_next_frame'):
                self.btn_next_frame.setEnabled(frame_index < num_frames - 1)

            # Refresh ROI stats when frame changes
            try:
                self.main_window.roi_manager.update_all_roi_stats()
            except Exception:
                pass
            try:
                if hasattr(self.main_window, 'info_2d_dock') and self.main_window.info_2d_dock is not None:
                    self.main_window.info_2d_dock.refresh()
            except Exception:
                pass

        except Exception as e:
            self.main_window.update_status(f"Error changing frame: {e}")

    def _update_vmin_vmax_for_frame(self, frame_data):
        """Update Vmin/Vmax spinbox ranges (and values when not in manual override) per current frame.

        Also refreshes the data-domain hint labels (Vmin(...): / Vmax(...):) to reflect this frame's extrema.
        Respects log scale ranges and _manual_levels_override.
        """
        try:
            if frame_data is None:
                return
            # Compute integer-friendly extrema for the current frame
            try:
                min_val = int(np.min(frame_data))
                max_val = int(np.max(frame_data))
            except Exception:
                return

            manual_override = bool(getattr(self, '_manual_levels_override', False))

            # Use log-scale friendly ranges when enabled
            log_enabled = hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked()
            if log_enabled:
                try:
                    pos = frame_data[frame_data > 0]
                    min_pos = int(np.min(pos)) if pos.size > 0 else 1
                except Exception:
                    min_pos = 1
                vmin_floor = max(1, min_pos)
                vmax_cap = max_val

                # Update ranges and values guarded
                try:
                    self._suppress_spinbox_handlers = True
                    if hasattr(self, 'sbVmin') and self.sbVmin is not None:
                        self.sbVmin.setRange(1, vmax_cap)
                        if not manual_override:
                            self.sbVmin.setValue(vmin_floor)
                    if hasattr(self, 'sbVmax') and self.sbVmax is not None:
                        upper = max(vmax_cap * 2, int(getattr(self, '_manual_vmax', vmax_cap) or vmax_cap))
                        self.sbVmax.setRange(vmin_floor + 1, upper)
                        if not manual_override:
                            self.sbVmax.setValue(vmax_cap)
                finally:
                    try:
                        self._suppress_spinbox_handlers = False
                    except Exception:
                        pass
            else:
                # Linear scale: full data range
                try:
                    self._suppress_spinbox_handlers = True
                    if hasattr(self, 'sbVmin') and self.sbVmin is not None:
                        lower = min(min_val, int(getattr(self, '_manual_vmin', min_val) or min_val))
                        self.sbVmin.setRange(lower, max_val)
                        if not manual_override:
                            self.sbVmin.setValue(min_val)
                    if hasattr(self, 'sbVmax') and self.sbVmax is not None:
                        upper = max(max_val * 2, int(getattr(self, '_manual_vmax', max_val) or max_val))
                        self.sbVmax.setRange(min_val + 1, upper)
                        if not manual_override:
                            self.sbVmax.setValue(max_val)
                finally:
                    try:
                        self._suppress_spinbox_handlers = False
                    except Exception:
                        pass

            # Keep the hint labels in sync with current frame (data-domain)
            try:
                self._update_vmin_vmax_hints()
            except Exception:
                pass
        except Exception:
            pass

    def start_playback(self):
        """Start frame playback if a 3D stack is loaded and controls are enabled."""
        try:
            if not hasattr(self, 'current_2d_data') or self.current_2d_data is None:
                return
            if self.current_2d_data.ndim != 3:
                return
            # Only play if more than 1 frame
            num_frames = self.current_2d_data.shape[0]
            if num_frames <= 1:
                print(f"[PLAYBACK] start_playback: num_frames={num_frames} -> not enough frames to play")
                return
            # Set timer interval from FPS
            fps = 2
            try:
                if hasattr(self, 'sb_fps'):
                    fps = max(1, int(self.sb_fps.value()))
            except Exception:
                fps = 2
            interval_ms = int(1000 / max(1, fps))
            print(f"[PLAYBACK] start_playback: num_frames={num_frames}, fps={fps}, interval_ms={interval_ms}")
            # Reset frame index to 0 at playback start to avoid stale index from previous data
            try:
                if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                    self.frame_spinbox.setValue(0)
            except Exception:
                pass
            if hasattr(self, 'play_timer') and self.play_timer is not None:
                try:
                    self.play_timer.setInterval(interval_ms)
                    self.play_timer.start()
                    try:
                        print(f"[PLAYBACK] timer state: {'active' if self.play_timer.isActive() else 'inactive'}")
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[PLAYBACK] ERROR starting timer: {e}")
            # Update control states
            try:
                self.btn_play.setEnabled(False)
                self.btn_pause.setEnabled(True)
            except Exception:
                pass
            self.main_window.update_status("Playback started")
        except Exception as e:
            self.main_window.update_status(f"Error starting playback: {e}")

    def pause_playback(self):
        """Pause frame playback."""
        try:
            if hasattr(self, 'play_timer') and self.play_timer is not None:
                try:
                    self.play_timer.stop()
                except Exception:
                    pass
            try:
                self.btn_play.setEnabled(True)
                self.btn_pause.setEnabled(False)
            except Exception:
                pass
            self.main_window.update_status("Playback paused")
        except Exception as e:
            self.main_window.update_status(f"Error pausing playback: {e}")

    def on_fps_changed(self, value):
        """Update timer interval when FPS changes."""
        try:
            fps = max(1, int(value))
            interval_ms = int(1000 / fps)
            if hasattr(self, 'play_timer') and self.play_timer is not None:
                try:
                    self.play_timer.setInterval(interval_ms)
                except Exception:
                    pass
        except Exception as e:
            self.main_window.update_status(f"Error updating FPS: {e}")

    def _advance_frame_playback(self):
        """Advance one frame; handle auto replay at end."""
        try:
            if not hasattr(self, 'current_2d_data') or self.current_2d_data is None:
                return
            if self.current_2d_data.ndim != 3:
                return
            num_frames = self.current_2d_data.shape[0]
            if num_frames <= 1:
                print("[PLAYBACK] tick: num_frames<=1 -> pausing")
                self.pause_playback()
                return
            idx = 0
            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                try:
                    idx = int(self.frame_spinbox.value())
                except Exception:
                    idx = 0
            # Clamp idx to valid range for current data
            if idx < 0 or idx >= num_frames:
                idx = 0
            next_idx = idx + 1
            if next_idx >= num_frames:
                # Auto replay from beginning if checked
                auto = False
                try:
                    auto = bool(self.cb_auto_replay.isChecked()) if hasattr(self, 'cb_auto_replay') else False
                except Exception:
                    auto = False
                print(f"[PLAYBACK] tick: idx={idx}, next_idx={next_idx} reached end, auto_replay={auto}")
                if auto:
                    next_idx = 0
                else:
                    self.pause_playback()
                    return
            print(f"[PLAYBACK] tick: advancing to next_idx={next_idx} of num_frames={num_frames}")
            # Set via spinbox to reuse existing update logic
            if hasattr(self, 'frame_spinbox'):
                try:
                    self.frame_spinbox.setValue(next_idx)
                except Exception as e:
                    print(f"[PLAYBACK] ERROR setting frame_spinbox: {e}")
        except Exception:
            pass

    def on_log_scale_toggled(self, checked):
        """Handle log scale checkbox toggle."""
        try:
            if hasattr(self, 'image_view') and hasattr(self, 'current_2d_data') and self.current_2d_data is not None:
                # Centralized refresh applies transform and synchronizes histogram/levels
                self._refresh_current_frame_image()

                # Refresh ROI stats to reflect displayed image
                try:
                    self.main_window.roi_manager.update_all_roi_stats()
                except Exception:
                    pass

                self.main_window.update_status(f"Log scale {'enabled' if checked else 'disabled'}")
                # Data-domain hint labels are informational; update them (values remain data-domain)
                try:
                    self._update_vmin_vmax_hints()
                except Exception:
                    pass
            else:
                print("No image data available for log scale")
        except Exception as e:
            self.main_window.update_status(f"Error toggling log scale: {e}")

    def update_vmin_vmax_for_log_scale(self, data, log_scale_enabled):
        """Update vmin/vmax controls based on log scale state."""
        try:
            if log_scale_enabled:
                # For log scale, set reasonable ranges
                min_val = max(1, int(np.min(data[data > 0]))) if np.any(data > 0) else 1
                max_val = int(np.max(data))

                # Respect manual override: update ranges only, keep values
                if hasattr(self, 'sbVmin'):
                    self.sbVmin.setRange(1, max_val)
                    if not bool(getattr(self, '_manual_levels_override', False)):
                        self.sbVmin.setValue(min_val)

                if hasattr(self, 'sbVmax'):
                    upper = max(max_val * 2, int(getattr(self, '_manual_vmax', max_val)))
                    self.sbVmax.setRange(min_val + 1, upper)
                    if not bool(getattr(self, '_manual_levels_override', False)):
                        self.sbVmax.setValue(max_val)
            else:
                # For linear scale, use full data range
                min_val = int(np.min(data))
                max_val = int(np.max(data))

                if hasattr(self, 'sbVmin'):
                    lower = min(min_val, int(getattr(self, '_manual_vmin', min_val)))
                    self.sbVmin.setRange(lower, max_val)
                    if not bool(getattr(self, '_manual_levels_override', False)):
                        self.sbVmin.setValue(min_val)

                if hasattr(self, 'sbVmax'):
                    upper = max(max_val * 2, int(getattr(self, '_manual_vmax', max_val)))
                    self.sbVmax.setRange(min_val + 1, upper)
                    if not bool(getattr(self, '_manual_levels_override', False)):
                        self.sbVmax.setValue(max_val)
        except Exception as e:
            self.main_window.update_status(f"Error updating vmin/vmax controls: {e}")

    def on_vmin_changed(self, value):
        """Handle vmin spinbox value change."""
        try:
            # Skip handler during programmatic updates from histogram
            if bool(getattr(self, '_suppress_spinbox_handlers', False)):
                return
            if hasattr(self, 'image_view') and hasattr(self, 'current_2d_data') and self.current_2d_data is not None:
                # Get current vmax
                vmax = self.sbVmax.value() if hasattr(self, 'sbVmax') else 100

                # Ensure vmin < vmax
                if value >= vmax:
                    return

                # Apply log scale if enabled
                if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                    vmin_display = np.log1p(value)
                    vmax_display = np.log1p(vmax)
                else:
                    vmin_display = value
                    vmax_display = vmax

                # Update image levels
                self.image_view.setLevels(min=vmin_display, max=vmax_display)
                # Also sync the histogram widget to reflect the user-entered spinbox levels
                try:
                    hist = self._get_histogram_widget()
                    if hist is not None:
                        try:
                            self._suppress_histogram_update = True
                            hist.setLevels(vmin_display, vmax_display)
                        finally:
                            self._suppress_histogram_update = False
                except Exception:
                    pass
                # Mark manual override active and cache
                try:
                    self._manual_levels_override = True
                    self._manual_vmin = float(value)
                    self._manual_vmax = float(vmax)
                    # Update display-domain cache precisely
                    if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                        self._manual_vmin_display = float(vmin_display)
                        self._manual_vmax_display = float(vmax_display)
                    else:
                        self._manual_vmin_display = float(value)
                        self._manual_vmax_display = float(vmax)
                except Exception:
                    pass
                # Refresh ROI stats (based on displayed image)
                try:
                    self.main_window.roi_manager.update_all_roi_stats()
                except Exception:
                    pass
                self.main_window.update_status(f"Vmin set to: {value}")
        except Exception as e:
            self.main_window.update_status(f"Error changing vmin: {e}")

    def on_vmax_changed(self, value):
        """Handle vmax spinbox value change."""
        try:
            # Skip handler during programmatic updates from histogram
            if bool(getattr(self, '_suppress_spinbox_handlers', False)):
                return
            if hasattr(self, 'image_view') and hasattr(self, 'current_2d_data') and self.current_2d_data is not None:
                # Get current vmin
                vmin = self.sbVmin.value() if hasattr(self, 'sbVmin') else 0

                # Ensure vmax > vmin
                if value <= vmin:
                    return

                # Apply log scale if enabled
                if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                    vmin_display = np.log1p(vmin)
                    vmax_display = np.log1p(value)
                else:
                    vmin_display = vmin
                    vmax_display = value

                # Update image levels
                self.image_view.setLevels(min=vmin_display, max=vmax_display)
                # Also sync the histogram widget to reflect the user-entered spinbox levels
                try:
                    hist = self._get_histogram_widget()
                    if hist is not None:
                        try:
                            self._suppress_histogram_update = True
                            hist.setLevels(vmin_display, vmax_display)
                        finally:
                            self._suppress_histogram_update = False
                except Exception:
                    pass
                # Mark manual override active and cache
                try:
                    self._manual_levels_override = True
                    self._manual_vmin = float(vmin)
                    self._manual_vmax = float(value)
                    # Update display-domain cache precisely
                    if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                        self._manual_vmin_display = float(vmin_display)
                        self._manual_vmax_display = float(vmax_display)
                    else:
                        self._manual_vmin_display = float(vmin)
                        self._manual_vmax_display = float(value)
                except Exception:
                    pass
                # Refresh ROI stats (based on displayed image)
                try:
                    self.main_window.roi_manager.update_all_roi_stats()
                except Exception:
                    pass
                self.main_window.update_status(f"Vmax set to: {value}")
        except Exception as e:
            self.main_window.update_status(f"Error changing vmax: {e}")

    def on_draw_roi_clicked(self):
        """Handle Draw ROI button click (delegated to ROIManager)."""
        try:
            self.main_window.roi_manager.create_and_add_roi()
        except Exception as e:
            self.main_window.update_status(f"Error drawing ROI: {e}")

    def on_ref_frame_changed(self, value):
        """Handle reference frame spinbox value change."""
        try:
            # Update the current frame to match reference frame
            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                self.frame_spinbox.setValue(value)
            self.main_window.update_status(f"Reference frame set to: {value}")
        except Exception as e:
            self.main_window.update_status(f"Error changing reference frame: {e}")

    def on_other_frame_changed(self, value):
        """Handle other frame spinbox value change."""
        try:
            self.main_window.update_status(f"Other frame set to: {value}")
        except Exception as e:
            self.main_window.update_status(f"Error changing other frame: {e}")

    def update_frame_controls_for_2d_data(self):
        """Update frame controls for 2D data (disable frame navigation)."""
        try:
            if hasattr(self, 'frame_spinbox'):
                self.frame_spinbox.setEnabled(False)
                self.frame_spinbox.setValue(0)
                self.frame_spinbox.setMaximum(0)

            if hasattr(self, 'btn_prev_frame'):
                self.btn_prev_frame.setEnabled(False)
            if hasattr(self, 'btn_next_frame'):
                self.btn_next_frame.setEnabled(False)

            # Stop playback timer and disable controls
            try:
                if hasattr(self, 'play_timer') and self.play_timer is not None:
                    self.play_timer.stop()
                if hasattr(self, 'btn_play'):
                    self.btn_play.setEnabled(False)
                if hasattr(self, 'btn_pause'):
                    self.btn_pause.setEnabled(False)
                if hasattr(self, 'sb_fps'):
                    self.sb_fps.setEnabled(False)
                if hasattr(self, 'cb_auto_replay'):
                    self.cb_auto_replay.setEnabled(False)
            except Exception:
                pass
        except Exception as e:
            self.main_window.update_status(f"Error updating frame controls for 2D data: {e}")

    def update_frame_controls_for_3d_data(self, num_frames):
        """Update frame controls for 3D data (enable frame navigation)."""
        try:
            if hasattr(self, 'frame_spinbox'):
                self.frame_spinbox.setEnabled(True)
                self.frame_spinbox.setMaximum(num_frames - 1)
                self.frame_spinbox.setValue(0)

            if hasattr(self, 'btn_prev_frame'):
                self.btn_prev_frame.setEnabled(False)  # Disabled for frame 0
            if hasattr(self, 'btn_next_frame'):
                self.btn_next_frame.setEnabled(num_frames > 1)

            # Enable or disable playback controls based on frame count
            try:
                enable_playback = num_frames > 1
                if hasattr(self, 'btn_play'):
                    self.btn_play.setEnabled(enable_playback)
                if hasattr(self, 'btn_pause'):
                    self.btn_pause.setEnabled(False)  # initially paused
                if hasattr(self, 'sb_fps'):
                    self.sb_fps.setEnabled(enable_playback)
                if hasattr(self, 'cb_auto_replay'):
                    self.cb_auto_replay.setEnabled(enable_playback)
                    try:
                        # Select auto replay when playback becomes available
                        self.cb_auto_replay.setChecked(True)
                    except Exception:
                        pass
                # Stop timer on reconfigure
                if hasattr(self, 'play_timer') and self.play_timer is not None:
                    self.play_timer.stop()
            except Exception:
                pass

            # Update data-domain hint labels for Vmin/Vmax for the new frame
            try:
                self._update_vmin_vmax_hints()
            except Exception:
                pass
        except Exception as e:
            self.main_window.update_status(f"Error updating frame controls for 3D data: {e}")

    def update_speckle_controls_for_data(self, data):
        """Update speckle analysis controls based on loaded data."""
        try:
            if data.ndim == 3:
                # 3D data - enable frame selection
                max_frame = data.shape[0] - 1

                if hasattr(self, 'sbRefFrame'):
                    self.sbRefFrame.setMaximum(max_frame)
                    self.sbRefFrame.setValue(0)
                    self.sbRefFrame.setEnabled(True)

                if hasattr(self, 'sbOtherFrame'):
                    self.sbOtherFrame.setMaximum(max_frame)
                    self.sbOtherFrame.setValue(min(1, max_frame))
                    self.sbOtherFrame.setEnabled(True)
            else:
                # 2D data - disable frame selection
                if hasattr(self, 'sbRefFrame'):
                    self.sbRefFrame.setValue(0)
                    self.sbRefFrame.setMaximum(0)
                    self.sbRefFrame.setEnabled(False)

                if hasattr(self, 'sbOtherFrame'):
                    self.sbOtherFrame.setValue(0)
                    self.sbOtherFrame.setMaximum(0)
                    self.sbOtherFrame.setEnabled(False)

        except Exception as e:
            self.main_window.update_status(f"Error updating speckle controls: {e}")

    def update_vmin_vmax_controls_for_data(self, data):
        """Update vmin/vmax controls with awareness of current scale mode (log vs linear).

        Delegates to update_vmin_vmax_for_log_scale() using the current cbLogScale state,
        and then refreshes the data-domain hint labels.
        """
        try:
            log_enabled = hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked()
            self.update_vmin_vmax_for_log_scale(data, log_enabled)
            # Keep hint labels in sync with the current data frame (data-domain)
            try:
                self._update_vmin_vmax_hints()
            except Exception:
                pass
        except Exception as e:
            self.main_window.update_status(f"Error updating vmin/vmax controls: {e}")

    def _update_vmin_vmax_hints(self):
        """Update the informational Vmin/Vmax hint labels based on the current frame, in data domain.

        Creates integer-friendly labels like "Vmin(123):" and "Vmax(456):".
        If no frame or no finite values exist, uses placeholders "Vmin(...):" and "Vmax(...):".
        Labels are informational only and do not affect spinboxes or histogram levels.
        """
        try:
            # Ensure labels exist
            if not hasattr(self, 'lblVminHint') or not hasattr(self, 'lblVmaxHint'):
                return
            lbl_min = getattr(self, 'lblVminHint', None)
            lbl_max = getattr(self, 'lblVmaxHint', None)
            if lbl_min is None and lbl_max is None:
                return

            frame = None
            try:
                frame = self.get_current_frame_data()
            except Exception:
                frame = None

            vmin_txt = "Vmin(...):"
            vmax_txt = "Vmax(...):"
            try:
                if frame is not None:
                    arr = np.asarray(frame)
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        dmin = int(np.min(finite))
                        dmax = int(np.max(finite))
                        vmin_txt = f"Vmin({dmin}):"
                        vmax_txt = f"Vmax({dmax}):"
            except Exception:
                # Keep placeholders on error
                pass

            try:
                if lbl_min is not None:
                    lbl_min.setText(vmin_txt)
            except Exception:
                pass
            try:
                if lbl_max is not None:
                    lbl_max.setText(vmax_txt)
            except Exception:
                pass
        except Exception:
            # Silent guard
            pass

    def get_current_frame_data(self):
        try:
            if not hasattr(self, 'current_2d_data') or self.current_2d_data is None:
                return None
            if self.current_2d_data.ndim == 3:
                frame_index = 0
                if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                    frame_index = self.frame_spinbox.value()
                if frame_index < 0 or frame_index >= self.current_2d_data.shape[0]:
                    frame_index = 0
                return np.asarray(self.current_2d_data[frame_index], dtype=np.float32)
            else:
                return np.asarray(self.current_2d_data, dtype=np.float32)
        except Exception:
            return None
