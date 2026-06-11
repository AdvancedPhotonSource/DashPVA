import os
import subprocess
import sys
import time

import numpy as np
import pyqtgraph as pg
import xrayutilities as xu
from epics import PV, caget, camonitor
from PyQt5 import uic
from PyQt5.QtCore import QByteArray, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSlider,
)
from pyqtgraph.colormap import get as get_colormap

# Custom imported classes
from dashpva.gui import configure_app, ui_path
from dashpva.gui.theme_colors import ROI_COLORS
from dashpva.utils import HDF5Writer, PVAReader, rotation_cycle
from dashpva.utils.mask_manager import MaskManager
from dashpva.viewer.area_det.docks import (
    AnalysisDock,
    ImageDock,
    MaskDock,
    MousePosDock,
    RoiDock,
    StatsDock,
)
from dashpva.viewer.core.base_window import BaseWindow
from dashpva.viewer.mask_viewer import MaskViewerWindow
from dashpva.viewer.roi_stats_dialog import RoiStatsDialog
from dashpva.viewer.roi_stats_plot import RoiStatsPlotDialog

rot_gen = rotation_cycle(1,5)

_PERF_TIMER_INTERVAL_MS = 1000
# Bump when the dock set changes — restoreState silently rejects mismatched
# versions so users with a stale saved layout fall back to defaults instead
# of getting a half-broken arrangement.
_DOCK_STATE_VERSION = 1


def _settings() -> QSettings:
    """QSettings handle for area-detector viewer state.

    Stored per-user via Qt's native backend (macOS plist, Linux ~/.config,
    Windows registry) — no hidden file in the repo root.
    """
    return QSettings("DashPVA", "Viewer")


class ConfigDialog(QDialog):

    def __init__(self):
        super(ConfigDialog, self).__init__()
        uic.loadUi(ui_path('pv_config.ui'), self)
        self.setWindowTitle('PV Config')
        self.prefix = ''
        self.input_channel = ''
        self.init_ui()
        self.btn_accept_reject.accepted.connect(self.dialog_accepted)

    def init_ui(self) -> None:
        self.lbl_input_channel.setText("Detector Prefix")
        self.le_input_channel.setPlaceholderText("e.g. s6lambda1")
        last = _settings().value("area_det_prefix", "", type=str)
        if last:
            self.le_input_channel.setText(last)

    def dialog_accepted(self) -> None:
        self.prefix = self.le_input_channel.text().strip()
        _settings().setValue("area_det_prefix", self.prefix)
        self.input_channel = f"{self.prefix}:Pva1:Image" if self.prefix else "pvapy:image"
        self.image_viewer = DiffractionImageWindow(input_channel=self.input_channel)


class DiffractionImageWindow(BaseWindow):
    hkl_data_updated = pyqtSignal(bool)
    # Emitted from the ROI/Stats connection thread so update_status can run on
    # the main (GUI) thread. Args: message, level ('info' | 'warning' | 'error').
    pv_pollers_status = pyqtSignal(str, str)
    # Emitted after start_roi_backup_monitor populates self.reader.rois so
    # add_rois() runs on the main thread (pg.ROI must be created in the GUI
    # thread).  Decouples the async caget sweep from rectangle creation.
    rois_ready = pyqtSignal()

    def __init__(self, input_channel='pvapy:image'):
        super().__init__(ui_file_name='imageshow.ui',
                         viewer_name='AreaDetector2D',
                         visible_actions=['Windows', 'Documentation'])
        self.setWindowTitle('DashPVA')
        saved_geom = _settings().value("area_det_window_geom", QByteArray(), type=QByteArray)
        if not saved_geom.isEmpty():
            try:
                self.restoreGeometry(saved_geom)
                avail = self.screen().availableGeometry()
                if self.width() > avail.width() or self.height() > avail.height():
                    self.resize(min(self.width(), avail.width()), min(self.height(), avail.height()))
            except Exception:
                pass
        self.show()
        self.reader = None
        self.image = None
        self.call_id_plot = 0
        self.first_plot = True
        self.image_is_transposed = False
        # True while _connect_pv_pollers is sweeping ROI/Stats PVs in its
        # background thread. Read-only flag for any code that wants to know if
        # the initial PV connection sweep has finished.
        self._pv_pollers_loading = False
        # Indeterminate spinner + label in the status bar, only visible while
        # the PV poller thread is running. Driven by _on_pv_pollers_status.
        self._build_pv_pollers_indicator()
        self.rot_num = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.rois: list[pg.ROI] = []
        self.stats_dialogs = {}
        self.stats_plot_dialogs = {}
        self.stats_data = {}
        self._input_channel = input_channel
        self.pv_prefix.setText(self._input_channel)
        self.pv_prefix.setPlaceholderText("e.g. s6lambda1:Pva1:Image")

        # Mask management
        self.mask_manager = MaskManager()
        self.mask_viewer = None
        self._dead_px_frames = []
        self._dead_px_collecting = False
        self._dead_px_target = 50
        self._dead_px_last_frame = 0
        self._dead_px_mode = 'illuminated'

        # Initializing but not starting timers so they can be reached by different functions
        self.timer_labels = QTimer()
        self.timer_plot = QTimer()
        self.file_writer_thread = QThread()
        self.timer_labels.timeout.connect(self.update_labels)
        self.timer_plot.timeout.connect(self.update_image)
        self.timer_plot.timeout.connect(self.update_rois)

        # HKL values
        self.is_hkl_ready = False
        self.hkl_config = None
        self.hkl_pvs = {}
        self.hkl_data = {}
        self.q_conv = None
        self.qx = None
        self.qy = None
        self.qz = None
        self.sample_circle_directions = []
        self.sample_circle_positions = []
        self.det_circle_directions = []
        self.det_circle_positions = []
        self.primary_beam_directions = []
        self.inplane_reference_directions = []
        self.sample_surface_normal_directions = []
        self.ub_matrix = None
        self.energy = None
        self.processes = {}
        
        # Initialize colormap once for better performance
        try:
            self.cet_colormap = get_colormap('viridis')
        except Exception:
            try:
                self.cet_colormap = get_colormap('viridis', source='colorcet')
            except Exception:
                self.cet_colormap = get_colormap('magma')  # fallback
        
        plot = pg.PlotItem()
        self.image_view = pg.ImageView(view=plot)
        # Set the default colormap when ImageView is created
        self.image_view.setColorMap(self.cet_colormap)
        # pyqtgraph turns every colormap stop into a draggable triangle on the
        # HistogramLUT gradient — viridis ships with ~256 stops, hence the
        # forest of arrows. Load the built-in preset which has only a handful.
        try:
            self.image_view.ui.histogram.gradient.loadPreset('viridis')
        except Exception:
            pass
        # Remove ImageView's internal buttons to reduce vertical margin mismatch
        self.image_view.ui.roiBtn.setParent(None)
        self.image_view.ui.menuBtn.setParent(None)
        self.viewer_layout.addWidget(self.image_view,0,1)
        self.image_view.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='SizeX [pixels]')

        # Initialize crosshair lines (hidden initially)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('red', width=2))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('red', width=2))
        self.image_view.addItem(self.crosshair_v, ignoreBounds=True)
        self.image_view.addItem(self.crosshair_h, ignoreBounds=True)
        self.crosshair_v.hide()
        self.crosshair_h.hide()
        self.crosshair_visible = False
        # second is a separate plot to show the horizontal avg of peaks in the image
        self.horizontal_avg_plot = pg.PlotWidget()
        self.horizontal_avg_plot.invertY(True)
        # Cap width by font metrics so the side plot stays narrow regardless
        # of DPI / system font size — pg.PlotWidget's default sizeHint is
        # generous, so QSizePolicy alone won't constrain it.
        self.horizontal_avg_plot.setMaximumWidth(self.fontMetrics().averageCharWidth() * 25)
        self.horizontal_avg_plot.getAxis('bottom').setLabel(text='Horizontal Avg.')
        self.horizontal_avg_plot.hideAxis('left')
        self.viewer_layout.addWidget(self.horizontal_avg_plot, 0,0)
        # Sync horizontal avg y-range to image view — use sigYRangeChanged
        # with padding=0 for precise alignment
        self.horizontal_avg_plot.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        self.image_view.getView().getViewBox().sigYRangeChanged.connect(self._sync_havg_yrange)

        # Build side panels as dock widgets and alias their members onto self
        self._setup_docks()

        # Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self._analysis_menu = QMenu(self)
        self._analysis_menu.addAction("pyFAI 1D Reduction", self._launch_pyfai)
        self._analysis_menu.addAction("XRD Phase Fitter", self._launch_phase_fitter)
        self._analysis_menu.addAction("HKL 3D Viewer", self._launch_hkl3d)
        self._analysis_menu.addAction("2D Scan Visualization", self._launch_scan_view)
        self.btn_analysis_window.setMenu(self._analysis_menu)
        for i in range(1, 6):
            getattr(self, f"btn_Stats{i}").clicked.connect(self.stats_button_clicked)
            getattr(self, f"btn_PlotStats{i}").clicked.connect(self.stats_plot_button_clicked)
        self.rbtn_C.clicked.connect(self.c_ordering_clicked)
        self.rbtn_F.clicked.connect(self.f_ordering_clicked)
        self.rotate90degCCW.clicked.connect(self.rotation_count)
        self.log_image.clicked.connect(self.update_image)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.display_rois.stateChanged.connect(self.show_rois_checked)
        self.chk_transpose.stateChanged.connect(self.transpose_image_checked)
        self.plotting_frequency.valueChanged.connect(self.start_timers)
        self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.min_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.chk_autoscale.stateChanged.connect(self.autoscale_checked)
        self.chk_threshold.stateChanged.connect(self.threshold_checked)
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)
        self.image_view.getView().mouseDoubleClickEvent = self.on_double_click
        self.hkl_data_updated.connect(self.handle_hkl_data_update)
        # Status messages from the ROI/Stats connection thread → main-thread label
        self.pv_pollers_status.connect(self._on_pv_pollers_status)
        # Background sweep finished populating reader.rois → build rectangles on GUI thread
        self.rois_ready.connect(self.add_rois)
        
        self.chk_autoscale.setChecked(True)
        self.chk_threshold.setChecked(False)
        
        # Initialize threshold label - show it since threshold is checked by default
        self.update_threshold_label()
        self.lbl_threshold_range.show()
        
        # Sync any post-init label state for the dock-mounted mask widgets
        self._update_mask_labels()

    def _setup_docks(self):
        """Build side panels as dock widgets and alias their members onto self.

        After this runs, code elsewhere can keep referring to widgets like
        self.frames_received_val or self.btn_Stats1 — they now resolve to the
        dock-owned widget instead of the original UI element.
        """
        # Drop the legacy sidebar from the central layout so the live-view
        # image fills the full width and there's no gap before the docks.
        if hasattr(self, 'scrollArea'):
            self.central_glayout.removeWidget(self.scrollArea)
            self.scrollArea.setParent(None)
            self.scrollArea.deleteLater()
        if hasattr(self, 'QGroupBox_live_view'):
            self.central_glayout.removeWidget(self.QGroupBox_live_view)
            self.central_glayout.addWidget(self.QGroupBox_live_view, 0, 0, 1, 2)

        # BaseDock.setup() auto-adds each dock into RightDockWidgetArea.
        # Default visible layout:
        #     [ Stats | Mask ]            (tabified, Stats raised)
        #     [ Image | Mouse Position ]  (tabified, Image raised)
        # ROI / Analysis start hidden — toggle from the Windows menu.
        self.stats_dock     = StatsDock(main_window=self)
        self.mask_dock      = MaskDock(main_window=self)
        self.image_dock     = ImageDock(main_window=self)
        self.mouse_pos_dock = MousePosDock(main_window=self)
        self.roi_dock       = RoiDock(main_window=self, show=False)
        self.analysis_dock  = AnalysisDock(main_window=self, show=False)

        self._apply_default_layout()
        # Restore the user's last layout if one was saved; falls through to
        # the defaults applied above on any failure.
        saved_state = _settings().value("area_det_dock_state", QByteArray(), type=QByteArray)
        if not saved_state.isEmpty():
            try:
                self.restoreState(saved_state, _DOCK_STATE_VERSION)
            except Exception:
                pass

        # Mask dock widgets
        self.lbl_mask_info        = self.mask_dock.lbl_mask_info
        self.lbl_mask_pixel_count = self.mask_dock.lbl_mask_pixel_count
        self.btn_load_mask        = self.mask_dock.btn_load_mask
        self.btn_show_mask        = self.mask_dock.btn_show_mask
        self.btn_detect_dead      = self.mask_dock.btn_detect_dead
        self.btn_clear_mask       = self.mask_dock.btn_clear_mask
        self.btn_export_json      = self.mask_dock.btn_export_json
        self.chk_apply_mask       = self.mask_dock.chk_apply_mask

        # Stats dock widgets
        self.frames_received_val = self.stats_dock.frames_received_val
        self.missed_frames_val   = self.stats_dock.missed_frames_val
        self.max_px_val          = self.stats_dock.max_px_val
        self.min_px_val          = self.stats_dock.min_px_val
        self.data_type_val       = self.stats_dock.data_type_val
        self.min_setting_val     = self.stats_dock.min_setting_val
        self.max_setting_val     = self.stats_dock.max_setting_val
        self.chk_autoscale       = self.stats_dock.chk_autoscale
        self.chk_threshold       = self.stats_dock.chk_threshold
        self.lbl_threshold_range = self.stats_dock.lbl_threshold_range

        # Mouse position dock widgets
        self.mouse_x_val  = self.mouse_pos_dock.mouse_x_val
        self.mouse_y_val  = self.mouse_pos_dock.mouse_y_val
        self.mouse_px_val = self.mouse_pos_dock.mouse_px_val
        self.mouse_h      = self.mouse_pos_dock.mouse_h
        self.mouse_k      = self.mouse_pos_dock.mouse_k
        self.mouse_l      = self.mouse_pos_dock.mouse_l

        # Image dock widgets
        self.plot_call_id       = self.image_dock.plot_call_id
        self.plotting_frequency = self.image_dock.plotting_frequency
        self.size_x_val         = self.image_dock.size_x_val
        self.size_y_val         = self.image_dock.size_y_val
        self.log_image          = self.image_dock.log_image
        self.freeze_image       = self.image_dock.freeze_image
        self.chk_transpose      = self.image_dock.chk_transpose
        self.display_rois       = self.image_dock.display_rois
        self.rbtn_C             = self.image_dock.rbtn_C
        self.rbtn_F             = self.image_dock.rbtn_F
        self.rotate90degCCW     = self.image_dock.rotate90degCCW
        self.stop_hkl           = self.image_dock.stop_hkl

        # ROI dock widgets
        for i in range(1, 5):
            setattr(self, f"lbl_ROI{i}", getattr(self.roi_dock, f"lbl_ROI{i}"))
            setattr(self, f"roi{i}_total_value", getattr(self.roi_dock, f"roi{i}_total_value"))
        self.lbl_image_total     = self.roi_dock.lbl_image_total
        self.stats5_total_value  = self.roi_dock.stats5_total_value
        self.chk_show_roi = [
            self.roi_dock.chk_show_roi1,
            self.roi_dock.chk_show_roi2,
            self.roi_dock.chk_show_roi3,
            self.roi_dock.chk_show_roi4,
        ]
        for chk in self.chk_show_roi:
            chk.stateChanged.connect(self._apply_roi_visibility)
        for i in range(1, 6):
            setattr(self, f"btn_Stats{i}", getattr(self.roi_dock, f"btn_Stats{i}"))
            setattr(self, f"btn_PlotStats{i}", getattr(self.roi_dock, f"btn_PlotStats{i}"))

        # Analysis dock widgets
        self.btn_analysis_window = self.analysis_dock.btn_analysis_window

        # Wire mask button signals (other docks are wired in __init__'s connection block)
        self.btn_load_mask.clicked.connect(self.load_mask_clicked)
        self.btn_show_mask.clicked.connect(self.show_mask_clicked)
        self.btn_detect_dead.clicked.connect(self.detect_dead_pixels_clicked)
        self.btn_clear_mask.clicked.connect(self.clear_mask_clicked)
        self.btn_export_json.clicked.connect(self.export_json_clicked)

        reset_action = QAction("Reset Dock Layout", self)
        reset_action.triggered.connect(self._reset_layout)
        self.add_windows_menu_action(reset_action, segment_name="other")

    def _apply_default_layout(self) -> None:
        self.splitDockWidget(self.stats_dock, self.image_dock, Qt.Vertical)
        self.tabifyDockWidget(self.stats_dock, self.mask_dock)
        self.tabifyDockWidget(self.image_dock, self.mouse_pos_dock)
        self.stats_dock.raise_()
        self.image_dock.raise_()
        dock_width = min(self.stats_dock.sizeHint().width(), self.width() // 3)
        dock_height = min(self.stats_dock.sizeHint().height(), self.height() // 2)
        self.resizeDocks([self.stats_dock], [dock_width], Qt.Horizontal)
        self.resizeDocks([self.stats_dock], [dock_height], Qt.Vertical)
        self.roi_dock.hide()
        self.analysis_dock.hide()

    def _reset_layout(self) -> None:
        _settings().remove("area_det_dock_state")
        geom = self.saveGeometry()
        for d in (self.stats_dock, self.mask_dock, self.image_dock,
                  self.mouse_pos_dock, self.roi_dock, self.analysis_dock):
            self.removeDockWidget(d)
            d.setFloating(False)
            self.addDockWidget(Qt.RightDockWidgetArea, d)
            d.show()
        self._apply_default_layout()
        self.restoreGeometry(geom)
        avail = self.screen().availableGeometry()
        if self.width() > avail.width() or self.height() > avail.height():
            self.resize(min(self.width(), avail.width()), min(self.height(), avail.height()))

    # ---- Perf status bar override ----
    # The area-detector viewer doesn't use the GPU, so skip BaseWindow's GPU
    # label and use a cross-platform CPU reader (psutil if available, /proc/stat
    # on Linux, otherwise "N/A") instead of BaseWindow's Linux-only /proc/stat.

    def init_perf_statusbar(self):
        sb = self.statusBar()
        self._cpu_label = QLabel("CPU: -%")
        self._runtime_label = QLabel("Runtime: 0s")
        sb.addPermanentWidget(self._cpu_label)
        sb.addPermanentWidget(self._runtime_label)
        self._start_time = time.monotonic()
        self._cpu_prev = None
        try:
            import psutil  # noqa: F401
            self._psutil = psutil
            self._psutil.cpu_percent(interval=None)
        except ImportError:
            self._psutil = None
        self._perf_timer = QTimer(self)
        self._perf_timer.setInterval(_PERF_TIMER_INTERVAL_MS)
        self._perf_timer.timeout.connect(self._update_perf_labels)
        self._perf_timer.start()

    def _update_perf_labels(self):
        if self._psutil is not None:
            self._cpu_label.setText(f"CPU: {self._psutil.cpu_percent(interval=None):.0f}%")
        else:
            try:
                with open("/proc/stat", "r") as f:
                    parts = f.readline().split()
                vals = list(map(int, parts[1:]))
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                total = sum(vals[:8]) if len(vals) >= 8 else sum(vals)
                if self._cpu_prev is not None:
                    ptotal, pidle = self._cpu_prev
                    dt = total - ptotal
                    didle = idle - pidle
                    if dt > 0:
                        self._cpu_label.setText(f"CPU: {(dt - didle) * 100.0 / dt:.0f}%")
                self._cpu_prev = (total, idle)
            except OSError:
                self._cpu_label.setText("CPU: N/A")
        self._runtime_label.setText(f"Runtime: {int(time.monotonic() - self._start_time)}s")

    # ---- Mask handler methods ----

    def load_mask_clicked(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Load Mask File', '',
            'Mask files (*.edf *.npy *.tif *.tiff *.json);;'
            'EDF files (*.edf);;NumPy files (*.npy);;'
            'TIFF files (*.tif *.tiff);;'
            'JSON BadPixel (*.json);;All files (*)')
        if not filepath:
            return

        try:
            det_shape = None
            if filepath.lower().endswith('.json'):
                if self.reader is not None and hasattr(self.reader, 'shape') and len(self.reader.shape) >= 2:
                    det_shape = self.reader.shape[:2]
                elif self.mask_manager.mask is not None:
                    det_shape = self.mask_manager.mask.shape
                # If det_shape is still None, _load_json_mask will try
                # to read "Detector size" from the JSON file itself.
            new_mask = self.mask_manager.load_mask(filepath, detector_shape=det_shape)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load mask:\n{e}')
            return

        # Ask whether to add or replace
        replace = True
        if self.mask_manager.mask is not None:
            reply = QMessageBox.question(
                self, 'Combine Mask',
                'A mask is already loaded.\n\n'
                'Click Yes to ADD (OR) with existing mask.\n'
                'Click No to REPLACE the existing mask.',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return
            replace = (reply == QMessageBox.No)

        # Warn if mask shape doesn't match current image
        if self.reader is not None and hasattr(self.reader, 'shape') and len(self.reader.shape) >= 2:
            img_shape = self.reader.shape[:2]
            if new_mask.shape != img_shape:
                QMessageBox.warning(
                    self, 'Shape Mismatch',
                    f'Mask shape {new_mask.shape} does not match '
                    f'image shape {img_shape}.\n\n'
                    f'The mask will be resized automatically, but this '
                    f'may indicate a configuration issue.')

        self.mask_manager.combine_masks(new_mask, replace=replace)
        self.mask_manager.mask_sources.append(filepath)
        self.mask_manager.save_active_mask()
        self._update_mask_labels()

    def show_mask_clicked(self):
        if self.mask_manager.mask is None:
            QMessageBox.information(self, 'No Mask', 'No mask is loaded.')
            return
        if self.mask_viewer is not None:
            self.mask_viewer.close()
        self.mask_viewer = MaskViewerWindow(
            mask=self.mask_manager.mask,
            mask_path=self.mask_manager.mask_path,
            parent=self)
        self.mask_viewer.mask_updated.connect(self._on_mask_edited)
        self.mask_viewer.show()

    def detect_dead_pixels_clicked(self):
        if self.reader is None:
            QMessageBox.warning(self, 'No Reader', 'Start live view first.')
            return
        if self._dead_px_collecting:
            QMessageBox.information(self, 'In Progress',
                                    'Dead pixel detection is already collecting frames.')
            return

        # Ask user which detection mode to use
        msg = QMessageBox(self)
        msg.setWindowTitle('Dead/Hot Pixel Detection')
        msg.setText('Select detection mode:')
        msg.setInformativeText(
            'Illuminated: mask pixels with zero variance\n'
            '(use with X-ray beam on, varying scattering)\n\n'
            'Dark: mask persistently bright pixels\n'
            '(use with beam off, rejects cosmic rays via median)')
        btn_light = msg.addButton('Illuminated', QMessageBox.AcceptRole)
        btn_dark = msg.addButton('Dark', QMessageBox.AcceptRole)
        msg.addButton(QMessageBox.Cancel)
        msg.exec_()

        clicked = msg.clickedButton()
        if clicked == btn_light:
            self._dead_px_mode = 'illuminated'
        elif clicked == btn_dark:
            self._dead_px_mode = 'dark'
        else:
            return

        self._dead_px_frames = []
        self._dead_px_collecting = True
        self._dead_px_last_frame = getattr(self.reader, 'frames_received', 0)
        self.btn_detect_dead.setText(f'Collecting 0/{self._dead_px_target}...')
        self.btn_detect_dead.setEnabled(False)

    def _collect_dead_pixel_frame(self):
        """Called from update_image to accumulate frames for dead pixel detection.
        Only collects on new PVA frames to avoid duplicate data."""
        if not self._dead_px_collecting or self.reader is None:
            return
        if self.reader.image is None or len(self.reader.shape) < 2:
            return
        # Only collect on new frames (avoid duplicates from timer ticks)
        current_frames = getattr(self.reader, 'frames_received', 0)
        if current_frames <= self._dead_px_last_frame:
            return
        self._dead_px_last_frame = current_frames

        self._dead_px_frames.append(self.reader.image.copy())
        self.btn_detect_dead.setText(
            f'Collecting {len(self._dead_px_frames)}/{self._dead_px_target}...')

        if len(self._dead_px_frames) >= self._dead_px_target:
            self._dead_px_collecting = False
            mode = getattr(self, '_dead_px_mode', 'illuminated')

            if mode == 'dark':
                result_mask = self.mask_manager.detect_hot_pixels(
                    self._dead_px_frames, sigma=5.0)
                label = 'hot'
            else:
                result_mask = self.mask_manager.detect_dead_pixels(
                    self._dead_px_frames, variance_threshold=1.0)
                label = 'stuck'

            self._dead_px_frames = []
            self.btn_detect_dead.setText('Detect Dead Px')
            self.btn_detect_dead.setEnabled(True)

            if result_mask is not None:
                num_flagged = int(np.sum(result_mask))
                if num_flagged == result_mask.size:
                    QMessageBox.warning(
                        self, 'Dead Pixel Detection',
                        f'All {num_flagged} pixels flagged.\n'
                        f'This usually means the detection mode does not match '
                        f'the current imaging conditions.\n'
                        f'Mask NOT updated.')
                    return
                self.mask_manager.combine_masks(result_mask)
                self.mask_manager.save_active_mask()
                self._update_mask_labels()
                # Refresh mask viewer if open
                if self.mask_viewer is not None and self.mask_viewer.isVisible():
                    self.mask_viewer.mask = self.mask_manager.mask.copy()
                    self.mask_viewer._refresh_display()
                QMessageBox.information(
                    self, 'Dead Pixel Detection',
                    f'Detected {num_flagged} {label} pixels '
                    f'({mode} mode, {self._dead_px_target} frames).\n'
                    f'Added to mask. Total masked: {self.mask_manager.num_masked_pixels}')

    def export_json_clicked(self):
        if self.mask_manager.mask is None:
            QMessageBox.information(self, 'No Mask', 'No mask to export.')
            return
        default_path = os.path.join(self.mask_manager.masks_dir, 'bad_pixels.json')
        filepath, _ = QFileDialog.getSaveFileName(
            self, 'Export JSON BadPixel File', default_path,
            'JSON files (*.json);;All files (*)')
        if not filepath:
            return
        try:
            self.mask_manager.export_json_mask(filepath)
            QMessageBox.information(
                self, 'Export Complete',
                f'Exported {self.mask_manager.num_masked_pixels} bad pixels to:\n{filepath}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to export JSON mask:\n{e}')

    def clear_mask_clicked(self):
        if self.mask_manager.mask is None:
            return
        reply = QMessageBox.question(
            self, 'Clear Mask',
            'Remove the active mask?',
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.mask_manager.clear_mask()
            self._update_mask_labels()

    def _on_mask_edited(self, mask):
        self.mask_manager.mask = mask.copy()
        self.mask_manager.save_active_mask()
        self._update_mask_labels()

    def _update_mask_labels(self):
        if self.mask_manager.mask is not None:
            path = self.mask_manager.mask_path or 'In memory'
            self.lbl_mask_info.setText(os.path.basename(path))
            self.lbl_mask_info.setToolTip(path)
            count = self.mask_manager.num_masked_pixels
            pct = self.mask_manager.mask_fraction * 100
            self.lbl_mask_pixel_count.setText(f'{count:,} ({pct:.1f}%)')
        else:
            self.lbl_mask_info.setText('No mask loaded')
            self.lbl_mask_info.setToolTip('')
            self.lbl_mask_pixel_count.setText('0')

    def start_timers(self) -> None:
        """
        Starts timers for updating labels and plotting at specified frequencies.
        """
        if self.reader is not None and self.reader.channel.isMonitorActive():
            self.timer_labels.start(int(1000/100))
            self.timer_plot.start(int(1000/self.plotting_frequency.value()))

    def stop_timers(self) -> None:
        """
        Stops the updating of main window labels and plots.
        """
        self.timer_plot.stop()
        self.timer_labels.stop()

    def set_pixel_ordering(self) -> None:
        """
        Checks which pixel ordering is selected on startup
        """
        if self.reader is not None:
            if self.rbtn_C.isChecked():
                self.reader.pixel_ordering = 'C'
                self.reader.image_is_transposed = True 
            elif self.rbtn_F.isChecked():
                self.reader.pixel_ordering = 'F'
                self.image_is_transposed = False
                self.reader.image_is_transposed = False
                
    def trigger_save_caches(self) -> None:
        if not self.file_writer_thread.isRunning():
                self.file_writer_thread.start()
        self.file_writer.save_caches_to_h5(clear_caches=True)

    def c_ordering_clicked(self) -> None:
        """
        Sets the pixel ordering to C style.
        """
        if self.reader is not None:
            self.reader.pixel_ordering = 'C'
            self.reader.image_is_transposed = True

    def f_ordering_clicked(self) -> None:
        """
        Sets the pixel ordering to Fortran style.
        """
        if self.reader is not None:
            self.reader.pixel_ordering = 'F'
            self.image_is_transposed = False
            self.reader.image_is_transposed = False

    def _launch_pyfai(self) -> None:
        pv_address = self._input_channel or "pvapy:image"
        cmd = [sys.executable, '-m', 'dashpva.viewer.pyFAI_analysis', '--pv-address', pv_address]
        threshold_enabled = self.chk_threshold.isChecked() if hasattr(self, 'chk_threshold') else False
        if threshold_enabled and self.reader is not None:
            min_thresh, max_thresh = self.get_threshold_range()
            if max_thresh > 0:
                cmd.extend(['--threshold-min', str(min_thresh), '--threshold-max', str(max_thresh)])
        mask_path = self.mask_manager.mask_path or os.path.join(
            self.mask_manager.masks_dir, self.mask_manager.DEFAULT_MASK_FILENAME)
        cmd.extend(['--mask-file', mask_path])
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None, start_new_session=True)
            print(f'[Area Detector] pyFAI launched with PV: {pv_address}')
        except Exception as e:
            print(f'[Area Detector] Failed to launch pyFAI: {e}')

    def _launch_phase_fitter(self) -> None:
        pv_address = self._input_channel or "pvapy:image"
        cmd = [sys.executable, '-m', 'dashpva.viewer.phase_fitter', '--pv-address', pv_address]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None, start_new_session=True)
            print(f'[Area Detector] Phase Fitter launched with PV: {pv_address}')
        except Exception as e:
            print(f'[Area Detector] Failed to launch Phase Fitter: {e}')

    def _launch_hkl3d(self) -> None:
        cmd = [sys.executable, '-m', 'dashpva.viewer.hkl3d.hkl_3d_viewer']
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None, start_new_session=True)
            print('[Area Detector] HKL 3D Viewer launched')
        except Exception as e:
            print(f'[Area Detector] Failed to launch HKL 3D Viewer: {e}')

    def _launch_scan_view(self) -> None:
        pv_address = self._input_channel or "pvapy:image"
        cmd = [sys.executable, '-m', 'dashpva.viewer.scan_view', '--channel', pv_address]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None, start_new_session=True)
            print(f'[Area Detector] 2D Scan Visualization launched with PV: {pv_address}')
        except Exception as e:
            print(f'[Area Detector] Failed to launch 2D Scan Visualization: {e}')

    def start_live_view_clicked(self) -> None:
        """
        Initializes the connections to the PVA channel using the provided Channel Name.

        This method ensures that any existing connections are cleared and re-initialized.
        Also starts monitoring the stats and adds ROIs to the viewer.
        """
        # Surface a progress indicator the moment the user clicks; the spinner
        # stays visible across PVA connect → ROI sweep → Stats sweep, hidden by
        # the terminal "ROIs and stats ready" / error emit from _connect_pv_pollers.
        self.pv_pollers_status.emit(f"Connecting to {self._input_channel}…", "info")
        try:
            self.stop_timers()
            self.image_view.clear()
            self.horizontal_avg_plot.getPlotItem().clear()
            # Drop any ROI rectangles + "ROI too large" labels from the
            # previous PV. add_rois() appends both kinds to self.rois.
            for item in self.rois:
                self.image_view.getView().removeItem(item)
            self.rois.clear()
            self.reset_rsm_vars()
            if self.reader is None:
                self.reader = PVAReader(input_channel=self._input_channel)
                self.file_writer = HDF5Writer(self.reader.OUTPUT_FILE_LOCATION, self.reader)
                self.file_writer.moveToThread(self.file_writer_thread)
            else:
                if self.reader.channel.isMonitorActive():
                    self.reader.stop_channel_monitor()
                if self.file_writer_thread.isRunning():
                    self.file_writer_thread.quit()
                    self.file_writer_thread.wait()
                for hkl_pv in self.hkl_pvs.values():
                    try:
                        hkl_pv.clear_callbacks()
                    except Exception:
                        pass
                self.file_writer.hdf5_writer_finished.disconnect()
                del self.reader
                self.reader = PVAReader(input_channel=self._input_channel)
                self.file_writer.pva_reader = self.reader
            # Reconnecting signals
            self.reader.reader_scan_complete.connect(self.trigger_save_caches)
            self.file_writer.hdf5_writer_finished.connect(self.on_writer_finished)

            if self.reader.CACHING_MODE == 'scan':
                self.file_writer_thread.start()
            elif self.reader.CACHING_MODE == 'bin':
                self.slider = QSlider()
                self.slider.setRange(0, self.reader.BIN_COUNT-1)
                self.slider.setValue(0)
                self.slider.setOrientation(Qt.Horizontal) 
                self.slider.setTickPosition(QSlider.TicksAbove)
                self.viewer_layout.addWidget(self.slider, 1, 1)
                
            self.set_pixel_ordering()
            self.transpose_image_checked()
            self.reader.start_channel_monitor()
        except Exception as e:
            print(f'Failed to Connect to {self._input_channel}: {e}')
            # Terminal "error" message hides the spinner via _on_pv_pollers_status.
            self.pv_pollers_status.emit(f"Connect failed: {e}", "error")
            self.image_view.clear()
            self.horizontal_avg_plot.getPlotItem().clear()
            self.reset_rsm_vars()
            del self.file_writer
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
            self.file_writer_thread.terminate()
        
        try:
            if self.reader is not None:
                # All PV connections (METADATA.CA + HKL + ROI + Stats) are
                # done on one shared daemon thread so live-view startup never
                # blocks the UI on slow CA timeouts. Previously METADATA.CA
                # and HKL ran synchronously here and could freeze the GUI for
                # several seconds when any PV was dead (caget × 0.5s timeout
                # × N dead PVs).
                import threading
                threading.Thread(target=self._connect_pv_pollers, daemon=True).start()
                # add_rois() runs from the rois_ready signal once the background
                # sweep populates reader.rois — calling it here would race the
                # async caget and find an empty dict.
                self.start_timers()
        except Exception as e:
            print(f'[Diffraction Image Viewer] Error Starting Image Viewer {e}')

    def stop_live_view_clicked(self) -> None:
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        if self.reader is not None:
            if self.reader.channel.isMonitorActive():
                self.reader.stop_channel_monitor()
            self.stop_timers()
            for key in self.stats_dialogs:
                self.stats_dialogs[key] = None
            for hkl_pv in self.hkl_pvs.values():
                hkl_pv.clear_callbacks()
                hkl_pv.disconnect()
            self.hkl_pvs = {}
            self.hkl_data = {}
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

    def stats_button_clicked(self) -> None:
        """
        Creates a popup dialog for viewing the stats of a specific button.

        This method identifies the button pressed and opens the corresponding stats dialog.
        """
        if self.reader is not None:
            sending_button = self.sender()
            text = sending_button.text()
            self.stats_dialogs[text] = RoiStatsDialog(parent=self,
                                                     stats_text=text,
                                                     timer=self.timer_labels)
            self.stats_dialogs[text].show()

    def stats_plot_button_clicked(self) -> None:
        """
        Creates a live-updating plot dialog for the corresponding Stats device.
        Button text format: 'Plot Stats1' -> stats_text = 'Stats1'
        """
        if self.reader is not None:
            sending_button = self.sender()
            # Extract stats name: 'Plot Stats1' -> 'Stats1'
            stats_text = sending_button.text().replace('Plot ', '')
            # Close existing plot dialog for this stats if open
            existing = self.stats_plot_dialogs.get(stats_text)
            if existing is not None:
                existing.close()
            self.stats_plot_dialogs[stats_text] = RoiStatsPlotDialog(
                parent=self,
                stats_text=stats_text,
                timer=self.timer_labels
            )
            self.stats_plot_dialogs[stats_text].show()

    STATS_GROUPS = ('Stats1', 'Stats2', 'Stats3', 'Stats4', 'Stats5')
    STATS_FIELDS = ('Total_RBV', 'MinValue_RBV', 'MaxValue_RBV', 'Sigma_RBV', 'MeanValue_RBV')

    def start_stats_monitors(self) -> None:
        """Connect to Stats PVs built from the detector prefix.

        Names are constructed as ``{prefix}:Stats{N}:{field}`` for N=1..5 and
        the standard area-detector field set. Per group, as soon as one PV
        fails to connect, the rest of that group is skipped — if Total isn't
        present, the rest aren't useful. Called from a background thread
        (see ``_connect_pv_pollers``).
        """
        prefix = self.reader.pva_prefix
        if not prefix:
            return
        for group in self.STATS_GROUPS:
            for field in self.STATS_FIELDS:
                pv = f"{prefix}:{group}:{field}"
                try:
                    pv_value = caget(pv, timeout=0.15)
                    if pv_value is None:
                        break
                    self.stats_data[pv] = pv_value
                    camonitor(pvname=pv, callback=self.stats_ca_callback)
                except Exception:
                    break

    def _connect_pv_pollers(self) -> None:
        """Single background sweep: HKL → ROI → Stats → METADATA.CA.

        All four share one daemon thread so we don't fan out to multiple
        connection threads on every Start Live View, and so the GUI thread
        never blocks on slow CA timeouts. Each step internally short-circuits
        per-group on first failure, so worst-case wait is small even when many
        PVs are dead. Sets ``self._pv_pollers_loading`` for the duration and
        posts progress to the GUI via ``pv_pollers_status``.

        METADATA.CA is intentionally last — it tends to be the largest set
        and most likely to contain dead PVs, so connecting the visible
        pieces (ROI rectangles + stats labels) first keeps the user-facing
        UI responsive even if the metadata sweep takes a while.
        """
        self._pv_pollers_loading = True
        try:
            if self.reader is None:
                return
            self.reader.start_scan_monitor()
            self.pv_pollers_status.emit("Loading HKL monitors…", "info")
            self.start_hkl_monitors()
            if not self.reader.rois:
                self.pv_pollers_status.emit("Loading ROIs…", "info")
                self.reader.start_roi_backup_monitor()
            if self.reader.rois:
                self.rois_ready.emit()
            self.pv_pollers_status.emit("Loading stats…", "info")
            self.start_stats_monitors()
            # Metadata CA sweep disabled — was suspected of slowing the GUI.
            # if 'METADATA' in self.reader.config:
            #     self.pv_pollers_status.emit("Loading metadata PVs…", "info")
            #     self.reader.start_metadata_ca_monitor()
            self.pv_pollers_status.emit("ROIs and stats ready", "info")
        except Exception as e:
            self.pv_pollers_status.emit(f"PV poller error: {e}", "error")
        finally:
            self._pv_pollers_loading = False

    def _build_pv_pollers_indicator(self) -> None:
        """Add a hidden indeterminate progress bar + label to the status bar."""
        try:
            sb = self.statusBar()
            self._pv_pollers_label = QLabel("")
            self._pv_pollers_spinner = QProgressBar()
            self._pv_pollers_spinner.setRange(0, 0)        # indeterminate animation
            self._pv_pollers_spinner.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            self._pv_pollers_spinner.setTextVisible(False)
            # Add to the LEFT side of the status bar (next to where messages live)
            sb.addWidget(self._pv_pollers_spinner)
            sb.addWidget(self._pv_pollers_label)
            self._pv_pollers_spinner.hide()
            self._pv_pollers_label.hide()
        except Exception:
            self._pv_pollers_spinner = None
            self._pv_pollers_label = None

    def _on_pv_pollers_status(self, message: str, level: str) -> None:
        """Slot for ``pv_pollers_status`` — runs on the GUI thread.

        Shows an indeterminate spinner + the latest progress text while the
        ROI/Stats sweep is running; hides both as soon as the sweep finishes
        (success or error). Also routes through ``update_status`` so the
        ``LogManager`` captures the message at the appropriate level.
        """
        try:
            lower = message.lower()
            done = ('ready' in lower) or ('error' in lower) or ('done' in lower)
            if self._pv_pollers_spinner is not None and self._pv_pollers_label is not None:
                if done:
                    self._pv_pollers_spinner.hide()
                    self._pv_pollers_label.hide()
                    self._pv_pollers_label.setText("")
                else:
                    self._pv_pollers_label.setText(message)
                    self._pv_pollers_spinner.show()
                    self._pv_pollers_label.show()
        except Exception:
            pass
        self.update_status(message, level=level)

    def stats_ca_callback(self, pvname, value, **kwargs) -> None:
        """
        Updates the stats PV value based on changes observed by `camonitor`.

        Args:
            pvname (str): The name of the specific Stat PV that has been updated.
            value: The new value sent by the monitor for the PV.
            **kwargs: Additional keyword arguments sent by the monitor.
        """
        self.stats_data[pvname] = value
        
    def on_writer_finished(self, message) -> None:
        print(message)
        self.file_writer_thread.quit()
        self.file_writer_thread.wait()

    def show_rois_checked(self) -> None:
        """Global show/hide; per-ROI checkboxes act as a sub-mask on top."""
        self._apply_roi_visibility()

    def _apply_roi_visibility(self) -> None:
        """An ROI is visible iff the global ``display_rois`` is checked AND
        its per-ROI checkbox in the ROI dock is checked."""
        if self.reader is None:
            return
        global_on = self.display_rois.isChecked()
        for idx, roi in enumerate(self.rois):
            if roi is None:
                continue
            per_roi_on = self.chk_show_roi[idx].isChecked() if idx < len(self.chk_show_roi) else True
            if global_on and per_roi_on:
                roi.show()
            else:
                roi.hide()

    def freeze_image_checked(self) -> None:
        """
        Toggles freezing/unfreezing of the plot based on the checked state
        without stopping the collection of PVA objects.
        """
        if self.reader is not None:
            if self.freeze_image.isChecked():
                self.stop_timers()
            else:
                self.start_timers()
    
    def transpose_image_checked(self) -> None:
        """
        Toggles whether the image data is transposed based on the checkbox state.
        """
        if self.reader is not None:
            if self.chk_transpose.isChecked():
                self.image_is_transposed = True
            else: 
                self.image_is_transposed = False

    def reset_first_plot(self) -> None:
        """
        Resets the `first_plot` flag, ensuring the next plot behaves as the first one.
        """
        self.first_plot = True

    def rotation_count(self) -> None:
        """
        Cycles the image rotation number between 1 and 4.
        """
        self.rot_num = next(rot_gen)

    def add_rois(self) -> None:
        """Adds ROIs to the image viewer using the themed ROI palette."""
        try:
            roi_colors = ROI_COLORS
            # Track how many ROIs are too big for offset calculation
            too_big_count = 0
            for roi_num, roi in self.reader.rois.items():
                x = roi.get("MinX", 0) if not(self.image_is_transposed) else roi.get('MinY',0)
                y = roi.get("MinY", 0) if not(self.image_is_transposed) else roi.get('MinX',0)
                width = roi.get("SizeX", 0) if not(self.image_is_transposed) else roi.get('SizeY',0)
                height = roi.get("SizeY", 0) if not(self.image_is_transposed) else roi.get('SizeX',0)
                
                # Skip ROIs with invalid dimensions
                if width <= 0 or height <= 0:
                    continue
                
                # Extract ROI number from key (e.g., "ROI1" -> 1, "ROI2" -> 2)
                try:
                    # Try to extract number from ROI key (e.g., "ROI1" -> 1)
                    # Look for digits in the key
                    digits = ''.join(c for c in roi_num if c.isdigit())
                    if digits:
                        roi_index = int(digits) - 1
                    else:
                        # Fallback: use index in dictionary
                        roi_index = list(self.reader.rois.keys()).index(roi_num)
                except (ValueError, IndexError):
                    # Final fallback: use 0 (first color)
                    roi_index = 0
                
                # Ensure index is within bounds
                roi_index = max(0, min(roi_index, len(roi_colors) - 1))
                roi_color = roi_colors[roi_index]
                
                # Flag ROIs larger than the actual image
                roi_too_big = False
                image_shape = self.reader.shape if hasattr(self.reader, 'shape') and len(self.reader.shape) >= 2 else None
                if image_shape and image_shape[0] > 0 and image_shape[1] > 0:
                    img_width = image_shape[1] if not self.image_is_transposed else image_shape[0]
                    img_height = image_shape[0] if not self.image_is_transposed else image_shape[1]
                    if width > img_width or height > img_height:
                        roi_too_big = True
                        x = -50 + (too_big_count * 250)
                        y = -50
                        width = img_width + 100
                        height = img_height + 100
                        too_big_count += 1
                
                # Create ROI with final dimensions (ensured to be reasonable)
                roi = pg.ROI(pos=[x,y],
                            size=[width, height],
                            movable=False,
                            pen=pg.mkPen(color=roi_color))
                self.rois.append(roi)
                self.image_view.addItem(roi)
                
                # Add label if ROI is too big. Append to self.rois so
                # start_live_view_clicked removes it on the next PV switch.
                if roi_too_big:
                    label = pg.TextItem(f'{roi_num} - ROI too large', color=roi_color, anchor=(0, 0))
                    label.setPos(x + 5, y + 5)
                    self.image_view.addItem(label)
                    self.rois.append(label)
                
                roi.sigRegionChanged.connect(self.update_roi_region)
            self._apply_roi_visibility()
        except Exception as e:
            print(f'[Diffraction Image Viewer] Failed to add ROIs:{e}')


    def start_hkl_monitors(self) -> None:
        """
        Initializes camonitors for HKL values and stores them in a dictionary.

        Runs on the background poller thread (see ``_connect_pv_pollers``).
        Connection + initial-get timeouts are kept short so a dead PV doesn't
        stall the whole sweep — pyepics defaults to ~5 s per dead get.
        """
        try:
            if self.reader.HKL_IN_CONFIG:
                # Use the HKL config values verbatim — different profiles store
                # PV names differently (some bare, some pre-prefixed like
                # '6idb1:DetectorSetup:...'), so prepending pva_prefix here
                # would double-prefix the ones that already include it.
                self.hkl_config = self.reader.config["HKL"]
                if not self.hkl_pvs:
                    for section, pv_dict in self.hkl_config.items():
                        for section_key, pv_name in pv_dict.items():
                            if pv_name not in self.hkl_pvs:
                                self.hkl_pvs[pv_name] = PV(
                                    pvname=pv_name,
                                    connection_timeout=0.15,
                                )
                for pv_name, pv_obj in self.hkl_pvs.items():
                    self.hkl_data[pv_name] = pv_obj.get(timeout=0.15)
                    pv_obj.add_callback(callback=self.hkl_ca_callback)
                missing = [n for n, v in self.hkl_data.items() if v is None]
                got = len(self.hkl_data) - len(missing)
                print(f"[Diffraction Image Viewer] HKL monitors started: "
                      f"{got}/{len(self.hkl_data)} PVs returned values"
                      + (f"; missing/None: {missing}" if missing else ""))
                self.hkl_data_updated.emit(True)
        except Exception as e:
            print(f"[Diffraction Image Viewer] Failed to initialize HKL monitors: {e}")

    def hkl_ca_callback(self, pvname, value, **kwargs) -> None:
        """
        Callback for updating HKL values based on changes observed by `camonitor`.

        Args:
            pvname (str): The name of the PV that has been updated.
            value: The new value sent by the monitor for the PV.
            **kwargs: Additional keyword arguments sent by the monitor.
        """
        self.hkl_data[pvname] = value
        # Always re-emit so handle_hkl_data_update re-runs hkl_setup(); the prior
        # qx/qy/qz guard created a deadlock — those only get set by update_rsm(),
        # so a failed bootstrap meant no future callback could ever recover.
        self.hkl_data_updated.emit(True)

    def handle_hkl_data_update(self):
        if self.reader is not None and not self.stop_hkl.isChecked() and self.hkl_data:
            try:
                self.hkl_setup()
                if self.q_conv is not None:
                    self.update_rsm()
            except Exception as e:
                print(f'[DashPVA] HKL update failed (will retry next frame): {e}')
  
                
    def hkl_setup(self) -> None:
        if (self.hkl_config is not None) and (not self.stop_hkl.isChecked()):
            try:
                # Get everything for the sample circles
                sample_circle_keys = [pv_name for section, pv_dict in self.hkl_config.items() if section.startswith('SAMPLE_CIRCLE') for pv_name in pv_dict.values()]
                self.sample_circle_directions = []
                self.sample_circle_names = []
                self.sample_circle_positions = []
                for pv_key in sample_circle_keys:
                    if pv_key.endswith('DirectionAxis'):
                        direction = self.hkl_data.get(pv_key)
                        if direction is None:
                            raise ValueError(f"Missing sample circle direction PV data: {pv_key}")
                        self.sample_circle_directions.append(direction)
                    elif pv_key.endswith('SpecMotorName'):
                        name = self.hkl_data.get(pv_key)
                        if name is None:
                            raise ValueError(f"Missing sample circle motor name PV data: {pv_key}")
                        self.sample_circle_names.append(name)
                    elif pv_key.endswith('Position'):
                        position = self.hkl_data.get(pv_key)
                        if position is None:
                            raise ValueError(f"Missing sample circle position PV data: {pv_key}")
                        self.sample_circle_positions.append(position)
                
                # Get everything for the detector circles
                det_circle_keys = [pv_name for section, pv_dict in self.hkl_config.items() if section.startswith('DETECTOR_CIRCLE') for pv_name in pv_dict.values()]
                self.det_circle_directions = []
                self.det_circle_names = []
                self.det_circle_positions = []
                for pv_key in det_circle_keys:
                    if pv_key.endswith('DirectionAxis'):
                        direction = self.hkl_data.get(pv_key)
                        if direction is None:
                            raise ValueError(f"Missing detector circle direction PV data: {pv_key}")
                        self.det_circle_directions.append(direction)
                    elif pv_key.endswith('SpecMotorName'):
                        name = self.hkl_data.get(pv_key)
                        if name is None:
                            raise ValueError(f"Missing detector circle motor name PV data: {pv_key}")
                        self.det_circle_names.append(name)
                    elif pv_key.endswith('Position'):
                        position = self.hkl_data.get(pv_key)
                        if position is None:
                            raise ValueError(f"Missing detector circle position PV data: {pv_key}")
                        self.det_circle_positions.append(position)
                
                # Primary Beam Direction
                primary_beam_pvs = self.hkl_config.get('PRIMARY_BEAM_DIRECTION', {}).values()
                self.primary_beam_directions = [self.hkl_data.get(axis_number) for axis_number in primary_beam_pvs]
                if any(val is None for val in self.primary_beam_directions):
                    raise ValueError("Missing primary beam direction PV data")
                
                # Inplane Reference Direction
                inplane_ref_pvs = self.hkl_config.get('INPLANE_REFERENCE_DIRECITON', {}).values()
                self.inplane_reference_directions = [self.hkl_data.get(axis_number) for axis_number in inplane_ref_pvs]
                if any(val is None for val in self.inplane_reference_directions):
                    raise ValueError("Missing inplane reference direction PV data")

                # Sample Surface Normal Direction
                surface_normal_pvs = self.hkl_config.get('SAMPLE_SURFACE_NORMAL_DIRECITON', {}).values()
                self.sample_surface_normal_directions = [self.hkl_data.get(axis_number) for axis_number in surface_normal_pvs]
                if any(val is None for val in self.sample_surface_normal_directions):
                    raise ValueError("Missing sample surface normal direction PV data")

                # UB Matrix
                ub_matrix_pv = self.hkl_config['SPEC']['UB_MATRIX_VALUE']
                self.ub_matrix = self.hkl_data.get(ub_matrix_pv)
                if self.ub_matrix is None or not isinstance(self.ub_matrix, np.ndarray) or self.ub_matrix.size != 9:
                    raise ValueError("Invalid UB Matrix data")
                self.ub_matrix = np.reshape(self.ub_matrix,(3,3))

                # Energy
                energy_pv = self.hkl_config['SPEC']['ENERGY_VALUE']
                self.energy = self.hkl_data.get(energy_pv)
                if self.energy is None:
                    raise ValueError("Missing energy PV data")
                self.energy *= 1000

                # Make sure all values are setup correctly before instantiating QConversion
                if all([self.sample_circle_directions, self.det_circle_directions, self.primary_beam_directions]):
                    self.q_conv = xu.experiment.QConversion(self.sample_circle_directions, 
                                                            self.det_circle_directions, 
                                                            self.primary_beam_directions)
                else:
                    self.q_conv = None
                    raise ValueError("QConversion initialization failed due to missing PV data.")

            except Exception as e:
                print(f'[Diffraction Image Viewer] Error Setting up HKL: {e}')
                self.q_conv = None # Reset to None on failure to prevent invalid calculations
                
    def create_rsm(self) -> np.ndarray:
        """
        Creates a reciprocal space map (RSM) from the current detector image.

        This method uses the xrayutilities package to convert detector coordinates 
        to reciprocal space coordinates (Q-space). It requires:
        - Valid detector statistics data
        - Properly initialized HKL parameters
        - Detector setup parameters (pixel directions, center channel, size, etc.)

        Returns:
            numpy.ndarray:
            - The Q-space coordinates for the current detector image,
                         or None if required data is missing.
            - shape (N, 3) containing qx, qy, and qz values.

        The conversion uses the current sample and detector angles along with the UB matrix
        to transform from angular to reciprocal space coordinates.
        """
        if self.reader is not None and self.hkl_data and (not self.stop_hkl.isChecked()):
            try:
                if (self.q_conv is None or
                    not self.inplane_reference_directions or
                    not self.sample_surface_normal_directions or
                    self.energy is None):
                    return None

                hxrd = xu.HXRD(self.inplane_reference_directions,
                            self.sample_surface_normal_directions,
                            en=self.energy,
                            qconv=self.q_conv)

                roi = [0, self.reader.shape[0], 0, self.reader.shape[1]]
                # Look up by the PV name in the active HKL config so any prefix
                # scheme (xidb:, 6idb1:, none) works without code edits — same
                # pattern HpcRsmProcessor uses.
                ds_cfg = self.hkl_config.get('DETECTOR_SETUP', {}) or {}
                cch1, cch2 = self.hkl_data[ds_cfg['CENTER_CHANNEL_PIXEL']][:2]
                distance = self.hkl_data[ds_cfg['DISTANCE']]
                pixel_dir1 = self.hkl_data[ds_cfg['PIXEL_DIRECTION_1']]
                pixel_dir2 = self.hkl_data[ds_cfg['PIXEL_DIRECTION_2']]
                nch1 = self.reader.shape[0] # Number of detector pixels along direction 1
                nch2 = self.reader.shape[1] # Number of detector pixels along direction 2
                size_xy = self.hkl_data[ds_cfg['SIZE']]
                pixel_width1 = size_xy[0] / nch1
                pixel_width2 = size_xy[1] / nch2

                hxrd.Ang2Q.init_area(pixel_dir1, pixel_dir2, cch1=cch1, cch2=cch2,
                                    Nch1=nch1, Nch2=nch2, pwidth1=pixel_width1, 
                                    pwidth2=pixel_width2, distance=distance, roi=roi)
                
                angles = [*self.sample_circle_positions, *self.det_circle_positions]
                
                return hxrd.Ang2Q.area(*angles, UB=self.ub_matrix)
            except Exception as e:
                print(f'[Diffration Image Viewer] Error Creating RSM: {e}')
                return
        else:
            return
        
    def reset_rsm_vars(self) -> None:
        self.hkl_data = {}
        self.rois.clear()
        self.qx = None
        self.qy = None
        self.qz = None


    def update_rois(self) -> None:
        """
        Updates the positions and sizes of ROIs based on changes from the EPICS software.
        Loops through the cached ROIs and adjusts their parameters accordingly.
        """
        too_big_count = 0
        image_shape = self.reader.shape if hasattr(self.reader, 'shape') and len(self.reader.shape) >= 2 else None
        for roi, roi_dict in zip(self.rois, self.reader.rois.values()):
            if roi is None:
                continue
            x_pos = roi_dict.get("MinX",0) if not(self.image_is_transposed) else roi_dict.get('MinY',0)
            y_pos = roi_dict.get("MinY",0) if not(self.image_is_transposed) else roi_dict.get('MinX',0)
            width = roi_dict.get("SizeX",0) if not(self.image_is_transposed) else roi_dict.get('SizeY',0)
            height = roi_dict.get("SizeY",0) if not(self.image_is_transposed) else roi_dict.get('SizeX',0)

            if image_shape:
                img_width = image_shape[1] if not self.image_is_transposed else image_shape[0]
                img_height = image_shape[0] if not self.image_is_transposed else image_shape[1]
                if width > img_width or height > img_height:
                    width = img_width + 100
                    height = img_height + 100
                    x_pos = -50 + (too_big_count * 250)
                    y_pos = -50
                    too_big_count += 1

            roi.setPos(pos=x_pos, y=y_pos)
            roi.setSize(size=(width, height))
        self.image_view.update()

    def update_roi_region(self) -> None:
        """
        Forces the image viewer to refresh when an ROI region changes.
        """
        self.image_view.update()

    def update_pv_prefix(self) -> None:
        self._input_channel = self.pv_prefix.text()
    
    def on_double_click(self, event) -> None:
        """
        Handle double-click on the image view to place crosshairs.
        Double-click to show/move crosshairs, triple-click to hide them.
        """
        if self.reader is None or self.reader.image is None:
            return
        if self.crosshair_visible:
            # Already showing — hide on next double-click (acts as toggle)
            self.crosshair_v.hide()
            self.crosshair_h.hide()
            self.crosshair_visible = False
        else:
            scene_pos = event.scenePos()
            view_box = self.image_view.getView().getViewBox()
            view_pos = view_box.mapSceneToView(scene_pos)
            self.crosshair_v.setPos(view_pos.x())
            self.crosshair_h.setPos(view_pos.y())
            self.crosshair_v.show()
            self.crosshair_h.show()
            self.crosshair_visible = True
    
    def _sync_havg_yrange(self, vb, y_range):
        """Keep horizontal avg plot y-range in sync with the image view."""
        self.horizontal_avg_plot.getPlotItem().getViewBox().setYRange(
            y_range[0], y_range[1], padding=0)

    def update_mouse_pos(self, pos) -> None:
        """
        Maps the mouse position in the Image Viewer to the corresponding pixel value.

        Args:
            pos (QPointF): Position event sent by the mouse moving.
        """
        if pos is not None:
            if self.reader is not None:
                img = self.image_view.getImageItem()
                q_pointer = img.mapFromScene(pos)
                self.mouse_x, self.mouse_y = int(q_pointer.x()), int(q_pointer.y())
                self.update_mouse_labels()
    
    def update_mouse_labels(self) -> None:
            if self.reader is not None:
                if self.image is not None:
                    if 0 <= self.mouse_x < self.image.shape[0] and 0 <= self.mouse_y < self.image.shape[1]:
                        self.mouse_x_val.setText(f"{self.mouse_x}")
                        self.mouse_y_val.setText(f"{self.mouse_y}")
                        self.mouse_px_val.setText(f'{self.image[self.mouse_x][self.mouse_y]:.2f}')
                        # Stop HKL is meant to freeze the visible HKL output, not
                        # just halt the qx/qy/qz recompute. Without this gate the
                        # mouse H/K/L labels keep changing on every mouse move
                        # (different indices into the stale array).
                        if (self.qx is not None and len(self.qx) > 0
                                and not self.stop_hkl.isChecked()):
                            self.mouse_h.setText(f'{self.qx[self.mouse_x][self.mouse_y]:.7f}')
                            self.mouse_k.setText(f'{self.qy[self.mouse_x][self.mouse_y]:.7f}')
                            self.mouse_l.setText(f'{self.qz[self.mouse_x][self.mouse_y]:.7f}')

    def update_labels(self) -> None:
        """
        Updates the UI labels with current connection and cached data.
        """
        if self.reader is not None:
            provider_name = f"{self.reader.provider if self.reader.channel.isMonitorActive() else 'N/A'}"
            is_connected = 'Connected' if self.reader.channel.isMonitorActive() else 'Disconnected'
            self.provider_name.setText(provider_name)
            self.is_connected.setText(is_connected)
            self.missed_frames_val.setText(f'{self.reader.frames_missed:d}')
            self.frames_received_val.setText(f'{self.reader.frames_received:d}')
            self.plot_call_id.setText(f'{self.call_id_plot:d}')
            self.update_mouse_labels()
            if len(self.reader.shape):
                self.size_x_val.setText(f'{self.reader.shape[0]:d}')
                self.size_y_val.setText(f'{self.reader.shape[1]:d}')
            self.data_type_val.setText(self.reader.display_dtype)
            self.update_threshold_label()
            for i in range(1, 5):
                label = getattr(self, f"roi{i}_total_value")
                label.setText(f"{float(self.stats_data.get(f'{self.reader.pva_prefix}:Stats{i}:Total_RBV', 0.0)):.2f}")
            self.stats5_total_value.setText(f"{float(self.stats_data.get(f'{self.reader.pva_prefix}:Stats5:Total_RBV', 0.0)):.2f}")

    def update_rsm(self) -> None:
        if (self.reader is not None) and (not self.stop_hkl.isChecked()):
            if self.hkl_data:
                qxyz = self.create_rsm()
                self.qx = qxyz[0].T if self.image_is_transposed else qxyz[0]
                self.qy = qxyz[1].T if self.image_is_transposed else qxyz[1]
                self.qz = qxyz[2].T if self.image_is_transposed else qxyz[2]

    def update_image(self) -> None:
        """
        Redraws plots based on the configured update rate.

        Processes the image data according to main window settings, such as rotation
        and log transformations. Also sets initial min/max pixel values in the UI.
        """
        if self.reader is not None:
            self.call_id_plot +=1
            if self.reader.CACHING_MODE in ['', 'alignment', 'scan']:
                self.image = self.reader.image
            elif self.reader.CACHING_MODE == 'bin':
                index = self.slider.value()
                self.image = np.reshape(np.mean(np.stack(self.reader.cached_images[index]), axis=0), self.reader.shape) # (self.reader.shape)

            if self.image is not None:
                # Collect frame for dead pixel detection if active
                self._collect_dead_pixel_frame()
                # Apply vectorized thresholding if enabled
                if self.chk_threshold.isChecked():
                    self.image = self.apply_threshold(self.image)
                # Apply mask if enabled (before transpose/rotation)
                if self.chk_apply_mask.isChecked() and self.mask_manager.mask is not None:
                    self.image = self.mask_manager.apply_to_image(self.image)
                    if self.mask_manager.shape_mismatch_info is not None:
                        mask_shape, img_shape = self.mask_manager.shape_mismatch_info
                        self.mask_manager.shape_mismatch_info = None
                        QMessageBox.warning(self, 'Mask Shape Mismatch',
                            f'Mask shape {mask_shape} does not match image shape {img_shape}.\n'
                            f'The mask is being resized automatically.\n\n'
                            f'Consider rotating/transposing the mask in the Mask Viewer\n'
                            f'to match your detector orientation, then save.')
                self.image = np.transpose(self.image) if self.image_is_transposed else self.image
                self.image = np.rot90(m=self.image, k=self.rot_num)
                if len(self.image.shape) == 2:
                    min_level, max_level = np.min(self.image), np.max(self.image)
                    if self.log_image.isChecked():
                        # Ensure non negative values
                        self.image = np.maximum(self.image, 0)
                        epsilon = 1e-10
                        self.image = np.log10(self.image + 1)
                        min_level = np.log10(max(min_level, epsilon) + 1)
                        max_level = np.log10(max_level + 1)
                    if self.first_plot:
                        self.image_view.setImage(self.image,
                                                autoRange=False,
                                                autoLevels=False,
                                                levels=(min_level, max_level),
                                                autoHistogramRange=False)
                        # Set colormap separately, then re-collapse the gradient
                        # to the lightweight preset so we don't end up with one
                        # tick handle per colormap stop.
                        self.image_view.setColorMap(self.cet_colormap)
                        try:
                            self.image_view.ui.histogram.gradient.loadPreset('viridis')
                        except Exception:
                            pass
                        # Auto sets the max value based on first incoming image
                        if not self.chk_autoscale.isChecked():
                            self.max_setting_val.setValue(max_level)
                            self.min_setting_val.setValue(min_level)
                        self.first_plot = False
                    else:
                        self.image_view.setImage(self.image,
                                                autoRange=False,
                                                autoLevels=False,
                                                autoHistogramRange=False)
                    if self.chk_autoscale.isChecked():
                        self.apply_autoscale()
                # Separate image update for horizontal average plot
                self.horizontal_avg_plot.plot(x=np.mean(self.image, axis=0),
                                            y=np.arange(self.image.shape[1]),
                                            clear=True)

                self.min_px_val.setText(f"{min_level:.2f}")
                self.max_px_val.setText(f"{max_level:.2f}")
    
    def update_min_max_setting(self) -> None:
        # Passive when autoscale is on — autoscale drives the LUT each frame.
        if not self.chk_autoscale.isChecked():
            self.image_view.setLevels(self.min_setting_val.value(),
                                      self.max_setting_val.value())

    def autoscale_checked(self) -> None:
        if self.chk_autoscale.isChecked() and self.image is not None:
            self.apply_autoscale()

    def apply_autoscale(self) -> None:
        if self.image is None:
            return
        intensities = self.image.flatten()
        intensities = intensities[np.isfinite(intensities)]
        if len(intensities) == 0:
            return
        min_pct, max_pct = np.percentile(intensities, [5, 95])
        min_pct = float(min_pct)
        max_pct = float(max_pct)
        self.min_setting_val.blockSignals(True)
        self.max_setting_val.blockSignals(True)
        self.min_setting_val.setValue(min_pct)
        self.max_setting_val.setValue(max_pct)
        self.min_setting_val.blockSignals(False)
        self.max_setting_val.blockSignals(False)
        self.image_view.setLevels(min_pct, max_pct)
    
    def get_threshold_range(self) -> tuple:
        """
        Determines the threshold range based on the data type.
        
        Returns:
            tuple: (min_threshold, max_threshold) based on data type
        """
        if self.reader is None or self.reader.display_dtype is None:
            return (0, 0)
        
        dtype_map = {
            'ubyteValue': (0, 2**8 - 2),      # uint8: 0 to 254 (accounting for zero-based indexing)
            'ushortValue': (0, 2**16 - 2),    # uint16: 0 to 65534 (accounting for zero-based indexing)
            'uintValue': (0, 2**32 - 2),      # uint32: 0 to 4294967294 (accounting for zero-based indexing)
            'floatValue': (0, 2**16 - 1),     # float32: 0 to 65535
            'doubleValue': (0, 2**16 - 1),    # float64: 0 to 65535
        }
        
        return dtype_map.get(self.reader.display_dtype, (0, 0))
    
    def update_threshold_label(self) -> None:
        """
        Updates the threshold range label based on current data type.
        """
        min_thresh, max_thresh = self.get_threshold_range()
        self.lbl_threshold_range.setText(f"{min_thresh} to {max_thresh}")
    
    def threshold_checked(self) -> None:
        """
        Handles threshold checkbox state changes.
        Updates the label visibility and applies thresholding if enabled.
        """
        if self.chk_threshold.isChecked():
            self.update_threshold_label()
            self.lbl_threshold_range.show()
        else:
            self.lbl_threshold_range.hide()
    
    def apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Applies vectorized thresholding to the image based on data type limits.
        Values below min_thresh are set to min_thresh, values above max_thresh are set to 0.
        
        Args:
            image: Input image array
            
        Returns:
            Thresholded image array
        """
        min_thresh, max_thresh = self.get_threshold_range()
        if min_thresh == 0 and max_thresh == 0:
            return image
        
        # Vectorized thresholding: 
        # - Values below min_thresh are set to min_thresh
        # - Values above max_thresh are set to 0
        result = image.copy()
        result[result < min_thresh] = min_thresh
        result[result > max_thresh] = 0
        return result
    
    def closeEvent(self, event):
        """
        Custom close event to clean up resources, including stat dialogs.

        Args:
            event (QCloseEvent): The close event triggered when the main window is closed.
        """
        try:
            s = _settings()
            s.setValue("area_det_dock_state", self.saveState(_DOCK_STATE_VERSION))
            s.setValue("area_det_window_geom", self.saveGeometry())
        except Exception:
            pass
        del self.stats_dialogs # otherwise dialogs stay in memory
        del self.stats_plot_dialogs
        if self.mask_viewer is not None:
            self.mask_viewer.close()
        if self.file_writer_thread.isRunning():
            self.file_writer_thread.quit()
            self.file_writer_thread
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    # Fusion is scoped to this viewer because the dock-title QSS rules only
    # take effect under Fusion — native Linux styles (Breeze/GTK) draw dock
    # titles in native code and ignore stylesheets. Other viewers keep the
    # platform's native style.
    from PyQt5.QtWidgets import QStyleFactory
    app.setStyle(QStyleFactory.create("Fusion"))
    configure_app(app)
    window = ConfigDialog()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()