# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

import gc
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyvista as pyv
from PyQt5 import uic

# from epics import caget
from PyQt5.QtCore import QByteArray, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox
from pyvistaqt import QtInteractor

import dashpva.settings as app_settings
from dashpva.gui import configure_app, ui_path
from dashpva.utils import HDF5Handler, PVAReader, SizeManager
from dashpva.utils.log_manager import LogMixin
from dashpva.utils.point_sampling import (
    evenly_spaced_indices,
    sampled_point_cloud,
    sampled_point_cloud_chunks,
)
from dashpva.utils.rsm_grid_transport import (
    GridControlClient,
    GridTransportError,
    preview_from_status,
)
from dashpva.viewer.core.base_window import BaseWindow
from dashpva.viewer.hkl3d.docks.grid_control import GridControlDock
from dashpva.viewer.hkl3d.docks.image import ImageDock
from dashpva.viewer.hkl3d.docks.plot_mode import PlotModeDock
from dashpva.viewer.hkl3d.docks.stats import StatsDock
from dashpva.viewer.hkl3d.live_grid_volume import LiveGridVolume
from dashpva.viewer.hkl_3d_slice_window import HKL3DSliceWindow

# Bump when the dock set changes — restoreState rejects mismatched versions,
# so a stale saved layout falls back to the default arrangement.
_DOCK_STATE_VERSION = 1


def _settings() -> QSettings:
    return QSettings("DashPVA", "Viewer")


class ConfigDialog(QDialog, LogMixin):

    def __init__(self):
        """
        Class that does initial setup for getting the pva prefix, collector address,
        and the path to the json that stores the pvs that will be observed

        Attributes:
            input_channel (str): Input channel for PVA.
            config_path (str): Path to the ROI configuration file.
        """
        super(ConfigDialog,self).__init__()
        uic.loadUi(ui_path("pv_config.ui"), self)
        try:
            self.set_log_manager(viewer_name="HKLConfigDialog")
        except Exception:
            pass
        self.setWindowTitle('HKL 3D Config')
        self.input_channel = ""
        self.init_ui()
        self.btn_accept_reject.accepted.connect(self.dialog_accepted)

    def init_ui(self) -> None:
        self.le_input_channel.setPlaceholderText("e.g. processor:1:analysis")
        self.le_input_channel.setText(app_settings.get_input_channel_hkl3d())

    def dialog_accepted(self) -> None:
        """Open the HKL viewer with the given input channel; config comes from settings.py."""
        self.input_channel = self.le_input_channel.text()
        app_settings.save_input_channel_hkl3d(self.input_channel)
        self.hkl_3d_viewer = HKLImageWindow(input_channel=self.input_channel)


class HKLImageWindow(BaseWindow):
    images_plotted = pyqtSignal(bool)

    def __init__(self, input_channel=None, grid_client=None, grid_executor=None):
        """
        Initializes the main window for real-time image visualization and manipulation.

        Args:
            input_channel (str): The PVA input channel for the detector.
        """
        super().__init__(ui_file_name='hkl_viewer_window.ui', viewer_name='HKLViewer',
                         visible_actions=['Windows', 'Documentation'],
                         size_policy={})
        self.setWindowTitle('HKL Viewer')
        self.resize(1280, 800)

        # Initializing Viewer variables
        self.reader = None
        self.image = None
        self.call_id_plot = 0
        self.image_is_transposed = False
        self._input_channel = input_channel or app_settings.get_input_channel_hkl3d()
        self.pv_prefix.setText(self._input_channel)
        self._set_connection_label(False)

        # Initializing but not starting timers so they can be reached by different functions
        self.timer_labels = QTimer()
        self.file_writer_thread = QThread()
        self.timer_labels.timeout.connect(self.update_labels)

        # HKL values
        self.hkl_config = None
        self.hkl_data = {}
        self.qx = None
        self.qy = None
        self.qz = None
        self.processes = {}
        
        # Docks
        self.plot_mode_dock = PlotModeDock(main_window=self)
        self.plot_mode_dock.mode_changed.connect(self._on_mode_changed)
        self.plot_mode_dock.plot_timer_fired.connect(self._on_timer_plot)

        self.grid_dock = GridControlDock(main_window=self)
        self.grid_dock.start_requested.connect(self._on_grid_start)
        self.grid_dock.stop_requested.connect(self._on_grid_stop)
        self.grid_dock.clear_requested.connect(self._on_grid_clear)
        self.grid_dock.save_requested.connect(self._on_grid_save)
        self.grid_client = grid_client
        self._grid_executor = grid_executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='DashPVA-grid-control'
        )
        self._owns_grid_executor = grid_executor is None
        self._grid_future = None
        self._grid_job = (None, None)
        self._grid_queue = deque()
        self._grid_preview_key = None
        self._grid_last_payload = None
        self._grid_clear_pending = False
        self._autoscaled_limits = None
        self._grid_timer = QTimer(self)
        self._grid_timer.setInterval(app_settings.RSM_GRID_STATUS_POLL_MS)
        self._grid_timer.timeout.connect(self._grid_tick)

        self.stats_dock = StatsDock(main_window=self)
        self.image_dock = ImageDock(main_window=self)
        # One right-hand column: Plot Mode, Live Grid, then Stats/Image as tabs.
        self.splitDockWidget(self.plot_mode_dock, self.grid_dock, Qt.Vertical)
        self.splitDockWidget(self.grid_dock, self.stats_dock, Qt.Vertical)
        self.tabifyDockWidget(self.stats_dock, self.image_dock)
        self.stats_dock.raise_()
        self.resizeDocks(
            [self.plot_mode_dock, self.grid_dock, self.stats_dock],
            [150, 430, 320],
            Qt.Vertical,
        )
        self.resizeDocks([self.grid_dock], [480], Qt.Horizontal)
        self._restore_layout()

        # Aliases so the rest of the file can use self.widget_name unchanged
        self.frames_received_val = self.stats_dock.frames_received_val
        self.missed_frames_val = self.stats_dock.missed_frames_val
        self.max_px_val = self.stats_dock.max_px_val
        self.min_px_val = self.stats_dock.min_px_val
        self.data_type_val       = self.stats_dock.data_type_val
        self.sbox_min_intensity  = self.stats_dock.sbox_min_intensity
        self.sbox_max_intensity  = self.stats_dock.sbox_max_intensity
        self.sbox_min_opacity    = self.stats_dock.sbox_min_opacity
        self.sbox_max_opacity    = self.stats_dock.sbox_max_opacity

        self.rbtn_C              = self.image_dock.rbtn_C
        self.rbtn_F              = self.image_dock.rbtn_F
        self.log_image           = self.image_dock.log_image
        self.btn_reset_camera    = self.image_dock.btn_reset_camera
        self.btn_3d_slice_window = self.image_dock.btn_3d_slice_window
        self.btn_plot_cache      = self.image_dock.btn_plot_cache
        self.btn_save_h5         = self.image_dock.btn_save_h5

        # Adding widgets manually to have better control over them
        pyv.set_plot_theme('dark')
        self.plotter = QtInteractor(self)
        self.viewer_layout.addWidget(self.plotter,1,1)

        # pyvista vars
        self.actor = None
        self.lut = None
        self.cloud = None
        self.min_intensity = 0.0
        self.max_intensity = 0.0
        self.min_opacity = 0.0
        self.max_opacity = 1.0
        self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')
        self.grid_volume = LiveGridVolume(self.plotter)

        # Ring buffer for cumulative mode
        self._CUMULATIVE_MAX = min(app_settings.PREVIEW['HKL_MAX_FRAMES'], app_settings.PREVIEW['HKL_MAX_POINTS'])
        self._CUMULATIVE_MAX_PTS = app_settings.PREVIEW['HKL_MAX_POINTS']
        self._PER_FRAME_MAX_PTS = app_settings.PREVIEW['HKL_MAX_POINTS']
        self._POST_SCAN_MAX_PTS = app_settings.PREVIEW['HKL_MAX_POINTS']
        self._cum_frame_size     = 0   # raw pixels per frame
        self._cum_pts_per_frame  = 0   # sampled points per frame kept in ring buffer
        self._cum_indices        = np.empty(0, dtype=np.intp)
        self._cum_n_frames       = 0   # frames currently in buffer (0–100)
        self._cum_write_slot     = 0   # next ring slot to write
        self._cum_pts_raw        = None  # plain np.ndarray (MAX*ppf, 3) — ring buffer for xyz
        self._cum_int_raw        = None  # plain np.ndarray (MAX*ppf,)   — ring buffer for intensity
        self._pending_frame = None
        # Auto-scale color range only on the first plot of each live-view session
        self._first_plot = True

        # Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        # One Start/Stop toggle, as in the Area Detector viewer.
        self._live_running = False
        self.stop_live_view.hide()
        self.start_live_view.setText("Connect")
        self.start_live_view.clicked.connect(self._toggle_live_view)
        for widget in (self.start_live_view, self.pv_prefix):
            widget.setProperty("compact", True)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        metrics = self.start_live_view.fontMetrics()
        self.start_live_view.setFixedWidth(
            max(metrics.horizontalAdvance(text) for text in ("Connect", "Disconnect")) + 40
        )
        # self.plotting_frequency.valueChanged.connect(self.start_timers)
        # self.log_image.clicked.connect(self.update_image)
        self.sbox_min_intensity.editingFinished.connect(self.update_intensity)
        self.sbox_max_intensity.editingFinished.connect(self.update_intensity)
        self.sbox_min_opacity.editingFinished.connect(self.update_opacity)
        self.sbox_max_opacity.editingFinished.connect(self.update_opacity)
        self.btn_3d_slice_window.clicked.connect(self.open_3d_slice_window)
        self.btn_reset_camera.clicked.connect(self.reset_camera)
        self.log_image.toggled.connect(self._on_log_toggled)

        self.show()
        self._show_cache_controls(not self.plot_mode_dock.is_gridded)
        if self.plot_mode_dock.is_gridded:
            self._start_grid_polling()

    def _teardown_reader(self) -> None:
        """Fully disconnect and release the current reader, its signals, and all resources."""
        self._pending_frame = None
        if self.reader is None:
            return
        try:
            self.reader.reader_scan_complete.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            self.reader.reader_new_frame.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            self.reader.stop_channel_monitor()
        except Exception:
            pass
        self.reader = None
        gc.collect()

    def start_timers(self) -> None:
        """
        Starts timers for updating labels and plotting at specified frequencies.
        """
        self.timer_labels.start(app_settings.PREVIEW['LABEL_INTERVAL_MS'])

    def stop_timers(self) -> None:
        """
        Stops the updating of main window labels and plots.
        """
        self.timer_labels.stop()
        self.plot_mode_dock.stop_plot_timer()

    def start_live_view_clicked(self) -> None:
        """
        Initializes the connections to the PVA channel using the provided Channel Name.

        This method ensures that any existing connections are cleared and re-initialized.
        Also starts monitoring the stats and adds ROIs to the viewer.
        """
        try:
            app_settings.reload()
            self.stop_timers()
            self.plotter.clear()
            if self.reader is None:
                self.reader = PVAReader(input_channel=self._input_channel,
                                         viewer_type='rsm')
                self.file_writer = HDF5Handler(self.reader.OUTPUT_FILE_LOCATION, self.reader)
                self.file_writer.moveToThread(self.file_writer_thread)
            else:
                try:
                    self.btn_save_h5.clicked.disconnect()
                except RuntimeError:
                    pass
                try:
                    self.btn_plot_cache.clicked.disconnect()
                except RuntimeError:
                    pass
                try:
                    self.file_writer.hdf5_writer_finished.disconnect()
                except (RuntimeError, TypeError):
                    pass
                if self.file_writer_thread.isRunning():
                    self.file_writer_thread.quit()
                    self.file_writer_thread.wait()
                self._teardown_reader()
                self.reader = PVAReader(input_channel=self._input_channel,
                                         viewer_type='rsm')
                self.file_writer.pva_reader = self.reader
            self.btn_save_h5.clicked.connect(self.save_caches_clicked)
            self.btn_plot_cache.clicked.connect(self.update_image_from_button)
            self.reader.reader_scan_complete.connect(self.update_image_from_scan)
            self.reader.reader_new_frame.connect(self._on_new_frame)
        except Exception as e:
            try:
                if hasattr(self, 'logger'):
                    self.logger.exception(f'Failed to Connect to {self._input_channel}: {e}')
            except Exception:
                pass
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self._set_connection_label(False)

        if self.reader is not None:
            self._first_plot = True
            self.actor = None
            self.cloud = None
            self.lut = None
            self._cum_frame_size    = 0
            self._cum_pts_per_frame = 0
            self._cum_n_frames      = 0
            self._cum_write_slot    = 0
            self._cum_pts_raw       = None
            self._cum_int_raw       = None
            self.reader.start_channel_monitor()
            self.reader.start_scan_monitor()
            self.start_timers()
            if not self.plot_mode_dock.is_post_scan:
                self.plot_mode_dock.start_plot_timer()
        self._set_live_running(self.reader is not None)

    def _toggle_live_view(self) -> None:
        if self._live_running:
            self.stop_live_view_clicked()
        else:
            self.start_live_view_clicked()

    def _set_live_running(self, running: bool) -> None:
        self._live_running = running
        self.start_live_view.setText("Disconnect" if running else "Connect")

    def stop_live_view_clicked(self) -> None:
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        self._teardown_reader()
        self.stop_timers()
        self.provider_name.setText('N/A')
        self._set_connection_label(False)
        self._set_live_running(False)

    def trigger_save_caches(self, clear_caches:bool=True) -> None:
        if not self.file_writer_thread.isRunning():
                self.file_writer_thread.start()
        self.file_writer.save_caches_to_h5(clear_caches=clear_caches)

    def save_caches_clicked(self) -> None:
        if not self.reader.channel.isMonitorActive():  
            if not self.file_writer_thread.isRunning():
                self.file_writer_thread.start()
            self.file_writer.save_caches_to_h5()
        else:
            QMessageBox.critical(None,
                                'Error',
                                'Stop Live View to Save Cache',
                                QMessageBox.Ok)
    
    def on_writer_finished(self, message) -> None:
        print(message)
        self.file_writer_thread.quit()
        self.file_writer_thread.wait()

    # def freeze_image_checked(self) -> None:
    #     """
    #     Toggles freezing/unfreezing of the plot based on the checked state
    #     without stopping the collection of PVA objects.
    #     """
    #     if self.reader is not None:
    #         if self.freeze_image.isChecked():
    #             self.stop_timers()
    #         else:
    #             self.start_timers()

    def update_pv_prefix(self) -> None:
        """
        Updates the input channel prefix based on the value entered in the prefix field.
        """
        self._input_channel = self.pv_prefix.text()

    def _set_connection_label(self, connected: bool) -> None:
        state = "connected" if connected else "disconnected"
        if self.is_connected.property("connectionState") == state:
            return
        self.is_connected.setText("Connected" if connected else "Disconnected")
        self.is_connected.setProperty("connectionState", state)
        self.is_connected.style().unpolish(self.is_connected)
        self.is_connected.style().polish(self.is_connected)

    def update_labels(self) -> None:
        """
        Updates the UI labels with current connection and cached data.
        """
        if self.reader is not None:
            self.stats_dock.update_preview_metrics()
            provider_name = f"{self.reader.provider if self.reader.channel.isMonitorActive() else 'N/A'}"
            self.provider_name.setText(provider_name)
            self._set_connection_label(self.reader.channel.isMonitorActive())
            self.missed_frames_val.setText(f'{self.reader.frames_missed:d}')
            self.frames_received_val.setText(f'{self.reader.frames_received:d}')

    def update_image_from_scan(self) -> None:
        self.update_image(is_scan_signal=True)

    def update_image_from_button(self) -> None:
        self.update_image(is_scan_signal=False)

    def _on_new_frame(self) -> None:
        if self.reader is None or self.sender() is not self.reader:
            return
        frame = self.reader.take_latest_frame()
        if frame is None or frame is self._pending_frame:
            return
        self._pending_frame = frame
        self.plot_mode_dock.notify_new_frame()
        if self.plot_mode_dock.is_gridded:
            # The grid consumer computes Q and accumulates beside the incoming
            # frames. The GUI consumes only its bounded status preview.
            return
        if self.plot_mode_dock.is_realtime and self.reader is not None:
            image = frame.image
            rsm = frame.rsm_attributes
            if image is not None and rsm:
                raw_intensity = image
                qx = np.asarray(rsm['qx'])
                qy = np.asarray(rsm['qy'])
                qz = np.asarray(rsm['qz'])
                frame_size = raw_intensity.size
                if not (qx.size == qy.size == qz.size == frame_size):
                    raise ValueError("intensity and HKL arrays must have matching sizes")
                if self._cum_frame_size != frame_size or self._cum_pts_raw is None:
                    # First frame or detector size changed: reset ring buffer.
                    # Pass qx/qy/qz so placeholders are seeded at real HKL positions
                    # (avoids bounding box being anchored to the origin).
                    self._cum_frame_size = frame_size
                    self._cum_n_frames   = 0
                    self._cum_write_slot = 0
                    self._init_cumulative_cloud(frame_size, qx=qx, qy=qy, qz=qz)
                    if self._first_plot:
                        self.sbox_min_intensity.setValue(float(np.min(raw_intensity)))
                        self.sbox_max_intensity.setValue(float(np.max(raw_intensity)))
                        self._first_plot = False
                slot  = self._cum_write_slot
                ppf   = self._cum_pts_per_frame
                start = slot * ppf
                end   = start + ppf
                indices = self._cum_indices
                self._cum_pts_raw[start:end, 0] = qx.flat[indices]
                self._cum_pts_raw[start:end, 1] = qy.flat[indices]
                self._cum_pts_raw[start:end, 2] = qz.flat[indices]
                self._cum_int_raw[start:end] = raw_intensity.flat[indices]
                self._cum_write_slot = (slot + 1) % self._CUMULATIVE_MAX
                self._cum_n_frames   = min(self._cum_n_frames + 1, self._CUMULATIVE_MAX)


    def _init_cumulative_cloud(self, frame_size: int,
                               qx=None, qy=None, qz=None) -> None:
        """Allocate the plain-numpy ring buffer for cumulative mode.

        Selects matching indices within the total point budget.
        If the first frame's qx/qy/qz are provided, all placeholder slots are seeded
        with those positions so the bounding box is correct from the very first render
        (without qx/qy/qz the placeholders would be at the origin, inflating the axes).
        """
        per_frame_budget = max(1, self._CUMULATIVE_MAX_PTS // self._CUMULATIVE_MAX)
        self._cum_indices = evenly_spaced_indices(frame_size, per_frame_budget)
        self._cum_pts_per_frame = len(self._cum_indices)
        n_total = self._CUMULATIVE_MAX * self._cum_pts_per_frame
        if qx is not None:
            # Tile the first frame's positions across all slots so the bounding box
            # reflects real HKL space rather than being anchored to the origin.
            first_pts = np.column_stack([
                np.asarray(qx).flat[self._cum_indices],
                np.asarray(qy).flat[self._cum_indices],
                np.asarray(qz).flat[self._cum_indices],
            ])
            self._cum_pts_raw = np.tile(first_pts, (self._CUMULATIVE_MAX, 1)).astype(np.float32)
        else:
            self._cum_pts_raw = np.zeros((n_total, 3), dtype=np.float32)
        # All slots start invisible — intensity below LUT range hides unfilled frames.
        self._cum_int_raw = np.full(n_total, np.finfo(np.float32).min, dtype=np.float32)
        # Drop the old actor so _plot_point_cloud creates a fresh one sized to n_total
        if self.actor is not None:
            self.plotter.remove_actor(self.actor)
            self.actor = None
            self.cloud = None

    def _on_timer_plot(self, mode: str) -> None:
        if self.reader is None:
            return
        if mode == 'realtime':
            self.update_image_cumulative()
        elif mode == 'per_frame':
            self.update_image_current_frame()

    def _on_mode_changed(self, mode: str) -> None:
        self._show_cache_controls(mode != 'gridded')
        if mode == 'gridded':
            self._start_grid_polling()
        else:
            self._stop_grid_polling()
            self._reset_grid_view()
        if self.reader is None or not self.reader.channel.isMonitorActive():
            return
        if self.actor is not None:
            self.plotter.remove_actor(self.actor)
            self.actor = None
            self.cloud = None
        if mode == 'post_scan':
            self.plot_mode_dock.stop_plot_timer()
        else:
            if mode == 'realtime':
                self._cum_frame_size    = 0
                self._cum_pts_per_frame = 0
                self._cum_n_frames      = 0
                self._cum_write_slot    = 0
                self._cum_pts_raw       = None
                self._cum_int_raw       = None
            self.plot_mode_dock.start_plot_timer()

    def _plot_point_cloud(self, points: np.ndarray, intensity: np.ndarray) -> None:
        """Shared PyVista rendering.

        - Auto-scales only on the first plot of each session.
        - Creates the LUT and actor once; subsequent calls update data in-place
          so the color bar and axes never flicker.
        - Only recreates the actor when the point count changes (e.g., cumulative
          buffer filling up), using remove_actor instead of plotter.clear().
        """
        # One-time auto-scale
        if self._first_plot:
            self.sbox_min_intensity.setValue(float(np.min(intensity)))
            self.sbox_max_intensity.setValue(float(np.max(intensity)))
            self._first_plot = False

        # Create LUT once per session
        if self.lut is None:
            self.lut = pyv.LookupTable(cmap='viridis')
            self.lut.below_range_color = 'black'
            self.lut.above_range_color = 'black'
            self.lut.below_range_opacity = 0
            self.lut.above_range_opacity = 0
            self.update_opacity()
            self.update_intensity()

        n_pts = len(points)
        if self.actor is not None and self.cloud is not None and self.cloud.n_points == n_pts:
            # Same point count — write into existing VTK memory, no actor rebuild, no flicker
            self.cloud.points[:] = points
            self.cloud.point_data['intensity'][:] = intensity
            self.cloud.GetPoints().Modified()
            self.cloud.GetPointData().GetArray('intensity').Modified()
        else:
            # First render or point count changed — swap actor without clearing scene
            if self.actor is not None:
                self.plotter.remove_actor(self.actor)
            self.cloud = pyv.PolyData(points)
            self.cloud['intensity'] = intensity
            self.actor = self.plotter.add_mesh(
                self.cloud,
                scalars='intensity',
                cmap=self.lut,
                point_size=3
            )
            self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')
            self.plotter.show_bounds(xtitle='H Axis', ytitle='K Axis', ztitle='L Axis')

        self.plotter.render()


    # ---- Gridded volume mode --------------------------------------------
    # One timer, one in-flight job, a FIFO of commands: nothing is dropped, and
    # every result (status or command reply) goes through _on_grid_status.

    def _ensure_grid_client(self):
        if self.grid_client is None:
            control, status = app_settings.get_analysis_transport_channels()
            self.grid_client = GridControlClient(control, status)
        return self.grid_client

    def _start_grid_polling(self) -> None:
        if not self._grid_timer.isActive():
            self._grid_timer.start()
        self._grid_tick()

    def _stop_grid_polling(self) -> None:
        self._grid_timer.stop()

    def _queue_grid_command(self, command, payload=None, on_success=None) -> None:
        self._grid_queue.append((command, payload or {}, on_success))
        self.grid_dock.set_command_pending(True)
        self._grid_tick()

    def _grid_tick(self) -> None:
        if self._grid_future is not None:
            return
        try:
            client = self._ensure_grid_client()
        except (GridTransportError, RuntimeError) as exc:
            self._grid_queue.clear()
            self.grid_dock.set_command_pending(False)
            self.grid_dock.mark_unreachable(str(exc))
            return
        if self._grid_queue:
            command, payload, on_success = self._grid_queue.popleft()
            self._grid_job = (command, on_success)
            self._grid_future = self._grid_executor.submit(
                client.command, command, payload
            )
        else:
            self._grid_job = (None, None)
            self._grid_future = self._grid_executor.submit(client.refresh_status)
        QTimer.singleShot(50, self._grid_poll_future)

    def _grid_poll_future(self) -> None:
        future = self._grid_future
        if future is None:
            return
        if not future.done():
            QTimer.singleShot(50, self._grid_poll_future)
            return
        command, on_success = self._grid_job
        self._grid_future = None
        self._grid_job = (None, None)
        if not self._grid_queue:
            self.grid_dock.set_command_pending(False)
        try:
            state = future.result()
        except Exception as exc:
            if command is None:
                self.grid_dock.mark_unreachable(f'Live grid status unavailable: {exc}')
            else:
                if command == 'clear':
                    self._grid_clear_pending = False
                self.grid_dock.show_error(f'{command} failed: {exc}')
        else:
            if command is not None or self.grid_dock._local_error:
                self.grid_dock.show_error('')
            self._on_grid_status(state)
            if on_success is not None:
                on_success(state)
        if self._grid_queue:
            self._grid_tick()

    def _on_grid_status(self, state) -> None:
        self.grid_dock.update_status(state)
        try:
            payload = preview_from_status(state)
        except (GridTransportError, RuntimeError, ValueError) as exc:
            self.grid_dock.show_error(str(exc))
            return
        if payload is None or self._grid_clear_pending:
            if self.grid_volume.actor is not None:
                self._reset_grid_view()
            return
        key = (int(state.get('preview_publishes', 0)), int(state.get('frames_accepted', 0)))
        if key == self._grid_preview_key:
            return
        self._grid_preview_key = key
        self._render_grid_preview(payload)

    def _render_grid_preview(self, payload) -> None:
        self._grid_last_payload = payload
        autoscale = self.stats_dock.autoscale.isChecked()
        clim = None if autoscale else self._intensity_limits()
        try:
            low, high = self.grid_volume.update(
                payload,
                log=self.log_image.isChecked(),
                clim=clim,
                opacity=(self.sbox_min_opacity.value(), self.sbox_max_opacity.value()),
            )
        except Exception as exc:
            if hasattr(self, 'logger'):
                self.logger.exception(f'[HKL Viewer] Gridded render failed: {exc}')
            self.grid_dock.show_error(f'Could not render live-grid preview: {exc}')
            return
        if autoscale:
            self.sbox_min_intensity.setValue(low)
            self.sbox_max_intensity.setValue(high)
            self._autoscaled_limits = (self.sbox_min_intensity.value(),
                                       self.sbox_max_intensity.value())

    def _intensity_limits(self) -> tuple:
        low, high = self.sbox_min_intensity.value(), self.sbox_max_intensity.value()
        return (min(low, high), max(low, high))

    def _show_cache_controls(self, show: bool) -> None:
        # Legacy point-cloud cache workflow; the grid renders from its own preview.
        self.btn_plot_cache.setVisible(show)
        self.btn_save_h5.setVisible(show)

    def _reset_grid_view(self) -> None:
        self.grid_volume.clear()
        self._grid_preview_key = None
        self._grid_last_payload = None

    def _on_grid_start(self, payload: dict) -> None:
        self._queue_grid_command('start', payload)

    def _on_grid_stop(self) -> None:
        self._queue_grid_command('stop')

    def _on_grid_clear(self) -> None:
        # Status polls already in flight still carry the old preview; ignore
        # them until the clear itself is acknowledged.
        self._grid_clear_pending = True
        self._reset_grid_view()
        self._queue_grid_command('clear', on_success=self._grid_clear_done)

    def _grid_clear_done(self, _state) -> None:
        self._grid_clear_pending = False

    def _on_grid_save(self) -> None:
        name, accepted = QInputDialog.getText(
            self, 'Save gridded volume', 'File name:', text='live_grid.h5'
        )
        if not accepted or not name.strip():
            return
        self._queue_grid_command(
            'save',
            {'filename': name.strip()},
            on_success=self._grid_save_finished,
        )

    def _grid_save_finished(self, result) -> None:
        QMessageBox.information(
            self, 'Volume saved', f"Saved to {result['saved_path']}"
        )

    def update_image_cumulative(self) -> None:
        """Realtime mode: pass the bounded sampled ring to _plot_point_cloud.

        The ring buffer is always CUMULATIVE_MAX * pts_per_frame points (≤1M total).
        After the first render, _plot_point_cloud's in-place path is always taken
        (same constant point count), so there is no actor rebuild and no flicker.
        Unfilled slots carry intensity=finfo.min and are invisible via below_range_opacity=0.
        """
        if self._cum_pts_raw is None or self._cum_n_frames == 0:
            return
        try:
            self._plot_point_cloud(self._cum_pts_raw, self._cum_int_raw)
        except Exception as e:
            try:
                if hasattr(self, 'logger'):
                    self.logger.exception(f'[HKL Viewer] Failed to update cumulative plot: {e}')
            except Exception:
                pass

    def update_image_current_frame(self) -> None:
        """Per-frame mode: plot only the latest frame, independent of FLAG_PV."""
        if self.reader is None or self.reader.image is None:
            return
        frame = self._pending_frame
        if frame is None:
            return
        image = frame.image
        rsm = frame.rsm_attributes
        if image is None or not rsm:
            return
        try:
            points, intensity = sampled_point_cloud(
                image,
                rsm['qx'],
                rsm['qy'],
                rsm['qz'],
                self._PER_FRAME_MAX_PTS,
            )
            self._plot_point_cloud(points, intensity)
        except Exception as e:
            try:
                if hasattr(self, 'logger'):
                    self.logger.exception(f'[HKL Viewer] Failed to update per-frame plot: {e}')
            except Exception:
                pass

    def update_image(self, is_scan_signal:bool=False) -> None:
        """Post-scan mode: plot all scan-cached frames after FLAG_PV goes to 0."""
        if self.reader is None:
            return
        self.call_id_plot += 1
        if self.reader.cached_images is None or self.reader.cached_qx is None:
            return
        try:
            num_images = len(self.reader.cached_images)
            num_rsm = len(self.reader.cached_qx)
            if num_images != num_rsm:
                raise ValueError(f'Size of caches are uneven: images={num_images} qxyz={num_rsm}')
            points, flat_intensity = sampled_point_cloud_chunks(
                self.reader.cached_images,
                self.reader.cached_qx,
                self.reader.cached_qy,
                self.reader.cached_qz,
                self._POST_SCAN_MAX_PTS,
            )
        except Exception as e:
            try:
                if hasattr(self, 'logger'):
                    self.logger.exception(f'[HKL Viewer] Failed to concatenate caches: {e}')
            except Exception:
                pass
            return

        try:
            if is_scan_signal:
                self.images_plotted.emit(True)
            self._plot_point_cloud(points, flat_intensity)
        except Exception as e:
            try:
                if hasattr(self, 'logger'):
                    self.logger.exception(f'[HKL Viewer] Failed to update 3D plot: {e}')
            except Exception:
                pass

    def update_opacity(self) -> None:
        """Apply the min/max opacity boxes to the point cloud and live grid volume."""
        self.min_opacity = self.sbox_min_opacity.value()
        self.max_opacity = self.sbox_max_opacity.value()
        if self.min_opacity > self.max_opacity:
            self.min_opacity, self.max_opacity = self.max_opacity, self.min_opacity
            self.sbox_min_opacity.setValue(self.min_opacity)
            self.sbox_max_opacity.setValue(self.max_opacity)
        if self.lut is not None:
            self.lut.apply_opacity([self.min_opacity,self.max_opacity])
        self._apply_grid_display()

    def update_intensity(self) -> None:
        """
        Updates the min/max intensity levels in the HKL Viewer based on UI settings.
        """
        self.min_intensity = self.sbox_min_intensity.value()
        self.max_intensity = self.sbox_max_intensity.value()
        if self.min_intensity > self.max_intensity:
            self.min_intensity, self.max_intensity = self.max_intensity, self.min_intensity
            self.sbox_min_intensity.setValue(self.min_intensity)
            self.sbox_max_intensity.setValue(self.max_intensity)
        if self.lut is not None:
            self.lut.scalar_range = (self.min_intensity, self.max_intensity)
        if self.actor is not None:
            self.actor.mapper.scalar_range = (self.min_intensity, self.max_intensity)
            self.plotter.render()
        if self._autoscaled_limits is not None and self._intensity_limits() != self._autoscaled_limits:
            self.stats_dock.autoscale.setChecked(False)
        self._apply_grid_display()

    def _apply_grid_display(self) -> None:
        if self.grid_volume.actor is None:
            return
        clim = (self.grid_volume.auto_range
                if self.stats_dock.autoscale.isChecked() else self._intensity_limits())
        self.grid_volume.set_display(
            clim, (self.sbox_min_opacity.value(), self.sbox_max_opacity.value())
        )

    def _on_log_toggled(self, _checked: bool) -> None:
        if self._grid_last_payload is not None:
            self.stats_dock.autoscale.setChecked(True)
            self._render_grid_preview(self._grid_last_payload)
    
    def closeEvent(self, event):
        """Custom close event to clean up resources, including stat dialogs.

        Args:
            event (QCloseEvent): The close event triggered when the main window is closed.
        """
        self._teardown_reader()
        if self.file_writer_thread.isRunning():
            self.file_writer_thread.quit()
            self.file_writer_thread.wait()
        self._grid_timer.stop()
        if self._owns_grid_executor:
            self._grid_executor.shutdown(wait=False, cancel_futures=True)
        s = _settings()
        s.setValue("hkl3d_dock_state", self.saveState(_DOCK_STATE_VERSION))
        s.setValue("hkl3d_window_geom", self.saveGeometry())
        super().closeEvent(event)

    def _restore_layout(self) -> None:
        """Last window size and dock arrangement; the defaults above stay on failure."""
        s = _settings()
        geom = s.value("hkl3d_window_geom", QByteArray(), type=QByteArray)
        if not geom.isEmpty() and self.restoreGeometry(geom):
            avail = self.screen().availableGeometry()
            if self.width() > avail.width() or self.height() > avail.height():
                self.resize(min(self.width(), avail.width()), min(self.height(), avail.height()))
        state = s.value("hkl3d_dock_state", QByteArray(), type=QByteArray)
        if not state.isEmpty():
            self.restoreState(state, _DOCK_STATE_VERSION)

    def reset_camera(self) -> None:
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def open_3d_slice_window(self) -> None:
        try:
            self.slice_window = HKL3DSliceWindow(self) 
            self.slice_window.show()
        except Exception as e:
            try:
                if hasattr(self, 'logger'):
                    self.logger.exception("Failed to open 3D slice window", exc_info=e)
            except Exception:
                pass


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        configure_app(app)
        window = ConfigDialog()
        window.show()
        size_manager = SizeManager(app=app)
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        sys.exit(0)
