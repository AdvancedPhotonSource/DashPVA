import sys, pathlib
import os
import h5py
import time
import subprocess
import numpy as np
import os.path as osp
import pyqtgraph as pg
import pyvista as pyv
import pyvistaqt as pyvqt
from pyvistaqt import QtInteractor, BackgroundPlotter
from PyQt5 import uic
# from epics import caget
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QMessageBox)
# Custom imported classes
# Add the project root first so viewer.core resolves correctly even if viewer is already cached
_project_root = str(pathlib.Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# If viewer was already imported before our sys.path change, patch its __path__ so core is visible
if 'viewer' in sys.modules:
    import importlib
    _viewer_core = str(pathlib.Path(__file__).resolve().parent.parent / 'core')
    if _viewer_core not in sys.modules['viewer'].__path__:
        sys.modules['viewer'].__path__.append(_viewer_core)
from utils import PVAReader, HDF5Writer, SizeManager
import settings as app_settings
from viewer.hkl_3d_slice_window import HKL3DSliceWindow
from utils.log_manager import LogMixin
from viewer.core.base_window import BaseWindow
from viewer.hkl3d.docks.plot_mode import PlotModeDock
from viewer.hkl3d.docks.stats import StatsDock
from viewer.hkl3d.docks.image import ImageDock


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
        uic.loadUi('gui/pv_config.ui', self)
        try:
            self.set_log_manager(viewer_name="HKLConfigDialog")
        except Exception:
            pass
        self.setWindowTitle('PV Config')
        self.input_channel = ""
        self.init_ui()
        self.btn_accept_reject.accepted.connect(self.dialog_accepted)

    def init_ui(self) -> None:
        self.le_input_channel.setText(self.le_input_channel.text())

    def dialog_accepted(self) -> None:
        """Open the HKL viewer with the given input channel; config comes from settings.py."""
        self.input_channel = self.le_input_channel.text()
        self.hkl_3d_viewer = HKLImageWindow(input_channel=self.input_channel)


class HKLImageWindow(BaseWindow):
    images_plotted = pyqtSignal(bool)

    def __init__(self, input_channel=None):
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
        self._input_channel = input_channel or 'pvapy:image'
        self.pv_prefix.setText(self._input_channel)

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

        self.stats_dock = StatsDock(main_window=self)
        self.image_dock = ImageDock(main_window=self)

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

        # Ring buffer for cumulative mode
        self._CUMULATIVE_MAX     = 100
        self._CUMULATIVE_MAX_PTS = 1_000_000  # hard cap: total strided points across all frames
        self._cum_frame_size     = 0   # raw pixels per frame
        self._cum_pts_per_frame  = 0   # strided points per frame kept in ring buffer
        self._cum_stride         = 1   # spatial stride applied when ingesting each frame
        self._cum_n_frames       = 0   # frames currently in buffer (0–100)
        self._cum_write_slot     = 0   # next ring slot to write
        self._cum_pts_raw        = None  # plain np.ndarray (MAX*ppf, 3) — ring buffer for xyz
        self._cum_int_raw        = None  # plain np.ndarray (MAX*ppf,)   — ring buffer for intensity
        # Auto-scale color range only on the first plot of each live-view session
        self._first_plot = True

        # Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        # self.plotting_frequency.valueChanged.connect(self.start_timers)
        # self.log_image.clicked.connect(self.update_image)
        self.sbox_min_intensity.editingFinished.connect(self.update_intensity)
        self.sbox_max_intensity.editingFinished.connect(self.update_intensity)
        self.sbox_min_opacity.editingFinished.connect(self.update_opacity)
        self.sbox_max_opacity.editingFinished.connect(self.update_opacity)
        self.btn_3d_slice_window.clicked.connect(self.open_3d_slice_window)

        self.show()

    def start_timers(self) -> None:
        """
        Starts timers for updating labels and plotting at specified frequencies.
        """
        self.timer_labels.start(int(1000/100))

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
            # A double check to make sure there isn't a connection already when starting
            self.stop_timers()
            self.plotter.clear()
            if self.reader is None:
                self.reader = PVAReader(input_channel=self._input_channel,
                                         viewer_type='rsm')
                self.file_writer = HDF5Writer(self.reader.OUTPUT_FILE_LOCATION, self.reader)
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
                if self.reader.channel.isMonitorActive():
                    try:
                        self.reader.stop_channel_monitor()
                    except Exception:
                        pass
                if self.file_writer_thread.isRunning():
                    self.file_writer_thread.quit()
                    self.file_writer_thread.wait()
                del self.reader
                self.reader = PVAReader(input_channel=self._input_channel,
                                         viewer_type='rsm')
                self.file_writer.pva_reader = self.reader
            self.btn_save_h5.clicked.connect(self.save_caches_clicked)
            self.btn_plot_cache.clicked.connect(self.update_image_from_button)
            self.reader.reader_scan_complete.connect(self.update_image_from_scan)
            self.reader.reader_new_frame.connect(self._on_new_frame)
            #self.images_plotted.connect(self.trigger_save_caches)
            #self.file_writer.hdf5_writer_finished.connect(self.on_writer_finished)
            if self.reader.CACHING_MODE == 'scan':
                self.file_writer_thread.start()
        except Exception as e:
            try:
                if hasattr(self, 'logger'):
                    self.logger.exception(f'Failed to Connect to {self._input_channel}: {e}')
            except Exception:
                pass
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
        
        if self.reader is not None:
            self._first_plot = True
            self.actor = None
            self.cloud = None
            self.lut = None
            self._cum_frame_size    = 0
            self._cum_pts_per_frame = 0
            self._cum_stride        = 1
            self._cum_n_frames      = 0
            self._cum_write_slot    = 0
            self._cum_pts_raw       = None
            self._cum_int_raw       = None
            self.reader.start_channel_monitor()
            self.start_timers()
            if not self.plot_mode_dock.is_post_scan:
                self.plot_mode_dock.start_plot_timer()

    def stop_live_view_clicked(self) -> None:
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        if self.reader is not None:
            if self.reader.channel.isMonitorActive():
                try:
                    self.reader.stop_channel_monitor()
                except Exception:
                    pass
            self.stop_timers()
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

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

    def update_image_from_scan(self) -> None:
        self.update_image(is_scan_signal=True)

    def update_image_from_button(self) -> None:
        self.update_image(is_scan_signal=False)

    def _on_new_frame(self) -> None:
        self.plot_mode_dock.notify_new_frame()
        if self.plot_mode_dock.is_realtime and self.reader is not None:
            rsm = self.reader.rsm_attributes
            if self.reader.image is not None and rsm:
                intensity = np.ravel(self.reader.image).astype(np.float32)
                qx = np.asarray(rsm['qx'], dtype=np.float32)
                qy = np.asarray(rsm['qy'], dtype=np.float32)
                qz = np.asarray(rsm['qz'], dtype=np.float32)
                frame_size = len(intensity)
                if self._cum_frame_size != frame_size or self._cum_pts_raw is None:
                    # First frame or detector size changed: reset ring buffer.
                    # Pass qx/qy/qz so placeholders are seeded at real HKL positions
                    # (avoids bounding box being anchored to the origin).
                    self._cum_frame_size = frame_size
                    self._cum_n_frames   = 0
                    self._cum_write_slot = 0
                    self._init_cumulative_cloud(frame_size, qx=qx, qy=qy, qz=qz)
                    if self._first_plot:
                        self.sbox_min_intensity.setValue(float(np.min(intensity)))
                        self.sbox_max_intensity.setValue(float(np.max(intensity)))
                        self._first_plot = False
                slot  = self._cum_write_slot
                ppf   = self._cum_pts_per_frame
                start = slot * ppf
                end   = start + ppf
                s = self._cum_stride
                self._cum_pts_raw[start:end, 0] = qx[::s][:ppf]
                self._cum_pts_raw[start:end, 1] = qy[::s][:ppf]
                self._cum_pts_raw[start:end, 2] = qz[::s][:ppf]
                self._cum_int_raw[start:end]     = intensity[::s][:ppf]
                self._cum_write_slot = (slot + 1) % self._CUMULATIVE_MAX
                self._cum_n_frames   = min(self._cum_n_frames + 1, self._CUMULATIVE_MAX)

    def _init_cumulative_cloud(self, frame_size: int,
                               qx=None, qy=None, qz=None) -> None:
        """Allocate the plain-numpy ring buffer for cumulative mode.

        Computes the stride so the total stored points stay ≤ CUMULATIVE_MAX_PTS.
        If the first frame's qx/qy/qz are provided, all placeholder slots are seeded
        with those positions so the bounding box is correct from the very first render
        (without qx/qy/qz the placeholders would be at the origin, inflating the axes).
        """
        total_raw = self._CUMULATIVE_MAX * frame_size
        self._cum_stride        = max(1, total_raw // self._CUMULATIVE_MAX_PTS)
        self._cum_pts_per_frame = frame_size // self._cum_stride
        n_total = self._CUMULATIVE_MAX * self._cum_pts_per_frame
        ppf, s  = self._cum_pts_per_frame, self._cum_stride
        if qx is not None:
            # Tile the first frame's positions across all slots so the bounding box
            # reflects real HKL space rather than being anchored to the origin.
            first_pts = np.column_stack([qx[::s][:ppf], qy[::s][:ppf], qz[::s][:ppf]])
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
                self._cum_stride        = 1
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

    def update_image_cumulative(self) -> None:
        """Realtime mode: pass the full ring buffer (with strided data) to _plot_point_cloud.

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
        if not self.reader.rsm_attributes:
            return
        try:
            intensity = np.asarray(np.ravel(self.reader.image), dtype=np.float32)
            qx = np.asarray(self.reader.rsm_attributes['qx'], dtype=np.float32)
            qy = np.asarray(self.reader.rsm_attributes['qy'], dtype=np.float32)
            qz = np.asarray(self.reader.rsm_attributes['qz'], dtype=np.float32)
            self._plot_point_cloud(np.column_stack((qx, qy, qz)), intensity)
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
            flat_intensity = np.concatenate(self.reader.cached_images, dtype=np.float32)
            qx = np.concatenate(self.reader.cached_qx, dtype=np.float32)
            qy = np.concatenate(self.reader.cached_qy, dtype=np.float32)
            qz = np.concatenate(self.reader.cached_qz, dtype=np.float32)
            points = np.column_stack((qx, qy, qz))
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
        """
        Updates the min/max intensity levels in the HKL Viewer based on UI settings.
        """
        """
        Updates the min/max intensity levels in the HKL Viewer based on UI settings.
        """
        self.min_opacity = self.sbox_min_opacity.value()
        self.max_opacity = self.sbox_max_opacity.value()
        if self.min_opacity > self.max_opacity:
            self.min_opacity, self.max_opacity = self.max_opacity, self.min_opacity
            self.sbox_min_opacity.setValue(self.min_opacity)
            self.sbox_max_opacity.setValue(self.max_opacity)
        if self.lut is not None:
            self.lut.apply_opacity([self.min_opacity,self.max_opacity])

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
    
    def closeEvent(self, event):
        """pass
        Custom close event to clean up resources, including stat dialogs.

        Args:
            event (QCloseEvent): The close event triggered when the main window is closed.
        """
        if self.file_writer_thread.isRunning():
            self.file_writer_thread.quit()
            self.file_writer_thread
        super().closeEvent(event)

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
        window = ConfigDialog()
        window.show()
        size_manager = SizeManager(app=app)
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        sys.exit(0)