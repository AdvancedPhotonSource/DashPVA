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
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QMessageBox
# Custom imported classes
# Add the parent directory to the path so the font_scaling.py file can be imported
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import PVAReader, HDF5Writer, SizeManager
import settings as app_settings
from hkl_3d_slice_window import HKL3DSliceWindow
from utils.log_manager import LogMixin


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


class HKLImageWindow(QMainWindow, LogMixin):
    images_plotted = pyqtSignal(bool)

    def __init__(self, input_channel=None):
        """
        Initializes the main window for real-time image visualization and manipulation.

        Args:
            input_channel (str): The PVA input channel for the detector.
        """
        super(HKLImageWindow, self).__init__()
        uic.loadUi('gui/hkl_viewer_window.ui', self)
        try:
            self.set_log_manager(viewer_name="HKLViewer")
        except Exception:
            pass
        self.setWindowTitle('HKL Viewer')
        self.show()

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
                self.btn_save_h5.clicked.disconnect()
                self.btn_plot_cache.clicked.disconnect()
                self.file_writer.hdf5_writer_finished.disconnect()
                if self.reader.channel.isMonitorActive():
                    self.reader.stop_channel_monitor()
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
            # self.set_pixel_ordering()
            self.reader.start_channel_monitor()
            self.start_timers()

    def stop_live_view_clicked(self) -> None:
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        if self.reader is not None:
            self.reader.stop_channel_monitor()
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
            
    def update_image(self, is_scan_signal:bool=False) -> None:
        """
        Redraws plots based on the configured update rate.

        Processes the image data according to main window settings, such as rotation
        and log transformations. Also sets initial min/max pixel values in the UI.
        """
        if self.reader is not None:
            self.call_id_plot +=1
            if self.reader.cached_images is not None and self.reader.cached_qx is not None:
                self.plotter.clear()
                try:
                    num_images = len(self.reader.cached_images)
                    num_rsm = len(self.reader.cached_qx)
                    if num_images !=  num_rsm:
                        raise ValueError(f'Size of caches are uneven:\nimages:{num_images}\nqxyz: {num_rsm}')
                    # connect all cached data
                    flat_intensity = np.concatenate(self.reader.cached_images, dtype=np.float32)
                    qx = np.concatenate(self.reader.cached_qx, dtype=np.float32)
                    qy = np.concatenate(self.reader.cached_qy, dtype=np.float32)
                    qz = np.concatenate(self.reader.cached_qz, dtype=np.float32)
                    
                    points = np.column_stack((
                        qx, qy, qz
                    ))
                except Exception as e:
                    try:
                        if hasattr(self, 'logger'):
                            self.logger.exception(f'[HKL Viewer] Failed to concatenate caches: {e}')
                    except Exception:
                        pass


                try:
                    if is_scan_signal:
                        clear_caches = True
                        self.images_plotted.emit(clear_caches)
                        
                    self.min_intensity = np.min(flat_intensity)
                    self.max_intensity = np.max(flat_intensity)
                    self.sbox_max_intensity.setValue(self.max_intensity)

                    self.cloud = pyv.PolyData(points)
                    self.cloud['intensity'] = flat_intensity 

                    self.lut = pyv.LookupTable(cmap='magma')  
                    self.lut.below_range_color = 'black'
                    self.lut.above_range_color = 'black'
                    self.lut.below_range_opacity = 0
                    self.lut.above_range_opacity = 0 
                    self.update_opacity()
                    self.update_intensity()
                
                    self.actor = self.plotter.add_mesh(
                        self.cloud,
                        scalars='intensity',
                        cmap=self.lut,
                        point_size=3
                    )

                    self.plotter.show_bounds(xtitle='H Axis', ytitle='K Axis', ztitle='L Axis')
                except Exception as e:
                    try:
                        if hasattr(self, 'logger'):
                            self.logger.exception(f"[HKL Viewer] Failed to update 3D plot: {e}")
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
            self.actor.mapper.scalar_range = (self.min_intensity,self.max_intensity)
    
    def closeEvent(self, event):
        """pass
        Custom close event to clean up resources, including stat dialogs.

        Args:
            event (QCloseEvent): The close event triggered when the main window is closed.
        """
        if self.file_writer_thread.isRunning():
            self.file_writer_thread.quit()
            self.file_writer_thread
        super(HKLImageWindow,self).closeEvent(event)

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