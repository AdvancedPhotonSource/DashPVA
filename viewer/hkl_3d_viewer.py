import sys
import subprocess
import numpy as np
import os.path as osp
import pyqtgraph as pg
import pyvista as pyv
import pyvistaqt as pyvqt
from pyvistaqt import QtInteractor
from PyQt5 import uic
# from epics import caget
from epics import camonitor, caget
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
# Custom imported classes
from pva_reader import PVAReader

class ConfigDialog(QDialog):

    def __init__(self):
        """
        Class that does initial setup for getting the pva prefix, collector address,
        and the path to the json that stores the pvs that will be observed

        Attributes:
            input_channel (str): Input channel for PVA.
            config_path (str): Path to the ROI configuration file.
        """
        super(ConfigDialog,self).__init__()
        uic.loadUi('gui/hkl_viewer_setup.ui', self)
        self.setWindowTitle('PV Config')
        # initializing variables to pass to Image Viewer
        self.input_channel = ""
        self.config_path =  ""
        # class can be prefilled with text
        self.init_ui()
        
        # Connecting signasl to 
        self.btn_clear.clicked.connect(self.clear_pv_setup)
        self.btn_browse.clicked.connect(self.browse_file_dialog)
        self.btn_accept_reject.accepted.connect(self.dialog_accepted) 

    def init_ui(self) -> None:
        """
        Prefills text in the Line Editors for the user.
        """
        self.le_input_channel.setText(self.le_input_channel.text())
        self.le_config.setText(self.le_onfig.text())

    def browse_file_dialog(self) -> None:
        """
        Opens a file dialog to select the path to a TOML configuration file.
        """
        self.pvs_path, _ = QFileDialog.getOpenFileName(self, 'Select TOML Config', 'pv_configs', '*.toml (*.toml)')

        self.le_config.setText(self.pvs_path)
    
    def clear_pv_setup(self) -> None:
        """
        Clears line edit that tells image view where the config file is.
        """
        self.le_config.clear()


    def dialog_accepted(self) -> None:
        """
        Handles the final step when the dialog's accept button is pressed.
        Starts the HKLImageWindow process with filled information.
        """
        self.input_channel = self.le_input_channel.text()
        self.config_path = self.le_config.text()
        if osp.isfile(self.config_path) or (self.config_path == ''):
            self.hkl_3d_viewer = HKLImageWindow(input_channel=self.input_channel,
                                            file_path=self.config_path,) 
        else:
            print('File Path Doesn\'t Exitst')  
            #TODO: ADD ERROR Dialog rather than print message so message is clearer
            self.new_dialog = ConfigDialog()
            self.new_dialog.show()    


class HKLImageWindow(QMainWindow):

    def __init__(self, input_channel='s6lambda1:Pva1:Image', file_path=''): 
        """
        Initializes the main window for real-time image visualization and manipulation.

        Args:
            input_channel (str): The PVA input channel for the detector.
            file_path (str): The file path for loading configuration.
        """
        super(HKLImageWindow, self).__init__()
        uic.loadUi('gui/hkl_viewer_window.ui', self)
        self.setWindowTitle('HKL Viewer')
        self.show()

        # Initializing Viewer variables
        self.reader = None
        self.image = None
        self.call_id_plot = 0
        self.first_plot = True
        self.image_is_transposed = False
        self._input_channel = input_channel
        self.pv_prefix.setText(self._input_channel)
        self._file_path = file_path

        # Initializing but not starting timers so they can be reached by different functions
        self.timer_labels = QTimer()
        # self.timer_plot = QTimer()
        self.timer_labels.timeout.connect(self.update_labels)
        # self.timer_plot.timeout.connect(self.update_image)

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
        self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')

        # Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        self.btn_plot_cache.clicked.connect(self.update_image)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.rbtn_C.clicked.connect(self.c_ordering_clicked)
        self.rbtn_F.clicked.connect(self.f_ordering_clicked)
        # self.log_image.clicked.connect(self.reset_first_plot)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.plotting_frequency.valueChanged.connect(self.start_timers)
        # self.log_image.clicked.connect(self.update_image)
        # TODO: create a spinbox that changes min and max setting for intensities 
        # self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        # self.min_setting_val.valueChanged.connect(self.update_min_max_setting)
        # self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)

    def start_timers(self) -> None:
        """
        Starts timers for updating labels and plotting at specified frequencies.
        """
        self.timer_labels.start(int(1000/100))
        # self.timer_plot.start(int(1000/self.plotting_frequency.value()))

    def stop_timers(self) -> None:
        """
        Stops the updating of main window labels and plots.
        """
        # self.timer_plot.stop()
        self.timer_labels.stop()

    #TODO: CHECK With 4id network camera to test if
    # start of X,Y and size of X,Y line up when transposed
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

    def start_live_view_clicked(self) -> None:
        """
        Initializes the connections to the PVA channel using the provided Channel Name.
        
        This method ensures that any existing connections are cleared and re-initialized.
        Also starts monitoring the stats and adds ROIs to the viewer.
        """
        try:
            # A double check to make sure there isn't a connection already when starting
            if self.reader is None:
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path,
                                         viewer_type='rsm')
                self.set_pixel_ordering()
                self.reader.start_channel_monitor()
            else:
                self.stop_timers()
                self.reader.stop_channel_monitor()
                del self.reader
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path,
                                         viewer_type='rsm')
                self.set_pixel_ordering()
                self.reader.start_channel_monitor()

            self.btn_save_h5.clicked.connect(self.reader.save_caches_to_h5)

        except:
            print(f'Failed to Connect to {self._input_channel}')
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
        
        if self.reader is not None:
            self.start_timers()

    def stop_live_view_clicked(self) -> None:
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        if self.reader is not None:
            self.reader.stop_channel_monitor()
            self.stop_timers()
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

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

    def reset_first_plot(self) -> None:
        """
        Resets the `first_plot` flag, ensuring the next plot behaves as the first one.
        """
        self.first_plot = True

    def reset_camera(self) -> None:
        """
        Resets plot view
        """
        if self.reader is not None:
            if self.plotter is not None:
                bounds = self.plotter.bounds
                self.plotter.set_position(bounds)

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
            self.plot_call_id.setText(f'{self.call_id_plot:d}')
            
    def update_image(self) -> None:
        """
        Redraws plots based on the configured update rate.

        Processes the image data according to main window settings, such as rotation
        and log transformations. Also sets initial min/max pixel values in the UI.
        """
        if self.reader is not None:
            self.call_id_plot +=1
            if self.reader.cache_images is not None and self.reader.cache_qx is not None:
                try:    
                    # Collect all cached data
                    flat_intensity = np.vstack([*self.reader.cache_images], dtype=np.float32).ravel()
                    qx = np.vstack([*self.reader.cache_qx], dtype=np.float32).ravel()
                    qy = np.vstack([*self.reader.cache_qy], dtype=np.float32).ravel()
                    qz = np.vstack([*self.reader.cache_qz], dtype=np.float32).ravel()

                    points = np.column_stack((
                        qx, qy, qz
                    ))

                    # First-time setup
                    if self.first_plot:
                        min = np.min(flat_intensity)
                        max = np.max(flat_intensity)
                        self.cloud = pyv.PolyData(points)
                        self.cloud['intensity'] = flat_intensity 

                        self.lut = pyv.LookupTable(cmap='viridis')
                        self.lut.scalar_range = (min, max)
                        self.lut.apply_opacity([0,1])
                        # self.lut.below_range_color = 'black
                        # self.lut.above_range_color = 'y'
                        
                        self.actor = self.plotter.add_mesh(
                            self.cloud,
                            scalars='intensity',
                            cmap=self.lut
                        )
                        self.first_plot = False
                    else:
                        self.plotter.mesh.points = points
                        self.cloud['intensity'] = flat_intensity
                        self.lut.scalar_range = (min, max)
                        self.actor.mapper.scalar_range = (min,max) #self.cloud.get_data_range('intensity')
                        self.lut.apply_opacity([0,1])
                    
                    self.plotter.show_bounds(xtitle='H Axis', ytitle='K Axis', ztitle='L Axis')
                    self.plotter.render()

                except Exception as e:
                    print(f"[Viewer] Failed to update 3D plot: {e}")

                    # if self.log_image.isChecked():
                    #         self.image = np.log1p(self.image + 1)
                    #         min_level = np.log1p(min_level + 1)
                    #         max_level = np.log1p(max_level + 1)
                    # if self.first_plot:
                    #     # Auto sets the max value based on first incoming image
                    #     self.max_setting_val.setValue(max_level)
                    #     self.min_setting_val.setValue(min_level)
                    #     self.first_plot = False
                    # else:
                # self.min_px_val.setText(f"{min_level:.2f}")
                # self.max_px_val.setText(f"{max_level:.2f}")

    # def update_min_max_setting(self) -> None:
    #     """
    #     Updates the min/max pixel levels in the Image Viewer based on UI settings.
    #     """
    #     min = self.min_setting_val.value()
    #     max = self.max_setting_val.value()
        # self.image_view.setLevels(min, max)
    
    def closeEvent(self, event):
        """
        Custom close event to clean up resources, including stat dialogs.

        Args:
            event (QCloseEvent): The close event triggered when the main window is closed.
        """
        # del self.stats_dialog # otherwise dialogs stay in memory
        super(HKLImageWindow,self).closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfigDialog()
    window.show()

    sys.exit(app.exec_())