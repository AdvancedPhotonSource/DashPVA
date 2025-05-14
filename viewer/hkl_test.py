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
            roi_config (str): Path to the ROI configuration file.
        """
        super(ConfigDialog,self).__init__()
        uic.loadUi('gui/hkl_viewer_setup.ui', self)
        self.setWindowTitle('PV Config')
        # initializing variables to pass to Image Viewer
        self.input_channel = ""
        self.roi_config =  ""
        # class can be prefilled with text
        self.init_ui()
        
        # Connecting signasl to 
        # self.btn_edit.clicked.connect(self.json_open_file_dialog)
        self.btn_browse.clicked.connect(self.browse_file_dialog)
        # self.btn_create.clicked.connect(self.new_pv_setup)
        self.btn_accept_reject.accepted.connect(self.dialog_accepted) 

    def init_ui(self) -> None:
        """
        Prefills text in the Line Editors for the user.
        """
        self.le_input_channel.setText(self.le_input_channel.text())
        self.le_roi_config.setText(self.le_roi_config.text())

    def browse_file_dialog(self) -> None:
        """
        Opens a file dialog to select the path to a TOML configuration file.
        """
        self.pvs_path, _ = QFileDialog.getOpenFileName(self, 'Select TOML Config', 'pv_configs', '*.toml (*.toml)')

        self.le_roi_config.setText(self.pvs_path)

    # def new_pv_setup(self) -> None:
    #     """
    #     Opens a new window for setting up a new PV configuration within the UI.
    #     """
    #     self.new_pv_setup_dialog = PVSetupDialog(parent=self, file_mode='w', path=None)
    
    # def edit_pv_setup(self) -> None:
    #     """
    #     Opens a window for editing an existing PV configuration.
    #     """
    #     if self.le_edit_file_path.text() != '':
    #         self.edit_pv_setup_dialog = PVSetupDialog(parent=self, file_mode='r+', path=self.pvs_path)
    #     else:
    #         print('file path empty')

    def dialog_accepted(self) -> None:
        """
        Handles the final step when the dialog's accept button is pressed.
        Starts the HKLImageWindow process with filled information.
        """
        self.input_channel = self.le_input_channel.text()
        self.roi_config = self.le_roi_config.text()
        if osp.isfile(self.roi_config) or (self.roi_config == ''):
            self.hkl_3d_viewer = HKLImageWindow(input_channel=self.input_channel,
                                            file_path=self.roi_config,) 
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
        self.setWindowTitle('Image Viewer with PVAaccess')
        self.show()
        # Initializing important variables
        self.reader = None
        self.image = None
        self.call_id_plot = 0
        self.first_plot = True
        self.image_is_transposed = False
        # self.rot_num = 0
        # self.rois: list[pg.ROI] = []
        # self.stats_dialog = {}
        # self.stats_data = {}
        self._input_channel = input_channel
        self.pv_prefix.setText(self._input_channel)
        self._file_path = file_path
        # Initializing but not starting timers so they can be reached by different functions
        self.timer_labels = QTimer()
        self.timer_plot = QTimer()
        # self.timer_rsm = QTimer()
        self.timer_labels.timeout.connect(self.update_labels)
        self.timer_plot.timeout.connect(self.update_image)
        # self.timer_plot.timeout.connect(self.update_rois)
        # For testing ROIs being sent from analysis window
        self.roi_x = 100
        self.roi_y = 200
        self.roi_width = 50
        self.roi_height = 50
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
        self.point_cloud = None
        self.actor = None
        self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')

        # plot = pg.PlotItem()        
        # self.image_view = pg.ImageView(view=plot)
        # self.viewer_layout.addWidget(self.image_view,1,1)
        # self.image_view.view.getAxis('left').setLabel(text='SizeY [pixels]')
        # self.image_view.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
        # second is a separate plot to show the horiontal avg of peaks in the image
        # self.horizontal_avg_plot = pg.PlotWidget()
        # self.horizontal_avg_plot.invertY(True)
        # self.horizontal_avg_plot.setMaximumWidth(175)
        # self.horizontal_avg_plot.setYLink(self.image_view.getView())
        # self.viewer_layout.addWidget(self.horizontal_avg_plot, 1,0)

        # Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        # self.btn_hkl_viewer.clicked.connect(self.start_hkl_viewer)
        # self.rbtn_C.clicked.connect(self.c_ordering_clicked)
        # self.rbtn_F.clicked.connect(self.f_ordering_clicked)
        # self.rotate90degCCW.clicked.connect(self.rotate_rois)
        # self.log_image.clicked.connect(self.reset_first_plot)
        # self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        # self.display_rois.stateChanged.connect(self.show_rois_checked)
        # self.plotting_frequency.valueChanged.connect(self.start_timers)
        # self.log_image.clicked.connect(self.update_image)
        self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.min_setting_val.valueChanged.connect(self.update_min_max_setting)
        # self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)

    def start_timers(self) -> None:
        """
        Starts timers for updating labels and plotting at specified frequencies.
        """
        self.timer_labels.start(int(1000/100))
        self.timer_plot.start(int(1000/self.plotting_frequency.value()))

    def stop_timers(self) -> None:
        """
        Stops the updating of main window labels and plots.
        """
        self.timer_plot.stop()
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
                                         config_filepath=self._file_path)
                self.set_pixel_ordering()
                self.reader.start_channel_monitor()
            else:
                self.stop_timers()
                self.reader.stop_channel_monitor()
                del self.reader
                # for roi in self.rois:
                #     pass
                    # self.image_view.getView().removeItem(roi)
                # self.rois = []
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path)
                self.set_pixel_ordering()
                self.reader.start_channel_monitor()
        except:
            print(f'Failed to Connect to {self._input_channel}')
            # self.image_view.clear()
            # self.horizontal_avg_plot.getPlotItem().clear()
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
        
        if self.reader is not None:
            # if not(self.reader.rois):
            #         if ('ROI' in self.reader.config):
            #             self.reader.start_roi_backup_monitor()
            # self.start_stats_monitors()
            # self.add_rois()
            self.start_timers()
            # try:
            #     self.init_hkl()
            #     if self.hkl_data:
            #         qxyz = self.create_rsm()
            #         self.qx = qxyz[0].T if self.image_is_transposed else qxyz[0]
            #         self.qy = qxyz[1].T if self.image_is_transposed else qxyz[1]
            #         self.qz = qxyz[2].T if self.image_is_transposed else qxyz[2]
            # except Exception as e:
            #     print('failed to create rsm: %s' % e)

    def stop_live_view_clicked(self) -> None:
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        if self.reader is not None:
            self.reader.stop_channel_monitor()
            self.stop_timers()
            # for key in self.stats_dialog:
            #     self.stats_dialog[key] = None
            # for roi in self.rois:
            #     pass
                # self.image_view.getView().removeItem(roi)
            # self.rois = []
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
  
    def show_rois_checked(self) -> None:
        """
        Toggles visibility of ROIs based on the checked state of the display checkbox.
        """
        if self.reader is not None:
            if self.display_rois.isChecked():
                for roi in self.rois:
                    roi.show()
            else:
                for roi in self.rois:
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

    def reset_first_plot(self) -> None:
        """
        Resets the `first_plot` flag, ensuring the next plot behaves as the first one.
        """
        self.first_plot = True

    def start_hkl_viewer(self) -> None:
        try:
            if self.reader is not None and 'HKL' in self.reader.config:
                qx = self.qx.flatten()
                qy = self.qy.flatten()
                qz = self.qz.flatten()
                intensity = self.reader.image.flatten()

                np.save('qx.npy', qx)
                np.save('qy.npy', qy)
                np.save('qz.npy', qz)
                np.save('intensity.npy', intensity)

                # cmd = ['python', 'viewer/hkl_3d_viewer.py',
                #        '--qx-file', 'qx.npy',
                #        '--qy-file', 'qy.npy',
                #        '--qz-file', 'qz.npy',
                #        '--intensity-file', 'intensity.npy']

                # process = subprocess.Popen(
                #     cmd,
                #     stdout=subprocess.PIPE,
                #     stderr=subprocess.STDOUT,
                #     preexec_fn=os.setsid,
                #     universal_newlines=True
                # )

                # self.processes[process.pid] = process

        except Exception as e:
            print(f'Failed to load HKL Viewer:{e}')


    def start_hkl_monitors(self) -> None:
        """
        Initializes camonitors for HKL values and stores them in a dictionary.
        """
        try:
            if "HKL" in self.reader.config:
                self.hkl_config = self.reader.config["HKL"]

                # Monitor each HKL parameter
                for section, pv_dict in self.hkl_config.items():
                    for key, pv_name in pv_dict.items():
                        self.hkl_data[pv_name] = caget(pv_name)
                        camonitor(pvname=pv_name, callback=self.hkl_ca_callback)
        except Exception as e:
            print(f"Failed to initialize HKL monitors: {e}")

    def hkl_ca_callback(self, pvname, value, **kwargs) -> None:
        """
        Callback for updating HKL values based on changes observed by `camonitor`.

        Args:
            pvname (str): The name of the PV that has been updated.
            value: The new value sent by the monitor for the PV.
            **kwargs: Additional keyword arguments sent by the monitor.
        """
        self.hkl_data[pvname] = value
        if self.qx is not None and self.qy is not None and self.qz is not None:
            self.update_rsm()

    def init_hkl(self) -> None:
        """
        Initializes HKL parameters by setting up monitors for each HKL value.
        """
        self.start_hkl_monitors()
        self.hkl_setup()
        
    def hkl_setup(self) -> None:
        pass
         
    def update_roi_region(self) -> None:
        """
        Forces the image viewer to refresh when an ROI region changes.
        """
        pass
        # self.image_view.update()

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
                # self.image = np.rot90(image, k=self.rot_num).T if self.image_is_transposed else np.rot90(image, k=self.rot_num)
                # if len(self.image.shape) == 2:
                # min_level, max_level = np.min(self.image), np.max(self.image)
                try:    
                    # Collect all cached data
                    flat_intensity = np.array(self.reader.cache_images).flatten()
                    # print('images:', self.reader.cache_images)
                    qx = np.array(self.reader.cache_qx).flatten()
                    # # print('qx:', self.reader.cache_qx)
                    qy = np.array(self.reader.cache_qy).flatten()
                    qz = np.array(self.reader.cache_qz).flatten()
                
                    flat_points = np.column_stack((
                        qx, qy, qz
                    ))

                    self.cloud = pyv.PolyData(flat_points)
                    self.cloud['intensity'] = flat_intensity                    

                    # First-time setup
                    if self.first_plot:
                        # self.lut = pyv.LookupTable()
                        # self.lut.cmap = 'viridis'
                        # self.lut.scalar_range=(np.min(flat_intensity), np.max(flat_intensity))
                        # self.lut.alpha_range=(0, 1)
                        self.actor = self.plotter.add_mesh(
                            self.cloud,
                            scalars='intensity',
                            cmap='viridis'
                            # point_size=10,
                        )
                        self.lut = self.actor.mapper.lookup_table
                        self.lut.scalar_range = (np.min(flat_intensity), np.max(flat_intensity))
                        self.lut.alpha_range = (0, 1)
                        # self.plotter.show_bounds(xtitle='H Axis', ytitle='K Axis', ztitle='L Axis')
                        self.plotter.render()
                        self.first_plot = False
                    else:
                        self.plotter.mesh.points = flat_points
                    #     # self.plotter.renderer.actors['hkl'](flat_intensity, render=False)
                    #     # self.plotter.render()
                    #     pass

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

    def update_min_max_setting(self) -> None:
        """
        Updates the min/max pixel levels in the Image Viewer based on UI settings.
        """
        min = self.min_setting_val.value()
        max = self.max_setting_val.value()
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