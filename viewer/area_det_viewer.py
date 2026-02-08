import os
import sys

# --- Stitcher environment (must be set before PVAReader/vit_stitch are imported) ---
# Set defaults only if not already set (user can export before running)
# Default CSV for VIT stitch — match LiveStitch7.py reference.
_default_csv = "/home/beams/AILEENLUO/ptycho-vit/workspace/positions_10um.csv"
if 'VIT_STITCH_POSITIONS_CSV' not in os.environ and os.path.exists(_default_csv):
    os.environ['VIT_STITCH_POSITIONS_CSV'] = _default_csv
if 'VIT_STITCH_STEP' not in os.environ:
    os.environ['VIT_STITCH_STEP'] = "0.05"
if 'VIT_STITCH_PIXEL_SIZE' not in os.environ:
    os.environ['VIT_STITCH_PIXEL_SIZE'] = "6.89e-9"
if 'VIT_STITCH_ID_OFFSET' not in os.environ:
    os.environ['VIT_STITCH_ID_OFFSET'] = "621356"

# Single consolidated print of VIT stitch env (all settable before running)
print("[DashPVA] To change VIT stitch, set before running:")
print(f"  export VIT_STITCH_POSITIONS_CSV='{os.environ.get('VIT_STITCH_POSITIONS_CSV', '')}'")
print(f"  export VIT_STITCH_ID_OFFSET='{os.environ.get('VIT_STITCH_ID_OFFSET', '621356')}'")
print(f"  export VIT_STITCH_STEP='{os.environ.get('VIT_STITCH_STEP', '0.05')}'")
print(f"  export VIT_STITCH_PIXEL_SIZE='{os.environ.get('VIT_STITCH_PIXEL_SIZE', '6.89e-9')}'")

# PVA network (vit setup): set before utils.PVAReader (pvaccess) is imported
if 'EPICS_PVA_ADDR_LIST' not in os.environ:
    os.environ['EPICS_PVA_ADDR_LIST'] = '10.54.116.22'
if 'EPICS_PVA_AUTO_ADDR_LIST' not in os.environ:
    os.environ['EPICS_PVA_AUTO_ADDR_LIST'] = 'NO'

import time
import subprocess
import numpy as np
import os.path as osp
import pyqtgraph as pg
import xrayutilities as xu
from PyQt5 import uic
# from epics import caget
from epics import PV, pv
from epics import camonitor, caget
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog, QFileDialog, QSlider, QCheckBox, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
# Custom imported classes
from roi_stats_dialog import RoiStatsDialog
from analysis_window import AnalysisWindow 
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import rotation_cycle
from utils import PVAReader #, HDF5Writer
# from ..utils.size_manager import SizeManager


rot_gen = rotation_cycle(1,5)         


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
        uic.loadUi('gui/pv_config.ui', self)
        self.setWindowTitle('PV Config')
        # initializing variables to pass to Image Viewer
        self.input_channel = ''
        self.config_path = ''
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
        self.le_config.setText(self.le_config.text())

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
        Starts the DiffractionImageWindow process with filled information.
        """
        self.input_channel = self.le_input_channel.text()
        self.config_path = self.le_config.text()
        if osp.isfile(self.config_path) or (self.config_path == ''):
            self.image_viewer = DiffractionImageWindow(input_channel=self.input_channel,
                                            file_path=self.config_path,) 
        else:
            print('File Path Doesn\'t Exitst')  
            #TODO: ADD ERROR Dialog rather than print message so message is clearer
            self.new_dialog = ConfigDialog()
            self.new_dialog.show()    


class DiffractionImageWindow(QMainWindow):
    hkl_data_updated = pyqtSignal(bool)

    def __init__(self, input_channel='vit:1:input_phase', file_path=''): 
        """
        Initializes the main window for real-time image visualization and manipulation.

        Args:
            input_channel (str): The PVA input channel for the detector.
            file_path (str): The file path for loading configuration.
        """
        super(DiffractionImageWindow, self).__init__()
        uic.loadUi('gui/imageshow.ui', self)
        self.setWindowTitle('DashPVA')
        self.show()
        # Initializing important variables
        self.reader = None
        self.image = None
        self.call_id_plot = 0
        self.first_plot = True
        self.image_is_transposed = False
        self.rot_num = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.rois: list[pg.ROI] = []
        self.stats_dialogs = {}
        self.stats_data = {}
        self._input_channel = input_channel
        self.pv_prefix.setText(self._input_channel)
        self._file_path = file_path

        # Initializing but not starting timers so they can be reached by different functions
        self.timer_labels = QTimer()
        self.timer_plot = QTimer()
        self.file_writer_thread = QThread()
        self.file_writer = None  # set when caches needed (e.g. HDF5Writer); vit:1:input_phase typically has no cache
        self.timer_labels.timeout.connect(self.update_labels)
        self.timer_plot.timeout.connect(self.update_image)
        self.timer_plot.timeout.connect(self.update_rois)

        # For testing ROIs being sent from analysis window
        self.roi_x = 100
        self.roi_y = 200
        self.roi_width = 50
        self.roi_height = 50

        # HKL values
        self.is_hkl_ready = False
        self.hkl_config = None
        self.hkl_pvs = {}
        self.hkl_data = {}
        self.q_conv = None
        self.qx = None
        self.qy = None
        self.qz = None
        self.processes = {}
        
        plot = pg.PlotItem()
        self.image_view = pg.ImageView(view=plot)
        # vit:1:input_phase — 5 panels: Transmission (left 3x3), Diffraction/Beam/Prediction (middle col), NN Stitched (right 3x3)
        self._is_vit_stitch = (self._input_channel == 'vit:1:input_phase')
        self.image_view_2 = None
        self.image_view_3 = None
        self.image_view_4 = None
        self.image_view_5 = None
        self.log_image_1 = None  # Log (diffraction only)
        if self._is_vit_stitch:
            # 7 columns x 3 rows: Transmission (0,0,3,3), Diffraction (0,3), Beam (1,3), Prediction (2,3), Stitched (0,4,3,3)
            self.viewer_layout.addWidget(self.image_view, 0, 0, 3, 3)  # Transmission
            plot2 = pg.PlotItem()
            self.image_view_2 = pg.ImageView(view=plot2)
            self.image_view_2.view.getAxis('left').setLabel(text='SizeY [pixels]')
            self.image_view_2.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
            try:
                self.image_view_2.getView().getViewBox().invertY(True)
            except Exception:
                pass
            self.viewer_layout.addWidget(self.image_view_2, 0, 3, 1, 1)  # Diffraction
            plot3 = pg.PlotItem()
            self.image_view_3 = pg.ImageView(view=plot3)
            self.image_view_3.view.getAxis('left').setLabel(text='SizeY [pixels]')
            self.image_view_3.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
            try:
                self.image_view_3.getView().getViewBox().invertY(True)
            except Exception:
                pass
            self.viewer_layout.addWidget(self.image_view_3, 1, 3, 1, 1)  # Beam position
            plot4 = pg.PlotItem()
            self.image_view_4 = pg.ImageView(view=plot4)
            self.image_view_4.view.getAxis('left').setLabel(text='SizeY [pixels]')
            self.image_view_4.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
            try:
                self.image_view_4.getView().getViewBox().invertY(True)
            except Exception:
                pass
            self.viewer_layout.addWidget(self.image_view_4, 2, 3, 1, 1)  # NN prediction
            plot5 = pg.PlotItem()
            self.image_view_5 = pg.ImageView(view=plot5)
            self.image_view_5.view.getAxis('left').setLabel(text='SizeY [pixels]')
            self.image_view_5.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
            try:
                self.image_view_5.getView().getViewBox().invertY(True)
            except Exception:
                pass
            self.viewer_layout.addWidget(self.image_view_5, 0, 4, 3, 3)  # NN Stitched
            self.viewer_layout.setRowStretch(0, 1)
            self.viewer_layout.setRowStretch(1, 1)
            self.viewer_layout.setRowStretch(2, 1)
            self.viewer_layout.setColumnStretch(0, 3)
            self.viewer_layout.setColumnStretch(3, 1)
            self.viewer_layout.setColumnStretch(4, 3)
        else:
            self.viewer_layout.addWidget(self.image_view, 0, 1)
        self.image_view.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
        if self._is_vit_stitch:
            # Log only for diffraction (panel 1)
            self.log_image_1 = QCheckBox('Log (diffraction)')
            img_layout = getattr(self, 'image_layout', None) or (getattr(self, 'formLayoutWidget', None) and self.formLayoutWidget.layout())
            if img_layout is not None:
                img_layout.insertRow(5, '', self.log_image_1)
            self.log_image_1.setChecked(True)
            self.log_image_1.stateChanged.connect(self.update_image)
            # Magma colormap for all five ImageViews
            self._vit_colormap = None
            try:
                self._vit_colormap = pg.colormap.get('magma')
            except Exception:
                try:
                    self._vit_colormap = pg.colormap.get('magma', source='colorcet')
                except Exception:
                    try:
                        self._vit_colormap = pg.colormap.get('Magma')
                    except Exception:
                        pass
            _vit_views = (self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5)
            if self._vit_colormap is not None:
                for view in _vit_views:
                    if view is not None and hasattr(view, 'setColorMap'):
                        view.setColorMap(self._vit_colormap)
                    elif view is not None and hasattr(view, 'imageItem'):
                        lut = self._vit_colormap.getLookupTable(nPts=256)
                        if lut is not None:
                            view.imageItem.setLookupTable(lut)
            self.btn_hkl_viewer.setVisible(False)
            self.btn_save_caches.setVisible(False)
            # Scale bar on NN Stitched panel (image_view_5)
            self._vit_scale_bar_line = None
            self._vit_scale_bar_text = None
            self._vit_scale_bar_added = False
        # Horizontal avg 1D plot (hidden when vit:1:input_phase)
        self.horizontal_avg_plot = pg.PlotWidget()
        self.horizontal_avg_plot.invertY(True)
        self.horizontal_avg_plot.setMaximumWidth(200)
        self.horizontal_avg_plot.getAxis('bottom').setLabel(text='Horizontal Avg.')
        if not self._is_vit_stitch:
            self.viewer_layout.addWidget(self.horizontal_avg_plot, 0, 0)

        # Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.btn_analysis_window.clicked.connect(self.open_analysis_window_clicked)
        self.btn_hkl_viewer.clicked.connect(self.start_hkl_viewer_clicked)
        self.btn_Stats1.clicked.connect(self.stats_button_clicked)
        self.btn_Stats2.clicked.connect(self.stats_button_clicked)
        self.btn_Stats3.clicked.connect(self.stats_button_clicked)
        self.btn_Stats4.clicked.connect(self.stats_button_clicked)
        self.btn_Stats5.clicked.connect(self.stats_button_clicked)
        self.rbtn_C.clicked.connect(self.c_ordering_clicked)
        self.rbtn_F.clicked.connect(self.f_ordering_clicked)
        self.rotate90degCCW.clicked.connect(self.rotation_count)
        # self.rotate90degCCW.clicked.connect(self.rotate_rois)
        self.log_image.clicked.connect(self.update_image)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.display_rois.stateChanged.connect(self.show_rois_checked)
        self.chk_transpose.stateChanged.connect(self.transpose_image_checked)
        self.plotting_frequency.valueChanged.connect(self.start_timers)
        self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.min_setting_val.valueChanged.connect(self.update_min_max_setting)
        btn_autoscale = getattr(self, "btn_autoscale", None)
        if btn_autoscale is None:
            btn_autoscale = QPushButton("Autoscale (5–95%)")
            btn_autoscale.clicked.connect(self.apply_autoscale)
            if hasattr(self, "formLayoutWidget_6") and self.formLayoutWidget_6.layout() is not None:
                self.formLayoutWidget_6.layout().addRow(btn_autoscale)
            else:
                parent = self.min_setting_val.parent()
                if parent is not None and parent.layout() is not None:
                    parent.layout().addWidget(btn_autoscale)
        else:
            btn_autoscale.clicked.connect(self.apply_autoscale)
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)
        self.hkl_data_updated.connect(self.handle_hkl_data_update)

    def _get_vit_view_levels(self, view) -> tuple:
        """Get current min/max levels from an ImageView so we can preserve user-set scale. Returns (min, max) or None."""
        if view is None:
            return None
        try:
            if getattr(view, 'ui', None) is not None and hasattr(view.ui, 'histogram'):
                return view.ui.histogram.getLevels()
            if hasattr(view, 'imageItem') and view.imageItem is not None:
                lev = getattr(view.imageItem, 'levels', None)
                if lev is not None and len(lev) == 2:
                    return tuple(lev)
                if callable(getattr(view.imageItem, 'getLevels', None)):
                    return view.imageItem.getLevels()
        except Exception:
            pass
        return None

    def _apply_vit_colormap(self) -> None:
        """Re-apply magma colormap to all five ImageViews. Call after setImage so LUT sticks."""
        if not getattr(self, '_vit_colormap', None):
            return
        for view in (self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5):
            if view is None:
                continue
            try:
                if hasattr(view, 'setColorMap'):
                    view.setColorMap(self._vit_colormap)
                elif hasattr(view, 'imageItem'):
                    lut = self._vit_colormap.getLookupTable(nPts=256)
                    if lut is not None:
                        view.imageItem.setLookupTable(lut)
            except Exception:
                pass

    def _add_vit_scale_bar(self, height: int, width: int, view=None) -> None:
        """Add a length scale bar (µm) on NN Stitched panel (image_view_5), bottom-left."""
        if getattr(self, "_vit_scale_bar_added", False):
            return
        target = view if view is not None else self.image_view_5
        if target is None:
            return
        try:
            pixel_size_m = float(os.environ.get("VIT_STITCH_PIXEL_SIZE", "6.89e-9"))
            # Bar length: default 1 µm (1000 nm); 1 pixel = pixel_size_m (m)
            # bar_px = (bar_nm * 1e-9) / pixel_size_m
            bar_nm = 1000.0  # 1 µm
            bar_px = (bar_nm * 1e-9) / pixel_size_m
            if bar_px > 0.45 * width:
                bar_nm = 500.0
                bar_px = (bar_nm * 1e-9) / pixel_size_m
            if bar_px < 20:
                bar_nm = 1000.0  # 1 µm
                bar_px = (bar_nm * 1e-9) / pixel_size_m
            margin = 20
            y_bar = height - 1 - margin
            x0, x1 = margin, margin + int(bar_px)
            color = "#00ffff"
            pen = pg.mkPen(color, width=2)
            self._vit_scale_bar_line = pg.PlotDataItem(
                x=[x0, x1], y=[y_bar, y_bar], pen=pen
            )
            self._vit_scale_bar_line.setZValue(100)
            target.view.addItem(self._vit_scale_bar_line)
            bar_label = "1 µm" if bar_nm >= 1000 else f"{bar_nm:.0f} nm"
            self._vit_scale_bar_text = pg.TextItem(
                text=bar_label, color=color, anchor=(0, 1)
            )
            self._vit_scale_bar_text.setZValue(100)
            self._vit_scale_bar_text.setPos(x0, y_bar - 4)
            target.view.addItem(self._vit_scale_bar_text)
            self._vit_scale_bar_added = True
        except Exception:
            self._vit_scale_bar_added = True  # avoid retry

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
        if self.file_writer is None:
            return
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

    def open_analysis_window_clicked(self) -> None:
        """
        Opens the unified Analysis/Status window.
        """
        # Create the window if it doesn't exist
        if not hasattr(self, 'analysis_window') or self.analysis_window is None:
            self.analysis_window = AnalysisWindow(parent=None) # Parent=None to make it a separate floating window
            
        self.analysis_window.show()
        self.analysis_window.raise_()
        self.analysis_window.activateWindow()       

    def start_live_view_clicked(self) -> None:
        """
        Initializes the connections to the PVA channel using the provided Channel Name.
        
        This method ensures that any existing connections are cleared and re-initialized.
        Also starts monitoring the stats and adds ROIs to the viewer.
        """
        try:
            self.stop_timers()
            self.image_view.clear()
            if self.image_view_2 is not None:
                self.image_view_2.clear()
            if self.image_view_3 is not None:
                self.image_view_3.clear()
            if self.image_view_4 is not None:
                self.image_view_4.clear()
            if self.image_view_5 is not None:
                self.image_view_5.clear()
            if self._is_vit_stitch:
                self._vit_last_autoscale_log_state = None
                self._vit_scale_bar_added = False
            self.reset_rsm_vars()
            if self.reader is None:
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path)
                if self.file_writer is not None:
                    self.file_writer.moveToThread(self.file_writer_thread)
            else:
                if self.reader.channel.isMonitorActive():
                    self.reader.stop_channel_monitor()
                if self.file_writer_thread.isRunning():
                    self.file_writer_thread.quit()
                    self.file_writer_thread.wait()
                for roi in self.rois:
                    self.image_view.getView().removeItem(roi)
                self.rois.clear()
                if self.file_writer is not None:
                    self.btn_save_caches.clicked.disconnect()
                    self.file_writer.hdf5_writer_finished.disconnect()
                del self.reader
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path)
                if self.file_writer is not None:
                    self.file_writer.pva_reader = self.reader
            # Reconnecting signals
            self.reader.reader_scan_complete.connect(self.trigger_save_caches)
            if self.file_writer is not None:
                self.file_writer.hdf5_writer_finished.connect(self.on_writer_finished)
                self.btn_save_caches.clicked.connect(self.save_caches_clicked)

            if self.reader.CACHING_MODE == 'scan' and self.file_writer is not None:
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
            self.image_view.clear()
            if self.image_view_2 is not None:
                self.image_view_2.clear()
            if self.image_view_3 is not None:
                self.image_view_3.clear()
            if self.image_view_4 is not None:
                self.image_view_4.clear()
            if self.image_view_5 is not None:
                self.image_view_5.clear()
            self.horizontal_avg_plot.getPlotItem().clear()
            self.reset_rsm_vars()
            if getattr(self, 'reader', None) is not None:
                del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
            if getattr(self, 'file_writer', None) is not None:
                try:
                    self.btn_save_caches.clicked.disconnect()
                except Exception:
                    pass
                del self.file_writer
                self.file_writer = None
            if self.file_writer_thread.isRunning():
                self.file_writer_thread.quit()
                self.file_writer_thread.wait()
        
        try:
            if self.reader is not None:
                if not(self.reader.rois):
                    if 'ROI' in self.reader.config:
                        self.reader.start_roi_backup_monitor()
                    self.start_hkl_monitors()
                    self.start_stats_monitors()
                    self.add_rois()
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
            # for roi in self.rois:
            #     self.image_view.getView().removeItem(roi)
            for hkl_pv in self.hkl_pvs.values():
                hkl_pv.clear_callbacks()
                hkl_pv.disconnect()
            self.hkl_pvs = {}
            self.hkl_data = {}
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

    def start_hkl_viewer_clicked(self) -> None:
        try:
            if self.reader is not None and self.reader.HKL_IN_CONFIG:
                cmd = ['python', 'viewer/hkl_3d_viewer.py',]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    universal_newlines=True
                )
                self.processes[process.pid] = process

        except Exception as e:
            print(f'[Diffraction Image Viewer] Failed to load HKL Viewer:{e}')

    def save_caches_clicked(self) -> None:
        if self.file_writer is None:
            return
        if not self.reader.channel.isMonitorActive():  
            if not self.file_writer_thread.isRunning():
                self.file_writer_thread.start()
            self.file_writer.save_caches_to_h5(clear_caches=True)
        else:
            QMessageBox.critical(None,
                                'Error',
                                'Stop Live View to Save Cache',
                                QMessageBox.Ok)

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

    def start_stats_monitors(self)  -> None:
        """
        Initializes monitors for updating stats values.

        This method uses `camonitor` to observe changes in the stats PVs and update
        them in the UI accordingly.
        """
        try:
            if self.reader.stats:
                for stat_num in self.reader.stats.keys():
                    for stat in self.reader.stats[stat_num].keys():
                        pv = f"{self.reader.stats[stat_num][stat]}"
                        self.stats_data[pv] = caget(pv)
                        camonitor(pvname=pv, callback=self.stats_ca_callback)
        except Exception as e:
            print(f"[Diffraction Image Viewer] Failed to Connect to Stats CA Monitors: {e}")

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
        """
        Adds ROIs to the image viewer and assigns them color codes.

        Color Codes:
            ROI1 -- Red (#ff0000)
            ROI2 -- Blue (#0000ff)
            ROI3 -- Green (#4CBB17)
            ROI4 -- Pink (#ff00ff)
        """
        try:
            roi_colors = ['#ff0000', '#0000ff', '#4CBB17', '#ff00ff']  
            # TODO: can just loop through values rather than lookup with keys
            for roi_num, roi in self.reader.rois.items():
                x = roi.get("MinX", 0) if not(self.image_is_transposed) else roi.get('MinY',0)
                y = roi.get("MinY", 0) if not(self.image_is_transposed) else roi.get('MinX',0)
                width = roi.get("SizeX", 0) if not(self.image_is_transposed) else roi.get('SizeY',0)
                height = roi.get("SizeY", 0) if not(self.image_is_transposed) else roi.get('SizeX',0)
                roi_color = int(roi_num[-1]) - 1 
                roi = pg.ROI(pos=[x,y],
                            size=[width, height],
                            movable=False,
                            pen=pg.mkPen(color=roi_colors[roi_color]))
                self.rois.append(roi)
                self.image_view.addItem(roi)
                roi.sigRegionChanged.connect(self.update_roi_region)
        except Exception as e:
            print(f'[Diffraction Image Viewer] Failed to add ROIs:{e}')


    def start_hkl_monitors(self) -> None:
        """
        Initializes camonitors for HKL values and stores them in a dictionary.
        """
        try:
            if self.reader.HKL_IN_CONFIG:
                self.hkl_config = self.reader.config["HKL"]
                if not self.hkl_pvs:
                    for section, pv_dict in self.hkl_config.items():
                        for section_key, pv_name in pv_dict.items():
                            if pv_name not in self.hkl_pvs:
                                self.hkl_pvs[pv_name] = PV(pvname=pv_name)
                for pv_name, pv_obj in self.hkl_pvs.items():
                    self.hkl_data[pv_name] = pv_obj.get() # Get current value
                    pv_obj.add_callback(callback=self.hkl_ca_callback)
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
        if self.qx is not None and self.qy is not None and self.qz is not None:
            self.hkl_data_updated.emit(True)

    def handle_hkl_data_update(self):
        if self.reader is not None and not self.stop_hkl.isChecked() and self.hkl_data:
            self.hkl_setup()
            if self.q_conv is None:
                raise ValueError("QConversion object is not initialized.")
            self.update_rsm()
  
                
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
                hxrd = xu.HXRD(self.inplane_reference_directions,
                            self.sample_surface_normal_directions, 
                            en=self.energy, 
                            qconv=self.q_conv)

                roi = [0, self.reader.shape[0], 0, self.reader.shape[1]]
                cch1 = self.hkl_data['DetectorSetup:CenterChannelPixel'][0] # Center Channel Pixel 1
                cch2 = self.hkl_data['DetectorSetup:CenterChannelPixel'][1] # Center Channel Pixel 2
                distance = self.hkl_data['DetectorSetup:Distance'] # Distance
                pixel_dir1 = self.hkl_data['DetectorSetup:PixelDirection1'] # Pixel Direction 1
                pixel_dir2 = self.hkl_data['DetectorSetup:PixelDirection2'] # PIxel Direction 2
                nch1 = self.reader.shape[0] # Number of detector pixels along direction 1
                nch2 = self.reader.shape[1] # Number of detector pixels along direction 2
                pixel_width1 = self.hkl_data['DetectorSetup:Size'][0] / nch1 # width of a pixel along direction 1
                pixel_width2 = self.hkl_data['DetectorSetup:Size'][1] / nch2 # width of a pixel along direction 2

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
        for roi, roi_dict in zip(self.rois, self.reader.rois.values()):
            x_pos = roi_dict.get("MinX",0) if not(self.image_is_transposed) else roi_dict.get('MinY',0)
            y_pos = roi_dict.get("MinY",0) if not(self.image_is_transposed) else roi_dict.get('MinX',0)
            width = roi_dict.get("SizeX",0) if not(self.image_is_transposed) else roi_dict.get('SizeY',0)
            height = roi_dict.get("SizeY",0) if not(self.image_is_transposed) else roi_dict.get('SizeX',0)
            roi.setPos(pos=x_pos, y=y_pos)
            roi.setSize(size=(width, height))
        self.image_view.update()

    def update_roi_region(self) -> None:
        """
        Forces the image viewer to refresh when an ROI region changes.
        """
        self.image_view.update()

    def update_pv_prefix(self) -> None:
        """
        Updates the input channel prefix based on the value entered in the prefix field.
        """
        self._input_channel = self.pv_prefix.text()
    
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
                        self.mouse_px_val.setText(f'{self.image[self.mouse_x][self.mouse_y]}')
                        if self.qx is not None and len(self.qx) > 0:
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
            self.roi1_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats1:Total_RBV', '0.0')}")
            self.roi2_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats2:Total_RBV', '0.0')}")
            self.roi3_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats3:Total_RBV', '0.0')}")
            self.roi4_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats4:Total_RBV', '0.0')}")
            self.stats5_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats5:Total_RBV', '0.0')}")

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
        For vit:1:input_phase, updates five ImageViews (transmission, diffraction, beam, prediction, stitched).
        Log scale only for diffraction (panel 1).
        """
        if self.reader is not None:
            self.call_id_plot += 1
            vit_panels = getattr(self.reader, 'vit_panels', None)
            if vit_panels is not None and len(vit_panels) == 5 and self.image_view_2 is not None:
                # 5 panels: [transmission, diffraction, beam_position, nn_prediction, nn_stitched]
                panels = [np.asarray(p, dtype=np.float32).copy() for p in vit_panels]
                panels[1] = np.maximum(panels[1], 0.0)  # diffraction >= 0
                log_diffraction = self.log_image_1.isChecked() if self.log_image_1 else True
                views = [self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5]
                for i, p in enumerate(panels):
                    if i == 1:
                        p = np.transpose(p)  # diffraction: transpose before display
                    if i in (0, 2, 4):
                        p = np.transpose(p)  # transmission, beam, stitched: swap x/y for correct raster scan order
                    p = np.transpose(p) if self.image_is_transposed else p
                    p = np.rot90(p, k=self.rot_num)
                    if i == 4:
                        p = p[:, ::-1].copy()  # stitched panel only: flip x axis on top of transpose
                    if i == 1 and log_diffraction:
                        p = np.maximum(p, 0)
                        p = np.log10(p + 1e-10)
                        p = np.maximum(p, 0.0)
                    panels[i] = p
                self._add_vit_scale_bar(panels[4].shape[0], panels[4].shape[1], view=self.image_view_5)
                last = getattr(self, "_vit_last_autoscale_log_state", None)
                run_for_panel = [True] * 5
                if last is not None and last != log_diffraction:
                    run_for_panel[1] = True
                if last is None or last != log_diffraction:
                    self._apply_autoscale_vit(panels, views, run_for_panel=run_for_panel)
                    self._vit_last_autoscale_log_state = log_diffraction
                for i, p in enumerate(panels):
                    view = views[i]
                    pmin, pmax = float(np.min(p)), float(np.max(p))
                    levels = self._get_vit_view_levels(view)
                    if levels is not None and len(levels) == 2 and levels[1] > levels[0]:
                        view.setImage(p, autoRange=False, autoLevels=False, levels=levels, autoHistogramRange=False)
                    elif pmin == pmax:
                        levels = (pmin, pmin + 1.0) if np.isfinite(pmin) else (0.0, 1.0)
                        view.setImage(p, autoRange=False, autoLevels=False, levels=levels, autoHistogramRange=False)
                    else:
                        view.setImage(p, autoRange=False, autoLevels=False, levels=(pmin, pmax), autoHistogramRange=False)
                    view.setVisible(True)
                for i, v in enumerate(views):
                    try:
                        ar = panels[i].shape[1] / float(max(panels[i].shape[0], 1))
                        v.view.setAspectLocked(lock=True, ratio=ar)
                    except Exception:
                        pass
                self.image = panels[1]
                mn, mx = np.min(panels[1]), np.max(panels[1])
                self.min_px_val.setText(f"{mn:.2f}")
                self.max_px_val.setText(f"{mx:.2f}")
                return
            elif self._is_vit_stitch and (vit_panels is None or len(vit_panels) != 5):
                pass  # not yet 5 panels
            if self.reader.CACHING_MODE in ['', 'alignment', 'scan']:
                self.image = self.reader.image
            elif self.reader.CACHING_MODE == 'bin':
                index = self.slider.value()
                self.image = np.reshape(np.mean(np.stack(self.reader.cached_images[index]), axis=0), self.reader.shape) # (self.reader.shape)

            if self.image is not None:
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
                        # Auto sets the max value based on first incoming image
                        self.max_setting_val.setValue(max_level)
                        self.min_setting_val.setValue(min_level)
                        self.first_plot = False
                    else:
                        self.image_view.setImage(self.image,
                                                autoRange=False,
                                                autoLevels=False,
                                                autoHistogramRange=False)
                # Separate image update for horizontal average plot
                self.horizontal_avg_plot.plot(x=np.mean(self.image, axis=0),
                                            y=np.arange(self.image.shape[1]),
                                            clear=True)

                self.min_px_val.setText(f"{min_level:.2f}")
                self.max_px_val.setText(f"{max_level:.2f}")
    
    def _percentile_levels(self, intensities: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> tuple:
        """
        Compute percentile range from the intensity population. Excludes log(0) floor
        (values <= -9) so 5th/95th are from the real distribution. Ensures lo <= data min
        so negative values are never clipped.
        """
        intensities = np.asarray(intensities).flatten()
        intensities = intensities[np.isfinite(intensities)]
        if len(intensities) == 0:
            return (0.0, 1.0)
        data_min = float(np.min(intensities))
        data_max = float(np.max(intensities))
        # In log scale, log10(1e-10) = -10; exclude that floor so percentiles use real population
        if data_min < -9.0:
            population = intensities[intensities > -9.0]
            if len(population) >= 100:
                intensities = population
        lo = float(np.percentile(intensities, p_lo))
        hi = float(np.percentile(intensities, p_hi))
        # Include full data range: never clip valid negatives (lo must be <= data min)
        if data_min < lo:
            lo = data_min
        if hi < data_max:
            hi = data_max
        if lo >= hi:
            hi = lo + 1.0
        return (lo, hi)

    def _apply_autoscale_vit(self, panels: list, views: list, run_for_panel: list = None) -> None:
        """
        Sets 5th/95th percentile levels for VIT panels. If run_for_panel is given, only
        updates levels for views where run_for_panel[i] is True. Min/max spinboxes track diffraction (panel 1).
        """
        if run_for_panel is None:
            run_for_panel = [True] * len(panels)
        for i, p in enumerate(panels):
            if i >= len(run_for_panel) or i >= len(views) or not run_for_panel[i]:
                continue
            intensities = p.flatten()
            intensities = intensities[np.isfinite(intensities)]
            if len(intensities) == 0:
                continue
            lo, hi = self._percentile_levels(intensities)
            views[i].setLevels(lo, hi)
        if len(panels) > 1 and len(run_for_panel) > 1 and run_for_panel[1]:
            intensities = panels[1].flatten()
            intensities = intensities[np.isfinite(intensities)]
            if len(intensities) > 0:
                lo, hi = self._percentile_levels(intensities)
                self.min_setting_val.blockSignals(True)
                self.max_setting_val.blockSignals(True)
                self.min_setting_val.setValue(lo)
                self.max_setting_val.setValue(hi)
                self.min_setting_val.blockSignals(False)
                self.max_setting_val.blockSignals(False)

    def apply_autoscale(self) -> None:
        """
        Sets min/max from 5th and 95th percentiles. In VIT 5-panel mode applies to all five;
        min/max spinboxes control diffraction (panel 1). Only runs when called (e.g. button).
        """
        if self._is_vit_stitch and self.image_view_2 is not None:
            vit_panels = getattr(self.reader, "vit_panels", None) if self.reader else None
            if vit_panels is not None and len(vit_panels) == 5:
                panels = [np.asarray(p, dtype=np.float32).copy() for p in vit_panels]
                panels[1] = np.maximum(panels[1], 0.0)
                log_diffraction = self.log_image_1.isChecked() if self.log_image_1 else True
                views = [self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5]
                for i, p in enumerate(panels):
                    if i == 1:
                        p = np.transpose(p)
                    if i in (0, 2, 4):
                        p = np.transpose(p)  # transmission, beam, stitched: swap x/y for raster order
                    p = np.transpose(p) if self.image_is_transposed else p
                    p = np.rot90(p, k=self.rot_num)
                    if i == 4:
                        p = p[:, ::-1].copy()  # stitched panel only: flip x axis on top of transpose
                    if i == 1 and log_diffraction:
                        p = np.maximum(p, 0)
                        p = np.log10(p + 1e-10)
                        p = np.maximum(p, 0.0)
                    panels[i] = p
                self._apply_autoscale_vit(panels, views)
                self._vit_last_autoscale_log_state = log_diffraction
                return
        if self.image is None:
            return
        intensities = self.image.flatten()
        intensities = intensities[np.isfinite(intensities)]
        if len(intensities) == 0:
            return
        min_percentile, max_percentile = self._percentile_levels(intensities)
        self.min_setting_val.blockSignals(True)
        self.max_setting_val.blockSignals(True)
        self.min_setting_val.setValue(min_percentile)
        self.max_setting_val.setValue(max_percentile)
        self.min_setting_val.blockSignals(False)
        self.max_setting_val.blockSignals(False)
        self.image_view.setLevels(min_percentile, max_percentile)

    def update_min_max_setting(self) -> None:
        """
        Updates the min/max pixel levels in the Image Viewer based on UI settings.
        For vit 5-panel mode the min/max spinboxes affect only the diffraction panel (image_view_2).
        """
        min_ = self.min_setting_val.value()
        max_ = self.max_setting_val.value()
        if self._is_vit_stitch and self.image_view_2 is not None:
            self.image_view_2.setLevels(min_, max_)
        else:
            self.image_view.setLevels(min_, max_)
    
    def closeEvent(self, event):
        """
        Custom close event to clean up resources, including stat dialogs.

        Args:
            event (QCloseEvent): The close event triggered when the main window is closed.
        """
        del self.stats_dialogs # otherwise dialogs stay in memory
        if self.file_writer_thread.isRunning():
            self.file_writer_thread.quit()
            self.file_writer_thread
        super(DiffractionImageWindow,self).closeEvent(event)

def main():
    app = QApplication(sys.argv)
    # size_manager = SizeManager(app=app)
    window = ConfigDialog()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()