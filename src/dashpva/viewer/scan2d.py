import sys

import numpy as np
import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow

import dashpva.settings as app_settings
from dashpva.gui import configure_app, ui_path
from dashpva.utils import PVAReader


class Scan2DWindow(QMainWindow):
    signal_start_monitor = pyqtSignal()

    def __init__(self, channel: str = ""):
        super().__init__()
        uic.loadUi(ui_path("scan2d.ui"), self)
        self.setWindowTitle('2D Scan Visualization')

        self.channel = channel or app_settings.get_input_channel("")
        self.config = app_settings.CONFIG or {}
        self.consumer_mode = app_settings.CONSUMER_MODE or "continuous"

        self.view_intensity = None
        self.view_comx = None
        self.view_comy = None
        self.plot_intensity = None
        self.plot_comx = None
        self.plot_comy = None

        self.update_counter = 0
        self.max_updates = 10
        self.analysis_index = None
        self.analysis_attributes = {}

        self.reader_thread = QThread()
        self.reader = None

        self.check_num_rois()
        self.configure_plots()

        self.timer_plot = QTimer()
        self.timer_plot.timeout.connect(self.plot_images)

        self.calc_freq.valueChanged.connect(self.frequency_changed)
        self.chk_freeze.stateChanged.connect(self.freeze_plotting_checked)
        self.btn_reset.clicked.connect(self.reset_plot)
        self.sbox_intensity_min.valueChanged.connect(self.min_max_changed)
        self.sbox_intensity_max.valueChanged.connect(self.min_max_changed)
        self.sbox_comx_min.valueChanged.connect(self.min_max_changed)
        self.sbox_comx_max.valueChanged.connect(self.min_max_changed)
        self.sbox_comy_min.valueChanged.connect(self.min_max_changed)
        self.sbox_comy_max.valueChanged.connect(self.min_max_changed)

        self.signal_start_monitor.connect(self._on_start_monitor)

        if self.channel:
            self._start_monitoring()

    def _start_monitoring(self):
        self.reader = PVAReader(input_channel=self.channel)
        self.reader.moveToThread(self.reader_thread)
        self.signal_start_monitor.connect(self.reader.start_channel_monitor)
        self.reader_thread.start()
        self.signal_start_monitor.emit()
        self.timer_plot.start(int(1000 / self.calc_freq.value()))
        self.status_text.setText(f"Listening: {self.channel}")

    def _on_start_monitor(self):
        if self.reader is not None:
            self.analysis_index = self.reader.analysis_index
            if self.analysis_index is not None:
                if self.consumer_mode == "vectorized":
                    self.analysis_attributes = self.reader.attributes[self.analysis_index]
                else:
                    self.analysis_attributes = self.reader.analysis_cache_dict

    def configure_plots(self):
        if self.consumer_mode == "continuous":
            self.init_scatter_plot()
        elif self.consumer_mode == "vectorized":
            self.init_image_view()

    def freeze_plotting_checked(self):
        if self.chk_freeze.isChecked():
            self.timer_plot.stop()
        else:
            self.timer_plot.start(int(1000 / self.calc_freq.value()))

    def reset_plot(self):
        if self.consumer_mode == "vectorized":
            if self.view_intensity:
                self.view_intensity.clear()
            if self.view_comx:
                self.view_comx.clear()
            if self.view_comy:
                self.view_comy.clear()
        else:
            if hasattr(self, 'scatter_item_intensity'):
                self.scatter_item_intensity.clear()
                self.scatter_item_comx.clear()
                self.scatter_item_comy.clear()
            if self.reader is not None:
                self.reader.analysis_cache_dict.update({
                    "Intensity": {}, "ComX": {}, "ComY": {}, "Position": {},
                })
                self.reader.frames_received = 0
                self.reader.frames_missed = 0

        self.timer_plot.start()
        self.update_counter = 0

    def check_num_rois(self):
        roi = self.config.get('ROI') or {}
        num_rois = len(roi)
        if num_rois > 0:
            for i in range(num_rois):
                self.cbox_select_roi.addItem(f'ROI{i + 1}')

    def frequency_changed(self):
        self.timer_plot.start(int(1000 / self.calc_freq.value()))

    def min_max_changed(self):
        self.min_intensity = self.sbox_intensity_min.value()
        self.max_intensity = self.sbox_intensity_max.value()
        self.min_comx = self.sbox_comx_min.value()
        self.max_comx = self.sbox_comx_max.value()
        self.min_comy = self.sbox_comy_min.value()
        self.max_comy = self.sbox_comy_max.value()

        if self.consumer_mode == 'continuous':
            self.plot_images()
        if self.consumer_mode == 'vectorized':
            if self.view_intensity:
                self.view_intensity.setLevels(self.min_intensity, self.max_intensity)
            if self.view_comx:
                self.view_comx.setLevels(self.min_comx, self.max_comx)
            if self.view_comy:
                self.view_comy.setLevels(self.min_comy, self.max_comy)

    def update_vectorized_image(self, intensity, com_x, com_y):
        size = int(np.sqrt(len(intensity)))
        intensity_matrix = np.reshape(intensity, (size, size))
        com_x_matrix = np.reshape(com_x, (size, size))
        com_y_matrix = np.reshape(com_y, (size, size))

        if self.update_counter == self.max_updates:
            self.view_intensity.setImage(
                img=intensity_matrix.T, autoRange=False, autoLevels=False,
                levels=(self.min_intensity, self.max_intensity), autoHistogramRange=False,
            )
            self.view_comx.setImage(
                img=com_x_matrix.T, autoRange=False, autoLevels=False,
                levels=(self.min_comx, self.max_comx), autoHistogramRange=False,
            )
            self.view_comy.setImage(
                img=com_y_matrix.T, autoRange=False, autoLevels=False,
                levels=(self.min_comy, self.max_comy), autoHistogramRange=False,
            )
            self.sbox_intensity_max.setValue(self.max_intensity)
            self.sbox_comx_max.setValue(self.max_comx)
            self.sbox_comy_max.setValue(self.max_comy)
        else:
            self.view_intensity.setImage(
                img=intensity_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False,
            )
            self.view_comx.setImage(
                img=com_x_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False,
            )
            self.view_comy.setImage(
                img=com_y_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False,
            )

    def update_continuous_image(self, intensity, com_x, com_y, position):
        intensity = np.array(intensity)
        com_x = np.array(com_x)
        com_y = np.array(com_y)
        position = np.array(position)

        intensity_filtered = np.clip(intensity, self.min_intensity, self.max_intensity)
        comx_filtered = np.clip(com_x, self.min_comx, self.max_comx)
        comy_filtered = np.clip(com_y, self.min_comy, self.max_comy)

        norm_intensity_colors = (intensity_filtered - self.min_intensity) / (self.max_intensity - self.min_intensity)
        norm_comx_colors = (comx_filtered - self.min_comx) / (self.max_comx - self.min_comx)
        norm_comy_colors = (comy_filtered - self.min_comy) / (self.max_comy - self.min_comy)

        cmap = pg.colormap.get("magma.csv")

        intensity_brushes = [pg.mkBrush(cmap.map(color, mode='qcolor')) for color in norm_intensity_colors]
        intensity_spots = [{'pos': pos, 'brush': brush, 'size': 10, 'symbol': 's'} for pos, brush in zip(position, intensity_brushes)]

        comx_brushes = [pg.mkBrush(cmap.map(color, mode='qcolor')) for color in norm_comx_colors]
        comx_spots = [{'pos': pos, 'brush': brush, 'size': 10, 'symbol': 's'} for pos, brush in zip(position, comx_brushes)]

        comy_brushes = [pg.mkBrush(cmap.map(color, mode='qcolor')) for color in norm_comy_colors]
        comy_spots = [{'pos': pos, 'brush': brush, 'size': 10, 'symbol': 's'} for pos, brush in zip(position, comy_brushes)]

        self.scatter_item_intensity.setData(intensity_spots)
        self.scatter_item_comx.setData(comx_spots)
        self.scatter_item_comy.setData(comy_spots)

    def plot_images(self):
        if self.reader is not None and self.analysis_index is None:
            self.analysis_index = self.reader.analysis_index
            if self.analysis_index is not None:
                if self.consumer_mode == "vectorized":
                    self.analysis_attributes = self.reader.attributes[self.analysis_index]
                else:
                    self.analysis_attributes = self.reader.analysis_cache_dict

        if self.analysis_index is None:
            return

        self.update_counter += 1

        if self.consumer_mode == "vectorized":
            self.analysis_attributes = self.reader.attributes[self.analysis_index]
            intensity = self.analysis_attributes["value"][0]["value"].get("Intensity", 0.0)
            com_x = self.analysis_attributes["value"][0]["value"].get("ComX", 0.0)
            com_y = self.analysis_attributes["value"][0]["value"].get("ComY", 0.0)
        elif self.consumer_mode == "continuous":
            intensity = list(self.analysis_attributes["Intensity"].values())
            com_x = list(self.analysis_attributes["ComX"].values())
            com_y = list(self.analysis_attributes["ComY"].values())
            position = list(self.analysis_attributes["Position"].values())
        else:
            return

        if len(intensity):
            if self.update_counter == 1:
                self.min_intensity = 0
                self.max_intensity = np.max(intensity)
                self.sbox_intensity_max.setValue(self.max_intensity)

                self.min_comx = 0
                self.max_comx = np.max(com_x)
                self.sbox_comx_max.setValue(self.max_comx)

                self.min_comy = 0
                self.max_comy = np.max(com_y)
                self.sbox_comy_max.setValue(self.max_comy)

            if self.consumer_mode == "vectorized":
                self.update_vectorized_image(intensity=intensity, com_x=com_x, com_y=com_y)
            elif self.consumer_mode == "continuous":
                self.update_continuous_image(intensity=intensity, com_x=com_x, com_y=com_y, position=position)

    def init_scatter_plot(self):
        self.scatter_item_intensity = pg.ScatterPlotItem()
        self.scatter_item_comx = pg.ScatterPlotItem()
        self.scatter_item_comy = pg.ScatterPlotItem()

        self.plot_intensity = pg.PlotWidget()
        self.plot_comx = pg.PlotWidget()
        self.plot_comy = pg.PlotWidget()

        self.plot_intensity.addItem(self.scatter_item_intensity)
        self.grid_a.addWidget(self.plot_intensity, 0, 0)
        self.plot_intensity.setLabel('bottom', 'Motor Position X')
        self.plot_intensity.setLabel('left', 'Motor Position Y')
        self.plot_intensity.invertY(True)

        self.plot_comx.addItem(self.scatter_item_comx)
        self.grid_b.addWidget(self.plot_comx, 0, 0)
        self.plot_comx.setLabel('bottom', 'Motor Position X')
        self.plot_comx.setLabel('left', 'Motor Position Y')
        self.plot_comx.invertY(True)

        self.plot_comy.addItem(self.scatter_item_comy)
        self.grid_c.addWidget(self.plot_comy, 0, 0)
        self.plot_comy.setLabel('bottom', 'Motor Position X')
        self.plot_comy.setLabel('left', 'Motor Position Y')
        self.plot_comy.invertY(True)

    def init_image_view(self):
        plot_item_intensity = pg.PlotItem()
        plot_item_comx = pg.PlotItem()
        plot_item_comy = pg.PlotItem()

        self.view_intensity = pg.ImageView(view=plot_item_intensity)
        self.view_comx = pg.ImageView(view=plot_item_comx)
        self.view_comy = pg.ImageView(view=plot_item_comy)

        self.grid_a.addWidget(self.view_intensity, 0, 0)
        self.view_intensity.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_intensity.view.getAxis('bottom').setLabel('Scan Position Cols')

        self.grid_b.addWidget(self.view_comx, 0, 0)
        self.view_comx.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_comx.view.getAxis('bottom').setLabel('Scan Position Cols')

        self.grid_c.addWidget(self.view_comy, 0, 0)
        self.view_comy.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_comy.view.getAxis('bottom').setLabel('Scan Position Cols')

    def closeEvent(self, event):
        self.timer_plot.stop()
        if self.reader_thread.isRunning():
            self.reader_thread.quit()
            self.reader_thread.wait(2000)
        event.accept()
        super().closeEvent(event)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='2D Scan Visualization')
    parser.add_argument('--channel', default='', help='PVA channel name')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    configure_app(app)
    window = Scan2DWindow(channel=args.channel)
    window.show()
    sys.exit(app.exec_())
