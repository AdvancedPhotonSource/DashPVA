import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5 import uic
import numpy as np
import h5py
from datetime import datetime
import time

class HDF5WriterThread(QThread):
    """
    A class that writes the image data to an HDF5 file using PyQt5.

    Keyword Args:
        file_written (pyqtSignal) -- A signal to indicate when the file is written.
        filename (str) -- Filename of the saved image file.
        images_cache (numpy.ndarray) -- A 3D numpy array that holds the images.
        scan_pos (dict) -- A dictionary of x and y positions of the image.
        metadata (dict) -- A dictionary of metadata.
        attributes (dict) -- A dictionary of attributes.
        intensity_values (numpy.ndarray) -- A 1D numpy array of intensity values.
        com_x_matrix (numpy.ndarray) -- A 2D numpy array of center of mass x values.
        com_y_matrix (numpy.ndarray) -- A 2D numpy array of center of mass y values.
    """
    # Signal to notify when writing is done
    # Has to be class level so PyQt can distinguish between signals and class attributes
    file_written = pyqtSignal()  
    
    def __init__(self, filename, images_cache, scan_pos, metadata, attributes, intensity_values, com_x_matrix, com_y_matrix):
        super(HDF5WriterThread, self).__init__()
        self.filename = filename
        self.images_cache = images_cache
        self.scan_pos = scan_pos
        self.metadata = metadata
        self.attributes = attributes
        self.intensity_values = intensity_values
        self.com_x_matrix = com_x_matrix
        self.com_y_matrix = com_y_matrix

    def run(self) -> None:
        """
        This function runs the thread which creates and saves the HDF5 file.
        """
        try:
            with h5py.File(self.filename, 'w') as h5file:
                #data
                h5file.create_group('/data')
                h5file.create_dataset('/data/images', data=self.images_cache)
                #scan pos
                h5file.create_group('/data/scan_pos')
                h5file.create_dataset('/data/scan_pos/x_positions', data=self.scan_pos['x_positions'])
                h5file.create_dataset('/data/scan_pos/y_positions', data=self.scan_pos['y_positions'])
                #save all metadata and attributes too. Expecting dictionaries
                h5file.create_group('/metadata')
                for key, value in self.metadata.items():
                    h5file['/metadata'].attrs[key] = value
                h5file.create_group('/attributes')
                for key, value in self.attributes.items():
                    h5file['/attributes'].attrs[key] = value
                #Analysis data 
                h5file.create_group('/analysis')
                h5file.create_dataset('/analysis/total_intensity', data=self.intensity_values)
                h5file.create_dataset('/analysis/com_x', data=self.com_x_matrix)
                h5file.create_dataset('/analysis/com_y', data=self.com_y_matrix)
                # Emit signal when file writing is done
                self.file_written.emit()  
        except Exception as e:
            print(e)



# Define the second window as a class
class AnalysisWindow(QMainWindow):
    """
    Main analysis window for visualizing and interacting with data.

    Attributes:
        parent (ImageWindow): Parent window object that contains image data and configurations.
        config (dict): Configuration settings from the parent.
        xpos_path (str): Path to the x-positions file.
        ypos_path (str): Path to the y-positions file.
        save_path (str): Path for saving HDF5 files.
        view_comx (pyqtgraph.ImageView): Image view for x center-of-mass if consumer type is vectorized.
        view_comy (pyqtgraph.ImageView): Image view for y center-of-mass if consumer type is vectorized.
        view_intensity (pyqtgraph.ImageView): Image view for intensity if consumer type is vectorized.
        plot_comx (pg.PlotWidget): Plot widget for x center-of-mass if consumer type is spontaneous.
        plot_comy (pg.PlotWidget): Plot widget for y center-of-mass if consumer type is spontaneous.
        plot_intensity (pg.PlotWidget): Plot widget for intensity if consumer type is spontaneous.
        update_counter (int): Counter for updates to plotting data.
        max_updates (int): Maximum number of updates allowed.
        analysis_index (int): Index for identifying analysis data from metadata.
        analysis_attributes (dict): Dictionary containing analysis attributes.
        timer_plot (QTimer): Timer for triggering plot updates.
    """
    
    def __init__(self, parent): 
        """
        Initializes the analysis window with parent and UI settings.

        Args:
            parent (ImageWindow): The parent window containing image and ROI data.
        """
        super(AnalysisWindow, self).__init__()
        self.parent = parent
        uic.loadUi('gui/analysis_window.ui', self)
        self.setWindowTitle('Analysis Window')
        # TODO: load config separately using the filepath provided by the parent
        self.config: dict = self.parent.reader.config
        self.consumer_type = self.config.get("CONSUMER_TYPE", "")
        self.xpos_path = None
        self.ypos_path = None
        self.save_path = None
        # for if widget is a ImageView
        self.view_intensity = None
        self.view_comx = None
        self.view_comy = None
        # for if widget is a Plot
        self.plot_intensity = None
        self.plot_comx = None
        self.plot_comy = None
        # configurations
        self.update_counter = 0
        self.max_updates = 10
        self.analysis_index = self.parent.reader.analysis_index
        if self.analysis_index is not None:
            self.analysis_attributes: dict = self.parent.reader.attributes[self.analysis_index] if self.consumer_type == "vectorized" else self.parent.reader.analysis_cache_dict 
        else:
            self.analysis_attributes = {}

        self.check_num_rois()
        self.configure_plots()

        self.timer_plot= QTimer()
        self.timer_plot.timeout.connect(self.plot_images)
        self.timer_plot.start(int(1000/self.calc_freq.value()))

        # self.btn_create_hdf5.clicked.connect(self.save_hdf5)
        self.calc_freq.valueChanged.connect(self.frequency_changed)
        self.cbox_select_roi.activated.connect(self.roi_selection_changed)
        self.chk_freeze.stateChanged.connect(self.freeze_plotting_checked)
        self.btn_reset.clicked.connect(self.reset_plot)
        self.sbox_intensity_min.valueChanged.connect(self.min_max_changed)
        self.sbox_intensity_max.valueChanged.connect(self.min_max_changed)
        self.sbox_comx_min.valueChanged.connect(self.min_max_changed)
        self.sbox_comx_max.valueChanged.connect(self.min_max_changed)
        self.sbox_comy_min.valueChanged.connect(self.min_max_changed)
        self.sbox_comy_max.valueChanged.connect(self.min_max_changed)

    def configure_plots(self) -> None:
        """
        Configures the plotting interface based on the consumer type.
        """
        # cmap = pg.colormap.getFromMatplotlib('viridis')
        if self.consumer_type == "spontaneous":
            self.init_scatter_plot()
        elif self.consumer_type == "vectorized":
            self.init_image_view()
        else:
            #TODO: replace with w/ a message box
            print("Config Not Set Up Correctly")

    def freeze_plotting_checked(self) -> None:
        """
        This function freezes the plot when the freeze plot checkbox is checked.
        """
        if self.chk_freeze.isChecked():
                self.timer_plot.stop()
        else:
            self.timer_plot.start(int(1000/self.calc_freq.value()))

    def check_if_running(self) -> None:
        """
        This function checks if the image scanning is running and sets the the status label's text.
        """
        if self.parent.reader.first_scan_detected:
            self.status_text.setText("Scanning...")
        else:
            self.status_text.setText("Waiting for the first scan...")

    def reset_plot(self) -> None:
        """
        This function resets the plot and clears all caches when the reset button is clicked.
        """
        # self.status_text.setText("Waiting for the first scan...")
        if self.consumer_type == "vectorized":
            self.view_intensity.clear()
            self.view_comx.clear()
            self.view_comy.clear()
        else:
            self.scatter_item_intensity.clear()
            self.scatter_item_comx.clear()
            self.scatter_item_comy.clear()
            self.parent.reader.analysis_cache_dict.update({"Intensity": {}})
            self.parent.reader.analysis_cache_dict.update({"ComX": {}})
            self.parent.reader.analysis_cache_dict.update({"Comy": {}})
            self.parent.reader.analysis_cache_dict.update({"Position": {}})

        self.timer_plot.start()
        self.update_counter = 0
        # Done because caching should be done from scratch
        self.parent.reader.frames_received = 0
        self.parent.reader.frames_missed = 0
        self.parent.start_timers()

    def check_num_rois(self) -> None:
        """
        Populates the dropdown menu with the number of available ROIs.
        """
        num_rois =  len(self.config.get('ROI'))
        if num_rois > 0:
            for i in range(num_rois):
                self.cbox_select_roi.addItem(f'ROI{i+1}')

    def roi_selection_changed(self) -> None:
        """
        Updates the ROI selection based on the user's choice in the dropdown.
        """
        text = self.cbox_select_roi.currentText()
        if text.startswith('ROI'):
            x = self.parent.reader.metadata[f"{self.parent.reader.pva_prefix}:{text}:MinX"]
            y = self.parent.reader.metadata[f"{self.parent.reader.pva_prefix}:{text}:MinY"]
            w= self.parent.reader.metadata[f"{self.parent.reader.pva_prefix}:{text}:SizeX"]
            h = self.parent.reader.metadata[f"{self.parent.reader.pva_prefix}:{text}:SizeY"]
            #change the roi values being analyzed
            self.parent.roi_x = x
            self.parent.roi_y = y
            self.parent.roi_width = w
            self.parent.roi_height = h
            # Make changes seen in the text boxes
            self.roi_x.setValue(x)
            self.roi_y.setValue(y)
            self.roi_width.setValue(w)
            self.roi_height.setValue(h)
        
    # def roi_boxes_changed(self) -> None:
    #     """
    #     This function is called when the ROI dimensions are changed.
    #     Changes the viewable roi to the values within the the boxes.
    #     """
    #     self.parent.roi_x = self.roi_x.value()
    #     self.parent.roi_y = self.roi_y.value()
    #     self.parent.roi_width = self.roi_width.value()
    #     self.parent.roi_height = self.roi_height.value()
    #     self.cbox_select_roi.setCurrentIndex(0)
    #     self.update_counter = 0
        
    def frequency_changed(self) -> None:
        """
        Adjusts the plot update frequency when the frequency spin box value changes.
        """
        self.timer_plot.start(int(1000/self.calc_freq.value()))

    def min_max_changed(self) -> None:
        """
        Updates the min and max values for intensity and center-of-mass image views/plots.
        """
        self.min_intensity = self.sbox_intensity_min.value()
        self.max_intensity = self.sbox_intensity_max.value()
        self.min_comx = self.sbox_comx_min.value()
        self.max_comx = self.sbox_comx_max.value()
        self.min_comy = self.sbox_comy_min.value()
        self.max_comy = self.sbox_comy_max.value()

        if self.config['CONSUMER_TYPE'] == 'spontaneous':
            self.plot_images()
        if self.config['CONSUMER_TYPE'] == 'vectorized':
            self.view_intensity.setLevels(self.min_intensity, self.max_intensity)
            self.view_comx.setLevels(self.min_comx, self.max_comx)
            self.view_comy.setLevels(self.min_comy, self.max_comy)

    def update_vectorized_image(self, intensity, com_x, com_y) -> None:
        """
        Updates the vectorized image views with intensity and center-of-mass data.

        Args:
            intensity (numpy.ndarray): Array representing intensity values.
            com_x (numpy.ndarray): Array representing center-of-mass x-values.
            com_y (numpy.ndarray): Array representing center-of-mass y-values.
        """
        size = int(np.sqrt(len(intensity)))
        intensity_matrix = np.reshape(intensity, (size, size))
        com_x_matrix = np.reshape(com_x,(size, size))
        com_y_matrix = np.reshape(com_y,(size, size))

        # USING IMAGE VIEW:
        if self.update_counter == self.max_updates:
            self.view_intensity.setImage(img=intensity_matrix.T, autoRange=False, autoLevels=False, levels=(self.min_intensity,self.max_intensity), autoHistogramRange=False)
            self.view_comx.setImage(img=com_x_matrix.T, autoRange=False, autoLevels=False, levels=(self.min_comx,self.max_comx), autoHistogramRange=False)
            self.view_comy.setImage(img=com_y_matrix.T, autoRange=False, autoLevels=False, levels=(self.min_comy,self.max_comy), autoHistogramRange=False)

            self.sbox_intensity_max.setValue(self.max_intensity)
            self.sbox_comx_max.setValue(self.max_comx)
            self.sbox_comy_max.setValue(self.max_comy)
        else:
            self.view_intensity.setImage(img=intensity_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False)
            self.view_comx.setImage(img=com_x_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False)
            self.view_comy.setImage(img=com_y_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False)

        
    def update_spontaneous_image(self, intensity, com_x, com_y, position) -> None:
        """
        Updates the scatter plots for spontaneous data.

        Args:
            intensity (list): List of intensity values.
            com_x (list): List of center-of-mass x-values.
            com_y (list): List of center-of-mass y-values.
            position (list): List of positions corresponding to the values.
        """
        intensity = np.array(intensity)
        com_x = np.array(com_x)
        com_y = np.array(com_y)
        position = np.array(position)

        # sets instead of no dots appearing if points are ourside fo min and max, clips them to be min and max values
        intensity_filtered = np.clip(intensity, self.min_intensity, self.max_intensity) 
        comx_filtered = np.clip(com_x, self.min_comx, self.max_comx)
        comy_filtered = np.clip(com_y, self.min_comy, self.max_comy)
        # normalization of data points based on min and max
        norm_intensity_colors = (intensity_filtered - self.min_intensity) / (self.max_intensity - self.min_intensity)
        norm_comx_colors = (comx_filtered - self.min_comx) / (self.max_comx - self.min_comx)
        norm_comy_colors = (comy_filtered - self.min_comy) / (self.max_comy - self.min_comy)
    
        cmap = pg.colormap.get("viridis.csv")

        # creating data for intensity plot
        intensity_brushes = [pg.mkBrush(cmap.map(color, mode='qcolor')) for color in norm_intensity_colors]
        intensity_spots = [{'pos':pos, 'brush': brush, 'size':10, 'symbol': 's'} for pos, brush in zip(position, intensity_brushes)]
        # creating data for com_x plot
        comx_brushes = [pg.mkBrush(cmap.map(color, mode='qcolor')) for color in norm_comx_colors]
        comx_spots = [{'pos':pos, 'brush': brush, 'size':10, 'symbol': 's'} for pos, brush in zip(position, comx_brushes)]
        # creating data for com_y plot
        comy_brushes = [pg.mkBrush(cmap.map(color, mode='qcolor')) for color in norm_comy_colors]
        comy_spots = [{'pos':pos, 'brush': brush, 'size':10, 'symbol': 's'} for pos, brush in zip(position, comy_brushes)]

        self.scatter_item_intensity.setData(intensity_spots)
        self.scatter_item_comx.setData(comx_spots)
        self.scatter_item_comy.setData(comy_spots)

    def plot_images(self) -> None:
        """
        Redraws plots based on the configured frequency.
        """
        if self.analysis_index is not None:

            self.update_counter += 1
            # TODO: test if this line can be assigned once and use update on its own
            if self.consumer_type == "vectorized":
                self.analysis_attributes = self.parent.reader.attributes[self.analysis_index] 
                intensity = self.analysis_attributes["value"][0]["value"].get("Intensity",0.0)
                com_x = self.analysis_attributes["value"][0]["value"].get("ComX",0.0)
                com_y = self.analysis_attributes["value"][0]["value"].get("ComY",0.0)
            elif self.consumer_type == "spontaneous":
                intensity = list(self.analysis_attributes["Intensity"].values())
                com_x = list(self.analysis_attributes["ComX"].values())
                com_y = list(self.analysis_attributes["ComY"].values())
                position = list(self.analysis_attributes["Position"].values())
                
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

                # print(intensity)
                if self.consumer_type == "vectorized":
                    self.update_vectorized_image(intensity=intensity, com_x=com_x, com_y=com_y,)
                elif self.consumer_type == "spontaneous":
                    self.update_spontaneous_image(intensity=intensity, com_x=com_x, com_y=com_y, position=position)  

    def init_scatter_plot(self) -> None:
        """
        Initializes scatter plots for intensity, com_x, and com_y data.
        called only if consumer type is spontaneous
        """
        self.scatter_item_intensity = pg.ScatterPlotItem()
        self.scatter_item_comx = pg.ScatterPlotItem()
        self.scatter_item_comy = pg.ScatterPlotItem()

        self.plot_intensity = pg.PlotWidget()
        self.plot_comx = pg.PlotWidget()
        self.plot_comy = pg.PlotWidget()

        self.plot_intensity.addItem(self.scatter_item_intensity)
        self.grid_a.addWidget(self.plot_intensity,0,0)
        self.plot_intensity.setLabel('bottom', 'Motor Position X' )
        self.plot_intensity.setLabel('left', 'Motor Posistion Y')
        self.plot_intensity.invertY(True)

        self.plot_comx.addItem(self.scatter_item_comx)
        self.grid_b.addWidget(self.plot_comx,0,0)
        self.plot_comx.setLabel('bottom', 'Motor Position X' )
        self.plot_comx.setLabel('left', 'Motor Posistion Y')
        self.plot_comx.invertY(True)

        self.plot_comy.addItem(self.scatter_item_comy)
        self.grid_c.addWidget(self.plot_comy,0,0)
        self.plot_comy.setLabel('bottom', 'Motor Position X' )
        self.plot_comy.setLabel('left', 'Motor Posistion Y')
        self.plot_comy.invertY(True)

    def init_image_view(self) -> None: 
        """
        Initializes Image Views for intensity, com_x, and com_y data.
        called only if cosumer type is vectorized
        """
        plot_item_intensity = pg.PlotItem()
        plot_item_comx = pg.PlotItem()
        plot_item_comy = pg.PlotItem()

        self.view_intensity = pg.ImageView(view=plot_item_intensity)
        self.view_comx = pg.ImageView(view=plot_item_comx)
        self.view_comy = pg.ImageView(view=plot_item_comy)

        self.grid_a.addWidget(self.view_intensity,0,0)
        self.view_intensity.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_intensity.view.getAxis('bottom').setLabel('Scan Position Cols')

        self.grid_b.addWidget(self.view_comx,0,0)
        self.view_comx.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_comx.view.getAxis('bottom').setLabel('Scan Position Cols')

        self.grid_c.addWidget(self.view_comy,0,0)
        self.view_comy.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_comy.view.getAxis('bottom').setLabel('Scan Position Cols')



    def closeEvent(self, event):
        """
        Handles cleanup operations when the analysis window is closed.

        Args:
            event (QCloseEvent): The close event triggered when the window is closed.
        """
        self.parent.start_timers()
        del self.parent.analysis_window
        event.accept()
        super(AnalysisWindow, self).closeEvent(event)


