import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5 import uic
import numpy as np
import h5py
from datetime import datetime
import time
from generators import rotation_cycle
# from area_det_viewer import ImageWindow

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

    def run(self):
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
        except Exception as e:
            print(e)

        self.file_written.emit()  # Emit signal when file writing is done


# Define the second window as a class
class AnalysisWindow(QMainWindow):
    """
    A class that displays and handles user interaction with the analysis window.

    Attributes:
        status_text (QLabel): A GUI label that shows the status of the analysis.
        parent (ImageWindow): The parent window object.
        xpos_path (str): The path to the x-positions file.
        ypos_path (str): The path to the y-positions file.
        save_path (str): The path to where the HDF5 file will be saved.
        timer_plot (QTimer): A timer to control the plot frequency.
        call_times (int): Number of times the plot has been called.
        intensity_matrix (numpy.ndarray): A 2D numpy array of intensity values.
        com_x_matrix (numpy.ndarray): A 2D numpy array of center of mass x values.
        com_y_matrix (numpy.ndarray): A 2D numpy array of center of mass y values.
    """
    
    def __init__(self, parent): 
        super(AnalysisWindow, self).__init__()
        self.parent = parent
        uic.loadUi('gui/analysis_window.ui', self)
        self.setWindowTitle('Analysis Window')
        self.xpos_path = None
        self.ypos_path = None
        self.save_path = None
        self.view_comx = None
        self.view_comy = None
        self.view_intensity = None
        self.analysis_index = self.parent.reader.analysis_index
        # self.load_path()

        #configuration
        # TODO: load config separately using the filepath provided by the parent
        # so it can be edited and still be loaded
        self.config: dict = self.parent.reader.config
        self.check_num_rois()
        self.configure_plots()

        self.timer_plot= QTimer()
        self.timer_plot.timeout.connect(self.plot_images)
        self.timer_plot.start(int(1000/self.calc_freq.value()))
        self.call_times = 0

        # self.btn_create_hdf5.clicked.connect(self.save_hdf5)
        # self.roi_x.valueChanged.connect(self.roi_boxes_changed)
        # self.roi_y.valueChanged.connect(self.roi_boxes_changed)
        # self.roi_width.valueChanged.connect(self.roi_boxes_changed)
        # self.roi_height.valueChanged.connect(self.roi_boxes_changed)
        self.calc_freq.valueChanged.connect(self.frequency_changed)
        self.cbox_select_roi.activated.connect(self.roi_selection_changed)
        self.chk_freeze.stateChanged.connect(self.freeze_plotting_checked)
        self.btn_reset.clicked.connect(self.reset_plot)

    def freeze_plotting_checked(self):
        """
        This function freezes the plot when the freeze plot checkbox is checked.
        """
        if self.chk_freeze.isChecked():
                self.timer_plot.stop()
        else:
            self.timer_plot.start(int(1000/self.calc_freq.value()))

    # def load_path(self):
    #     """
    #     This function loads the path information for the HDF5 file and uses it to populate other variables.
    #     """
    #     # TODO: These positions are references, use only when we miss frames.
    #     self.x_positions = np.load(self.xpos_path)
    #     self.y_positions = np.load(self.ypos_path)
    
    #     self.unique_x_positions = np.unique(self.x_positions) # Time Complexity = O(nlog(n))
    #     self.unique_y_positions = np.unique(self.y_positions) # Time Complexity = O(nlog(n))

    #     self.x_indices = np.searchsorted(self.unique_x_positions, self.x_positions) # Time Complexity = O(log(n))
    #     self.y_indices = np.searchsorted(self.unique_y_positions, self.y_positions) # Time Complexity = O(log(n))
         
    # def save_hdf5(self):
    #     """
    #     This function creates and saves the data as an HDF5 file with a timestamp as the name.
    #     """
    #     self.status_text.setText("Writing File...")
    #     # Get the current time as a timestamp for file name
    #     dt = datetime.fromtimestamp(time.time())
    #     formatted_time = dt.strftime('%Y%m%d%H%M')
    #     # put scan pos in dictionary 
    #     scan_pos = {'x_positions': self.x_positions,
    #                 'y_positions': self.y_positions}
    #     # start writer thread
    #     self.hdf5_writer_thread = HDF5WriterThread(
    #         f"{self.save_path}/{formatted_time}data.h5",
    #         self.parent.reader.images_cache, scan_pos, 
    #         self.parent.reader.metadata, 
    #         self.parent.reader.attributes, 
    #         self.intensity_matrix, 
    #         self.com_x_matrix, 
    #         self.com_y_matrix)
    #     self.hdf5_writer_thread.file_written.connect(self.on_file_written)
    #     self.hdf5_writer_thread.start()

    # def on_file_written(self):
    #     """
    #     This function is called when the file writing is done.
    #     """
    #     print("Signal Received")
    #     self.status_text.setText("File Written")
    #     QTimer.singleShot(10000, self.check_if_running)
        
    def check_if_running(self):
        """
        This function checks if the image scanning is running and sets the the status label's text.
        """
        if self.parent.reader.first_scan_detected:
            self.status_text.setText("Scanning...")
        else:
            self.status_text.setText("Waiting for the first scan...")

    def reset_plot(self):
        """
        This function resets the plot and clears all caches when the reset button is clicked.
        """
        self.parent.reader.first_scan_detected = False
        self.status_text.setText("Waiting for the first scan...")
        self.view_intensity.clear()
        self.view_comx.clear()
        self.view_comy.clear()
        self.call_times = 0
        # self.parent.reader.images_cache = None # [:,:,:] = 0 
        # Done because caching should be done from scratch
        self.parent.reader.frames_received = 0
        self.parent.reader.frames_missed = 0
        self.parent.reader.cache_id_gen = rotation_cycle(0,len(self.x_positions))
        self.parent.start_timers()

    def check_num_rois(self):
        """
        This function is called when the class is initialized to populate the dropdown of available ROIs
        """
        num_rois =  len(self.config.get('rois'))
        if num_rois > 0:
            for i in range(num_rois):
                self.cbox_select_roi.addItem(f'ROI{i+1}')

    def roi_selection_changed(self):
        """
        This function is called when the ROI is selected from the dropdown.
        Changes the viewable roi to one of the preset variables that we chose to monitor.
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
        
    # def roi_boxes_changed(self):
    #     """
    #     This function is called when the ROI dimensions are changed.
    #     Changes the viewable roi to the values within the the boxes.
    #     """
    #     self.parent.roi_x = self.roi_x.value()
    #     self.parent.roi_y = self.roi_y.value()
    #     self.parent.roi_width = self.roi_width.value()
    #     self.parent.roi_height = self.roi_height.value()
    #     self.cbox_select_roi.setCurrentIndex(0)
    #     self.call_times = 0
        
    def frequency_changed(self):
        self.timer_plot.start(int(1000/self.calc_freq.value()))

    def update_vectorized_image(self, intensity, com_x, com_y):
        size = int(np.sqrt(len(intensity)))
        intensity_matrix = np.reshape(intensity, (size, size))
        com_x_matrix = np.reshape(com_x,(size, size))
        com_y_matrix = np.reshape(com_y,(size, size))

        # USING IMAGE VIEW:

        if self.call_times == 5:
            self.view_intensity.setImage(img=intensity_matrix.T, autoRange=False, autoLevels=True, autoHistogramRange=False)
            self.view_comx.setImage(img=com_x_matrix.T, autoRange=False, autoHistogramRange=False)
            self.view_comy.setImage(img=com_y_matrix.T, autoRange=False, autoHistogramRange=False)

            self.view_comx.setLevels(0, self.roi_width.value())
            self.view_comy.setLevels(0,self.roi_height.value())
        else:
            self.view_intensity.setImage(img=intensity_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False)
            self.view_comx.setImage(img=com_x_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False)
            self.view_comy.setImage(img=com_y_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False)



    def plot_images(self):
        """
        Redraws plots based on rate entered in hz box.
        """

        if self.analysis_index is not None:

            self.call_times += 1
            analysis_attributes = self.parent.reader.attributes[self.analysis_index]
            #print(analysis_attributes)
            intensity = analysis_attributes["value"][0]["value"].get("Intensity",[])
            com_x = analysis_attributes["value"][0]["value"].get("ComX",[])
            com_y = analysis_attributes["value"][0]["value"].get("ComY",[])
            axis1 = analysis_attributes["value"][0]["value"].get("Axis1",0.0)
            axis2 = analysis_attributes["value"][0]["value"].get("Axis2",0.0)

            # print(intensity)
            if self.consumer_type == "vectorized":
                self.update_vectorized_image(intensity=intensity, com_x=com_x,com_y=com_y,)


    def init_scatter_plot(self):
        self.plot_intensity = pg.PlotWidget()
        self.plot_comx = pg.PlotWidget()
        self.plot_comy = pg.PlotWidget()

        scatter_item_intensity = pg.ScatterPlotItem()
        scatter_item_comx = pg.ScatterPlotItem()
        scatter_item_comy = pg.ScatterPlotItem()

        # image_item_intensity = pg.ImageItem()
        # image_item_comx = pg.ImageItem()
        # image_item_comy = pg.ImageItem()

        self.x_data = []
        self.y_data = []


        self.plot_intensity.addItem(scatter_item_intensity)
        # self.plot_intensity.addItem(image_item_intensity)
        self.grid_a.addWidget(self.plot_intensity,0,0)
        self.plot_intensity.setLabel('bottom', 'Motor Position X' )
        self.plot_intensity.setLabel('left', 'Motor Posistion Y')

        self.plot_comx.addItem(scatter_item_comx)
        # self.plot_comx.addItem(image_item_comx)
        self.grid_b.addWidget(self.plot_comx,0,0)
        self.plot_comx.setLabel('bottom', 'Motor Position X' )
        self.plot_comx.setLabel('left', 'Motor Posistion Y')

        self.plot_comy.addItem(scatter_item_comy)
        # self.plot_comy.addItem(image_item_comy)
        self.grid_c.addWidget(self.plot_comy,0,0)
        self.plot_comy.setLabel('bottom', 'Motor Position X' )
        self.plot_comy.setLabel('left', 'Motor Posistion Y')

    def init_image_view(self): 
        plot_item_intensity = pg.PlotItem()
        plot_item_comx = pg.PlotItem()
        plot_item_comy = pg.PlotItem()

        self.view_intensity = pg.ImageView(view=plot_item_intensity)
        self.grid_a.addWidget(self.view_intensity,0,0)
        self.view_intensity.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_intensity.view.getAxis('bottom').setLabel('Scan Position Cols')

        self.view_comx = pg.ImageView(view=plot_item_comx)
        self.grid_b.addWidget(self.view_comx,0,0)
        self.view_comx.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_comx.view.getAxis('bottom').setLabel('Scan Position Cols')

        self.view_comy = pg.ImageView(view=plot_item_comy)
        self.grid_c.addWidget(self.view_comy,0,0)
        self.view_comy.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_comy.view.getAxis('bottom').setLabel('Scan Position Cols')

    def configure_plots(self):
        """
        this function initializes the user interface with smaller ImageView classes and defining the axis.
        """
        # cmap = pg.colormap.getFromMatplotlib('viridis')
        self.consumer_type = self.config.get("ConsumerType", "")
        if self.consumer_type == "spontaneous":
            self.init_scatter_plot()
        elif self.consumer_type == "vectorized":
            self.init_image_view()
        else:
            #TODO: replace with w/ a message box
            print("Config Not Set Up Correctly")

        
    


    def closeEvent(self, event):
        self.parent.start_timers()
        del self.parent.analysis_window
        event.accept()
        super(AnalysisWindow, self).closeEvent(event)


