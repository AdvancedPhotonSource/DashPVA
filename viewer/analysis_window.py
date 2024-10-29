import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
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
    
    def __init__(self, parent,xpos_path, ypos_path, save_path): 
        super(AnalysisWindow, self).__init__()
        # self.parent : ImageWindow = parent
        self.parent = parent
        uic.loadUi('gui/analysis_window.ui', self)
        self.setWindowTitle('Analysis Window')
        self.init_ui()
        self.xpos_path = xpos_path
        self.ypos_path = ypos_path
        self.save_path = save_path
        self.load_path()

        self.timer_plot= QTimer()
        self.timer_plot.timeout.connect(self.plot_images)
        self.timer_plot.start(int(1000/self.calc_freq.value()))
        self.call_times = 0

        self.btn_create_hdf5.clicked.connect(self.save_hdf5)
        self.roi_x.valueChanged.connect(self.roi_boxes_changed)
        self.roi_y.valueChanged.connect(self.roi_boxes_changed)
        self.roi_width.valueChanged.connect(self.roi_boxes_changed)
        self.roi_height.valueChanged.connect(self.roi_boxes_changed)
        self.calc_freq.valueChanged.connect(self.frequency_changed)
        self.cbox_select_roi.activated.connect(self.roi_selection_changed)
        self.chk_freeze.stateChanged.connect(self.freeze_plotting_checked)
        self.btn_reset.clicked.connect(self.reset_plot)
        self.check_num_rois()
    
    def freeze_plotting_checked(self):
        """
        This function freezes the plot when the freeze plot checkbox is checked.
        """
        if self.chk_freeze.isChecked():
                self.timer_plot.stop()
        else:
            self.timer_plot.start(int(1000/self.calc_freq.value()))

    def load_path(self):
        """
        This function loads the path information for the HDF5 file and uses it to populate other variables.
        """
        # TODO: These positions are references, use only when we miss frames.
        self.x_positions = np.load(self.xpos_path)
        self.y_positions = np.load(self.ypos_path)
    
        self.unique_x_positions = np.unique(self.x_positions) # Time Complexity = O(nlog(n))
        self.unique_y_positions = np.unique(self.y_positions) # Time Complexity = O(nlog(n))

        self.x_indices = np.searchsorted(self.unique_x_positions, self.x_positions) # Time Complexity = O(log(n))
        self.y_indices = np.searchsorted(self.unique_y_positions, self.y_positions) # Time Complexity = O(log(n))
         
    def save_hdf5(self):
        """
        This function creates and saves the data as an HDF5 file with a timestamp as the name.
        """
        self.status_text.setText("Writing File...")
        # Get the current time as a timestamp for file name
        dt = datetime.fromtimestamp(time.time())
        formatted_time = dt.strftime('%Y%m%d%H%M')
        # put scan pos in dictionary 
        scan_pos = {
                'x_positions': self.x_positions,
                'y_positions': self.y_positions
            }
        # start writer thread
        self.hdf5_writer_thread = HDF5WriterThread(
            f"{self.save_path}/{formatted_time}data.h5",
            self.parent.reader.images_cache, scan_pos, 
            self.parent.reader.metadata, 
            self.parent.reader.attributes, 
            self.intensity_matrix, 
            self.com_x_matrix, 
            self.com_y_matrix)
        self.hdf5_writer_thread.file_written.connect(self.on_file_written)
        self.hdf5_writer_thread.start()

    def on_file_written(self):
        """
        This function is called when the file writing is done.
        """
        print("Signal Received")
        self.status_text.setText("File Written")
        QTimer.singleShot(10000, self.check_if_running)
        
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
        self.parent.reader.images_cache = None # [:,:,:] = 0 
        # Done because caching should be done from scratch
        self.parent.reader.frames_received = 0
        self.parent.reader.frames_missed = 0
        self.parent.reader.cache_id_gen = rotation_cycle(0,len(self.x_positions))
        self.parent.start_timers()

    def check_num_rois(self):
        """
        This function is called when the class is initialized to populate the dropdown of available ROIs
        """
        num_rois =  self.parent.reader.num_rois
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
        
    def roi_boxes_changed(self):
        """
        This function is called when the ROI dimensions are changed.
        Changes the viewable roi to the values within the the boxes.
        """
        self.parent.roi_x = self.roi_x.value()
        self.parent.roi_y = self.roi_y.value()
        self.parent.roi_width = self.roi_width.value()
        self.parent.roi_height = self.roi_height.value()
        self.cbox_select_roi.setCurrentIndex(0)
        self.call_times = 0
        
    def frequency_changed(self):
        self.timer_plot.start(int(1000/self.calc_freq.value()))

    def plot_images(self):
        """
        Redraws plots based on rate entered in hz box.
        Processes the images based on the different settings.
        """
        if self.parent.reader.first_scan_detected == False:
            self.status_text.setText("Waiting for the first scan...")
        else:

            if self.call_times == 0:
                    self.status_text.setText("Scanning...")
            
            if self.parent.reader.cache_id is not None:
                scan_id = self.parent.reader.cache_id
                xpos_det = self.parent.reader.positions_cache[scan_id, 0]
                xpos_plan = self.x_positions[scan_id]
                ypos_det = self.parent.reader.positions_cache[scan_id, 1]
                ypos_plan = self.y_positions[scan_id]

                # print(f'scan id: {scan_id}')
                # print(f'detector: {xpos_det},{ypos_det}')
                # print(f'scan plan: {xpos_plan},{ypos_plan}\n')

                x1, x2 = self.x_positions[0], self.x_positions[1]
                y1, y2 = self.y_positions[0], self.y_positions[30]

            if ((np.abs(xpos_plan-xpos_det) < (np.abs(x2-x1) * 0.2)) and (np.abs(ypos_plan-ypos_det) < (np.abs(y2-y1) * 0.2))): 
                self.call_times += 1

                image_rois = self.parent.reader.images_cache[:,
                                                        self.parent.roi_y:self.parent.roi_y + self.parent.roi_height,
                                                        self.parent.roi_x:self.parent.roi_x + self.parent.roi_width]# self.pv_dict.get('rois', [[]]) # Time Complexity = O(n)
                
                intensity_values = np.sum(image_rois, axis=(1, 2)) # Time Complexity = O(n)
                intensity_values_non_zeros = intensity_values # removed deep copy of intensity values as memory was cleared with every function call
                intensity_values_non_zeros[intensity_values_non_zeros==0] = 1E-6 # time complexity = O(1)

                y_coords, x_coords = np.indices((np.shape(image_rois)[1], np.shape(image_rois)[2])) # Time Complexity = 0(n)

                # Compute weighted sums
                weighted_sum_y = np.sum(image_rois[:, :, :] * y_coords[np.newaxis, :, :], axis=(1, 2)) # Time Complexity O(n)
                weighted_sum_x = np.sum(image_rois[:, :, :] * x_coords[np.newaxis, :, :], axis=(1, 2)) # Time Complexity O(n)
                # Calculate COM
                com_y = weighted_sum_y / intensity_values_non_zeros # time complexity = O(1)
                com_x = weighted_sum_x / intensity_values_non_zeros # time complexity = O(1)
                #filter out inf
                com_x[com_x==np.nan] = 0 # time complexity = O(1)
                com_y[com_y==np.nan] = 0 # time complexity = O(1)
                #Two lines below don't work if unique positions are messed by incomplete x y positions 
                self.intensity_matrix = np.zeros((len(self.unique_y_positions), len(self.unique_x_positions))) # Time Complexity = O(n)
                self.intensity_matrix[self.y_indices, self.x_indices] = intensity_values # Time Complexity = O(1)
                # gets the shape of the image to set the length of the axis
                self.com_x_matrix = np.zeros((len(self.unique_y_positions), len(self.unique_x_positions))) # Time Complexity = O(n)
                self.com_y_matrix = np.zeros((len(self.unique_y_positions), len(self.unique_x_positions))) # Time Complexity = O(n)
                # Populate the matrices using the indices
                self.com_x_matrix[self.y_indices, self.x_indices] = com_x # Time Complexity = O(1)
                self.com_y_matrix[self.y_indices, self.x_indices] = com_y # Time Complexity = O(1)
                
                # USING IMAGE VIEW:
                if self.call_times == 5:
                    self.view_intensity.setImage(img=self.intensity_matrix.T, autoRange=False, autoLevels=True, autoHistogramRange=False)
                    self.view_comx.setImage(img=self.com_x_matrix.T, autoRange=False, autoHistogramRange=False)
                    self.view_comy.setImage(img=self.com_y_matrix.T, autoRange=False, autoHistogramRange=False)

                    self.view_comx.setLevels(0, self.roi_width.value())
                    self.view_comy.setLevels(0,self.roi_height.value())
                else:
                    self.view_intensity.setImage(img=self.intensity_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False)
                    self.view_comx.setImage(img=self.com_x_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False)
                    self.view_comy.setImage(img=self.com_y_matrix.T, autoRange=False, autoLevels=False, autoHistogramRange=False)
            else:
                self.parent.reader.images_cache[scan_id,:,:] = 0
                self.parent.reader.frames_missed += 1

    def init_ui(self):
        """
        this function initializes the user interface with smaller ImageView classes and defining the axis.
        """
        # cmap = pg.colormap.getFromMatplotlib('viridis')
        plot_item_intensity = pg.PlotItem()
        self.view_intensity = pg.ImageView(view=plot_item_intensity)
        # self.view_intensity.setColorMap(cmap)
        self.grid_a.addWidget(self.view_intensity,0,0)
        self.view_intensity.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_intensity.view.getAxis('bottom').setLabel('Scan Position Cols')

        plot_item_comx = pg.PlotItem()
        self.view_comx = pg.ImageView(view=plot_item_comx)
        # self.view_comx.setColorMap(cmap)
        self.grid_b.addWidget(self.view_comx,0,0)
        self.view_comx.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_comx.view.getAxis('bottom').setLabel('Scan Position Cols')

        plot_item_comy = pg.PlotItem()
        self.view_comy = pg.ImageView(view=plot_item_comy)
        # self.view_comy.setColorMap(cmap)
        self.grid_c.addWidget(self.view_comy,0,0)
        self.view_comy.view.getAxis('left').setLabel('Scan Position Rows')
        self.view_comy.view.getAxis('bottom').setLabel('Scan Position Cols')

    def closeEvent(self, event):
        self.parent.start_timers()
        del self.parent.analysis_window
        event.accept()
        super(AnalysisWindow, self).closeEvent(event)


