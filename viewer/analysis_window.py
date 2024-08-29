import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QTimer
from PyQt5 import uic
import numpy as np
import h5py
from datetime import datetime
import time
from generators import rotation_cycle


# Define the second window as a class
class AnalysisWindow(QMainWindow):
    def __init__(self, parent,xpos_path, ypos_path, save_path): # pipe, roi_pipe
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
        if self.chk_freeze.isChecked():
                self.timer_plot.stop()
        else:
            self.timer_plot.start(int(1000/self.calc_freq.value()))

    def load_path(self):
        # TODO: These positions are references, use only when we miss frames.
        self.x_positions = np.load(self.xpos_path)
        self.y_positions = np.load(self.ypos_path)
        self.unique_x_positions = np.unique(self.x_positions) # Time Complexity = O(nlog(n))
        self.unique_y_positions = np.unique(self.y_positions) # Time Complexity = O(nlog(n))

        self.x_indices = np.searchsorted(self.unique_x_positions, self.x_positions) # Time Complexity = O(log(n))
        self.y_indices = np.searchsorted(self.unique_y_positions, self.y_positions) # Time Complexity = O(log(n))

    def create_hdf5_file(self, filename, images_cache, scan_pos, metadata, attributes, intensity_values, com_x_matrix, com_y_matrix):
        with h5py.File(filename, 'w') as h5file:
            #data
            h5file.create_group('/data')
            h5file.create_dataset('/data/images', data=images_cache)

            #scan pos
            h5file.create_group('/data/scan_pos')
            h5file.create_dataset('/data/scan_pos/x_positions', data=scan_pos['x_positions'])
            h5file.create_dataset('/data/scan_pos/y_positions', data=scan_pos['y_positions'])

            #save all metadata and attributes too. Expecting dictionaries
            h5file.create_group('/metadata')
            for key, value in metadata.items():
                h5file['/metadata'].attrs[key] = value
            h5file.create_group('/attributes')
            for key, value in attributes.items():
                h5file['/attributes'].attrs[key] = value

            #Analysis data 
            h5file.create_group('/analysis')
            h5file.create_dataset('/analysis/total_intensity', data=intensity_values)
            h5file.create_dataset('/analysis/com_x', data=com_x_matrix)
            h5file.create_dataset('/analysis/com_y', data=com_y_matrix)

            
    def save_hdf5(self):
        
        #time stamp for file name
        # Get the current time as a timestamp
        dt = datetime.fromtimestamp(time.time())
        formatted_time = dt.strftime('%Y%m%d%H%M')
        #put scan pos in dictionary 
        scan_pos = {
                'x_positions': self.x_positions,
                'y_positions': self.y_positions
            }
        # 
        self.status_text.setText("Writing File...")
        #call this to write file
        self.create_hdf5_file(f"{self.save_path}/{formatted_time}data.h5", self.parent.reader.images_cache, scan_pos, self.parent.reader.metadata, self.parent.reader.attributes, self.intensity_matrix, self.com_x_matrix, self.com_y_matrix)
        self.status_text.setText("File Written")
        QTimer.singleShot(10000, self.check_if_running)
        
    def check_if_running(self):
        if self.parent.reader.first_scan_detected:
            self.status_text.setText("Scanning...")
        else:
            self.status_text.setText("Waiting for the first scan...")

    def reset_plot(self):
        self.parent.reader.first_scan_detected = False
        self.status_text.setText("Waiting for the first scan...")
        self.view_intensity.clear()
        self.view_comx.clear()
        self.view_comy.clear()
        self.call_times = 0
        self.parent.reader.images_cache = None# [:,:,:] = 0 
        # Done because caching should be done from scratch
        self.parent.reader.frames_received = 0
        self.parent.reader.frames_missed = 0
        self.parent.reader.cache_id_gen = rotation_cycle(0,len(self.x_positions))
        self.parent.start_timers()

    def check_num_rois(self):
            num_rois =  self.parent.reader.num_rois
            if num_rois > 0:
                for i in range(num_rois):
                    self.cbox_select_roi.addItem(f'ROI{i+1}')

    def roi_selection_changed(self):
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
        self.parent.roi_x = self.roi_x.value()
        self.parent.roi_y = self.roi_y.value()
        self.parent.roi_width = self.roi_width.value()
        self.parent.roi_height = self.roi_height.value()
        self.cbox_select_roi.setCurrentIndex(0)
        self.call_times = 0
        
    def frequency_changed(self):
        self.timer_plot.start(int(1000/self.calc_freq.value()))

    def plot_images(self):
        if self.parent.reader.first_scan_detected == False:
            self.status_text.setText("Waiting for the first scan...")
        else:

            if self.call_times == 0:
                    self.status_text.setText("Scanning...")
            self.call_times += 1
            
            scan_id = self.parent.reader.cache_id
            xpos_reader = self.parent.reader.positions_cache[scan_id, 0]
            ypos_reader = self.parent.reader.positions_cache[scan_id, 1]
            xpos_scan = self.x_positions[scan_id]
            ypos_scan = self.y_positions[scan_id]

            # # print(f'xpos_reader: {xpos_reader}\nxpos_scan: {xpos_scan}')
            # # print(f'\nypos_reader: {ypos_reader}\nypos_scan: {ypos_scan}\n')


            if (np.abs((xpos_scan-xpos_reader)) < .05) and (np.abs((ypos_scan-ypos_reader)) < .05):
                image_rois = self.parent.reader.images_cache[:,
                                                        self.parent.roi_y:self.parent.roi_y + self.parent.roi_height,
                                                        self.parent.roi_x:self.parent.roi_x + self.parent.roi_width]# self.pv_dict.get('rois', [[]]) # Time Complexity = O(n)
                
                intensity_values = np.sum(image_rois, axis=(1, 2)) # Time Complexity = O(n)
                intensity_values_non_zeros = intensity_values # removed deep copy of intensity values as memory was cleared with every function call
                intensity_values_non_zeros[intensity_values_non_zeros==0] = 1E-6 # time complexity = O(1)
                
                # Instead of reading the scan positions from collector 
                # TODO: make sure the scan positions read from the numpy file is the same as those read through the collector 
                # and overwrite if not
                # we had made sure in area_det_viewr that the starting scan position match
                
                # print(f"x first pos: {x_positions[0]}, y first pos: {y_positions[0]}")

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


# if __name__ == '__main__':
#     import multiprocessing as mp
#     parent_pipe, child_pipe = mp.Pipe()
#     p = mp.Process(target=analysis_window_process, args=(child_pipe,))
    
#     p.start()
#     parent_pipe.send({1000:np.zeros((1024,1024))})
#     try:
#         while True:
#             # Handling messages from the main process if necessary
#             if parent_pipe.poll():
#                 message = parent_pipe.recv()
#                 if message == 'close':
#                     break
#     finally:
#         p.join()
