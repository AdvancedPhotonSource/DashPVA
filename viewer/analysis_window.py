import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSizePolicy#, QGridLayout
from PyQt5.QtCore import QTimer
# from PyQt5.QtChart import QChart
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
# from area_det_viewer import ImageWindow
from label_with_axis import MyLabel



# Global function so it can be called without needing to be within a class and get around
# not being able to pass arguments to pyqt slots
def analysis_window_process(pipe, roi_pipe):
    app = QApplication(sys.argv)
    window = AnalysisWindow(pipe, roi_pipe)
    window.show()
    app.exec_()


x_positions = np.load("xpos.npy")
y_positions = np.load("ypos.npy")
unique_x_positions = np.unique(x_positions) # Time Complexity = O(nlog(n))
unique_y_positions = np.unique(y_positions) # Time Complexity = O(nlog(n))
x_indices = np.searchsorted(unique_x_positions, x_positions) # Time Complexity = O(log(n))
y_indices = np.searchsorted(unique_y_positions, y_positions) # Time Complexity = O(log(n))

# Define the second window as a class
class AnalysisWindow(QMainWindow):
    def __init__(self, parent,): # pipe, roi_pipe
        super(AnalysisWindow, self).__init__()
        # self.pipe = pipe # used to share memory with another process
        # self.roi_pipe = roi_pipe
        # self.parent : ImageWindow = parent
        self.parent = parent
        uic.loadUi('gui/analysis_window.ui', self)
        self.setWindowTitle('Analysis Window')
        self.init_ui()

        # self.timer_poll = QTimer()
        # self.timer_poll.timeout.connect(self.timer_poll_pipe)
        # self.timer_poll.start()

        # self.timer_receive_num_rois = QTimer()
        # self.timer_receive_num_rois.timeout.connect(self.check_num_rois)
        # self.timer_receive_num_rois.start(int(1000/100))

        self.timer_plot= QTimer()
        self.timer_plot.timeout.connect(self.plot_images)
        self.timer_plot.start(int(1000/self.calc_freq.value()))

        self.cache_timer_corr = False
        # self.pv_dict = None
        # self.num_rois = None
        self.call_times = 0
        # self.time_to_avg = np.array([])

        self.roi_x.editingFinished.connect(self.roi_boxes_changed)
        self.roi_y.editingFinished.connect(self.roi_boxes_changed)
        self.roi_width.editingFinished.connect(self.roi_boxes_changed)
        self.roi_height.editingFinished.connect(self.roi_boxes_changed)
        self.calc_freq.editingFinished.connect(self.frequency_changed)
        self.cbox_select_roi.activated.connect(self.roi_selection_changed)
        self.check_num_rois()

    def check_num_rois(self):
            num_rois =  self.parent.reader.num_rois
        # if self.roi_pipe.poll():
            # num_message: dict = self.roi_pipe.recv()
            # self.num_rois = num_message.get('num_rois',0)
            # if self.num_rois > 0:
            if num_rois > 0:
                for i in range(num_rois):
                    self.cbox_select_roi.addItem(f'ROI{i+1}')
                # self.timer_receive_num_rois.stop()
            # elif self.num_rois == 0:
            #     self.timer_receive_num_rois.stop()


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


        # self.roi_pipe.send({'x': f'{text}:MinX',
        #                     'y': f'{text}:MinY',
        #                     'width': f'{text}:SizeX',
        #                     'height': f'{text}:SizeY'
        #                     })

    def roi_boxes_changed(self):
        self.parent.roi_x = self.roi_x.value()
        self.parent.roi_y = self.roi_y.value()
        self.parent.roi_width = self.roi_width.value()
        self.parent.roi_height = self.roi_height.value()
        self.cbox_select_roi.setCurrentIndex(0)
        # self.roi_pipe.send({
        #                     'x': self.roi_x.value(),
        #                     'y': self.roi_y.value(),
        #                     'width': self.roi_width.value(),
        #                     'height': self.roi_height.value()
        #                     })
        
    def frequency_changed(self):
        self.timer_plot.start(int(1000/self.calc_freq.value()))

    # def timer_poll_pipe(self):
    #     if self.pipe.poll():
    #         self.pv_dict : dict = self.pipe.recv() # try to send uniqueID
    #         if self.pv_dict is not None:
    #             self.timer.start(int(1000/float(self.pv_dict['cache_freq'])))#TODO: differernt frequency required... need a textbox

    def plot_images(self):
        # if self.pv_dict is not None:
            image_rois = self.parent.reader.images_cache[:,
                                                         self.parent.roi_y:self.parent.roi_y + self.parent.roi_height,
                                                         self.parent.roi_x:self.parent.roi_x + self.parent.roi_width]# self.pv_dict.get('rois', [[]]) # Time Complexity = O(n)
            # if self.cache_timer_corr == False:
            #     self.timer.start(int(1000/float(self.calc_freq.value())))#TODO: differernt frequency required... need a textbox
            #     self.cache_timer_corr = True 
            if self.parent.reader.first_scan_detected == False: # self.pv_dict.get('first_scan', [[]])
                self.status_text.setText("Waiting for the first scan...")
            else:
                # time_start = time.time()
                self.status_text.setText("Scanning...")
                self.call_times += 1
                # print(f"{self.call_times=}")
                
                intensity_values = np.sum(image_rois, axis=(1, 2)) # Time Complexity = O(n)
                intensity_values_non_zeros = intensity_values # removed deep copy of intensity values as memory was cleared with every function call
                intensity_values_non_zeros[intensity_values_non_zeros==0] = 1E-6 # time complexity = O(1)

                # print(f"total intensity val calculated: {np.count_nonzero(intensity_values)}")
                # x_positions = copy.deepcopy(self.pv_dict.get('x_pos',[]))
                # y_positions = copy.deepcopy(self.pv_dict.get('y_pos',[]))
                
                # Instead of reading the scan positions from collector 
                # TODO: make sure the scan positions read from the numpy file is the same as those read through the collector 
                # and overwrite if not
                # we had made sure in area_det_viewr that the starting scan position match
                
                
                # print(f"x first pos: {x_positions[0]}, y first pos: {y_positions[0]}")
                # if (x_positions[0] == 0 ) and (y_positions[0] == 0 ):                

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
                intensity_matrix = np.zeros((len(unique_y_positions), len(unique_x_positions))) # Time Complexity = O(n)
                intensity_matrix[y_indices, x_indices] = intensity_values # Time Complexity = O(1)
                # gets the shape of the image to set the length of the axis
                
                com_x_matrix = np.zeros((len(unique_y_positions), len(unique_x_positions))) # Time Complexity = O(n)
                com_y_matrix = np.zeros((len(unique_y_positions), len(unique_x_positions))) # Time Complexity = O(n)

                # Populate the matrices using the indices
                com_x_matrix[y_indices, x_indices] = com_x # Time Complexity = O(1)
                com_y_matrix[y_indices, x_indices] = com_y # Time Complexity = O(1)
                
                # scan_range = int(np.sqrt(np.shape(image_rois)[0]))
                # intensity_matrix = np.reshape(intensity_values, (scan_range,scan_range), order = "C")
                # print(f"intensity_matrix non zeros: {np.count_nonzero(intensity_matrix)}")
                
                # Normalize the data to the range [0, 65535]
                min_val = 0 # np.min(intensity_matrix) # Time Complexity = O(n)
                max_val = np.max(intensity_matrix) # Time Complexity = O(n)

                # Avoid division by zero if max_val equals min_val
                if max_val > min_val:
                    intensity_matrix = ((intensity_matrix - min_val) / (max_val - min_val)) * 65535 # Time Complexity = O(1)
                else:
                    intensity_matrix = np.zeros_like(intensity_matrix) # Time Complexity = O(n)  # If all values are the same, set to zero

                intensity_matrix = intensity_matrix.astype(np.uint16)

                # print(f"intensity_matrix non-zeros: {np.count_nonzero(intensity_matrix)}")

                height, width = intensity_matrix.shape
                bytes_per_line = width * 2  # Since it's 16 bits (2 bytes per pixel)

                # Ensure data is contiguous for QImage
                img = QImage(intensity_matrix.data, width, height, bytes_per_line, QImage.Format_Grayscale16)
                pixmap = QPixmap.fromImage(img)
                self.intensity_matrix.setPixmap(pixmap.scaled(self.intensity_matrix.size(), aspectRatioMode=Qt.KeepAspectRatio))                
                            
                # Normalize the com_x_matrix
                min_val = 0 # np.min(com_x_matrix) # Time Complexity = O(n)
                max_val = np.max(com_x_matrix) # Time Complexity = O(n)
                if max_val > min_val:
                    com_x_matrix = ((com_x_matrix - min_val) / (max_val - min_val)) * 65535
                else:
                    com_x_matrix = np.zeros_like(com_x_matrix) # Time Complexity = O(n)
                com_x_matrix = com_x_matrix.astype(np.uint16)

                # Create the QImage for com_x_matrix
                height, width = com_x_matrix.shape
                bytes_per_line = width * 2  # Since it's 16 bits (2 bytes per pixel)
                img = QImage(com_x_matrix.data, width, height, bytes_per_line, QImage.Format_Grayscale16)
                pixmap = QPixmap.fromImage(img)
                self.center_of_mass_x.setPixmap(pixmap.scaled(self.center_of_mass_x.size(), aspectRatioMode=Qt.KeepAspectRatio))

                # Normalize the com_y_matrix
                min_val = 0 # np.min(com_y_matrix) # Time Complexity = O(n)
                max_val = np.max(com_y_matrix) # Time Complexity = O(n)
                if max_val > min_val:
                    com_y_matrix = ((com_y_matrix - min_val) / (max_val - min_val)) * 65535
                else:
                    com_y_matrix = np.zeros_like(com_y_matrix) # Time Complexity = O(n)
                com_y_matrix = com_y_matrix.astype(np.uint16)

                # Create the QImage for com_y_matrix
                height, width = com_y_matrix.shape
                bytes_per_line = width * 2  # Since it's 16 bits (2 bytes per pixel)
                img = QImage(com_y_matrix.data, width, height, bytes_per_line, QImage.Format_Grayscale16)
                pixmap = QPixmap.fromImage(img)
                self.center_of_mass_y.setPixmap(pixmap.scaled(self.center_of_mass_y.size(), aspectRatioMode=Qt.KeepAspectRatio))

                # self.time_to_avg = np.append(self.time_to_avg, time.time()-time_start)
                # print(f'50x50: {np.average(self.time_to_avg)}')
                # print(f'100x100: {np.average(self.time_to_avg)}')
                # print(f'200x200: {np.average(self.time_to_avg)}')
                # print(f'400x400: {np.average(self.time_to_avg)}')

                # height, width = com_x_matrix.shape
                # img = QImage(com_x_matrix.data, width, height, width, QImage.Format_Grayscale8)
                # pixmap = QPixmap.fromImage(img)
                # self.center_of_mass_x.setPixmap(pixmap.scaled(self.center_of_mass_x.size(), aspectRatioMode=True))
                
                # height, width = com_y_matrix.shape
                # img = QImage(com_y_matrix.data, width, height, width, QImage.Format_Grayscale8)
                # pixmap = QPixmap.fromImage(img)
                # self.center_of_mass_y.setPixmap(pixmap.scaled(self.center_of_mass_y.size(), aspectRatioMode=True))
            # else:
            #     self.roll_nums += 1


    def init_ui(self):
        self.x_axis_intensity= MyLabel(location='bottom', img_resolution=(len(unique_x_positions), len(unique_y_positions)))
        self.x_axis_intensity.side_length = 379# self.intensity_matrix.width()
        # self.x_axis_intensity.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.h_layout_intensity.addWidget(self.x_axis_intensity)

        self.y_axis_intensity= MyLabel(location='left',img_resolution=(len(unique_x_positions), len(unique_y_positions)))
        self.y_axis_intensity.side_length = 379# self.intensity_matrix.width()
        # self.y_axis_intensity.h = len(unique_y_positions)
        # self.x_axis_intensity.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.v_layout_intensity.addWidget(self.y_axis_intensity)
        
        # self.plot = pg.PlotItem()
        # self.plot_viewer = pg.ImageView(view=self.plot,)
        # self.grid_a.addWidget(self.plot_viewer,0,0)

    # def init_ui(self):
    #     self.label_a = QLabel()
    #     self.label_a.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    #     # self.grid_a = QGridLayout()  # Make sure grid_a is initialized
    #     # self.setLayout(self.grid_a)  # Set grid_a as the layout for the window
    #     self.grid_a.addWidget(self.label_a, 0, 0)
    #     self.grid_a.setRowStretch(0, 1)
    #     self.grid_a.setColumnStretch(0, 1)

    # def closeEvent(self, event):
    #     self.pipe.send('close')
    #     self.pipe.close()
    #     event.accept()
    #     super(AnalysisWindow, self).closeEvent(event)


if __name__ == '__main__':
    import multiprocessing as mp
    parent_pipe, child_pipe = mp.Pipe()
    p = mp.Process(target=analysis_window_process, args=(child_pipe,))
    
    p.start()
    parent_pipe.send({1000:np.zeros((1024,1024))})
    try:
        while True:
            # Handling messages from the main process if necessary
            if parent_pipe.poll():
                message = parent_pipe.recv()
                if message == 'close':
                    break
    finally:
        p.join()
