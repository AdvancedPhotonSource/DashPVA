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



# Global function so it can be called without needing to be within a class and get around
# not being able to pass arguments to pyqt slots
def analysis_window_process(pipe):
    app = QApplication(sys.argv)
    window = AnalysisWindow(pipe)
    window.show()
    app.exec_()


# Define the second window as a class
class AnalysisWindow(QMainWindow):
    def __init__(self,pipe):
        super(AnalysisWindow, self).__init__()
        self.pipe = pipe # used to share memory with another process
        uic.loadUi('gui/analysis_window.ui', self)
        self.setWindowTitle('Analysis Window')
        # self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_poll_pipe)
        self.timer.start(10)
        self.pv_dict = None
        self.roll_nums = 1

    def timer_poll_pipe(self):

        if self.pipe.poll():
            self.pv_dict : dict = self.pipe.recv() # try to send uniqueID
            image_rois = copy.deepcopy(self.pv_dict.get('rois', [[]]))
            # roi_x = 100# int(self.roi_x.toPlainText())
            # roi_y = 100 #int(self.roi_y.toPlainText())
            # roi_width = 50 #int(self.roi_width.toPlainText())
            # roi_height = 50#int(self.roi_height.toPlainText())
            # image_rois=image_rois[:,roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            intensity_values = np.sum(image_rois, axis=(1, 2))
            x_positions = copy.deepcopy(self.pv_dict.get('x_pos',[]))
            y_positions = copy.deepcopy(self.pv_dict.get('y_pos',[]))
            
            #rolling to correct the starting point
            image_rois = np.roll(image_rois, -1*self.roll_nums, axis=0)
            x_positions = np.roll(x_positions, -1*self.roll_nums)
            y_positions = np.roll(y_positions, -1*self.roll_nums)
            
            print(f"x first pos: {x_positions[0]}, y first pos: {y_positions[0]}")
            if (x_positions[0] == 0 ) and (y_positions[0] == 0 ):
                unique_x_positions = np.unique(x_positions)
                unique_y_positions = np.unique(y_positions)
        


                x_indices = np.searchsorted(unique_x_positions, x_positions)
                y_indices = np.searchsorted(unique_y_positions, y_positions)

                y_coords, x_coords = np.indices((np.shape(image_rois)[1], np.shape(image_rois)[2]))

                # Compute weighted sums
                weighted_sum_y = np.sum(image_rois[:, :, :] * y_coords[np.newaxis, :, :], axis=(1, 2))
                weighted_sum_x = np.sum(image_rois[:, :, :] * x_coords[np.newaxis, :, :], axis=(1, 2))

                # Calculate COM
                com_y = weighted_sum_y / intensity_values
                com_x = weighted_sum_x / intensity_values
                
                #filter out inf
                com_x[com_x==np.nan] = 0
                com_y[com_y==np.nan] = 0
                #Two lines below don't work if unique positions are messed by incomplete x y positions 
                intensity_matrix = np.zeros((len(unique_y_positions), len(unique_x_positions)))
                intensity_matrix[y_indices, x_indices] = intensity_values
                
                # scan_range = int(np.sqrt(np.shape(image_rois)[0]))
                # intensity_matrix = np.reshape(intensity_values, (scan_range,scan_range), order = "C")
                
                com_x_matrix = np.zeros((len(unique_y_positions), len(unique_x_positions)))
                com_y_matrix = np.zeros((len(unique_y_positions), len(unique_x_positions)))
                # # Populate the matrices using the indices
                
                com_x_matrix[y_indices, x_indices] = com_x
                com_y_matrix[y_indices, x_indices] = com_y

                height, width = intensity_matrix.shape
                img = QImage(intensity_matrix.data, width, height, width, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(img)
                self.intensity_matrix.setPixmap(pixmap.scaled(self.intensity_matrix.size(), aspectRatioMode=True))
                
                height, width = com_x_matrix.shape
                img = QImage(com_x_matrix.data, width, height, width, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(img)
                self.center_of_mass_x.setPixmap(pixmap.scaled(self.center_of_mass_x.size(), aspectRatioMode=True))
                
                height, width = com_y_matrix.shape
                img = QImage(com_y_matrix.data, width, height, width, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(img)
                self.center_of_mass_y.setPixmap(pixmap.scaled(self.center_of_mass_y.size(), aspectRatioMode=True))
            else:
                self.roll_nums += 1


    # def init_ui(self):
    #     self.label_a = QLabel()
    #     self.label_a.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    #     self.grid_a.addWidget(self.label_a,0,0)
    #     # self.plot = pg.PlotItem()
    #     # self.plot_viewer = pg.ImageView(view=self.plot,)
    #     # self.grid_a.addWidget(self.plot_viewer,0,0)
    # def init_ui(self):
    #     self.label_a = QLabel()
    #     self.label_a.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    #     # self.grid_a = QGridLayout()  # Make sure grid_a is initialized
    #     # self.setLayout(self.grid_a)  # Set grid_a as the layout for the window
    #     self.grid_a.addWidget(self.label_a, 0, 0)
    #     self.grid_a.setRowStretch(0, 1)
    #     self.grid_a.setColumnStretch(0, 1)
    def closeEvent(self, event):
        self.pipe.send('close')
        self.pipe.close()
        event.accept()
        super(AnalysisWindow, self).closeEvent(event)


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
