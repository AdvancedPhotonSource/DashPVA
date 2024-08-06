import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtChart import QChart
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt
import numpy as np
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
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_poll_pipe)
        self.timer.start(1000)
        self.pv_dict = None

    def timer_poll_pipe(self):

        if self.pipe.poll():
            self.pv_dict : dict = self.pipe.recv() # try to send uniqueID

            image_rois = copy.deepcopy(self.pv_dict.get('rois', [[]]))
            intensity_values = copy.deepcopy(self.pv_dict.get('intensity', []))
            x_positions = copy.deepcopy(self.pv_dict.get('x_pos',[]))
            y_positions = self.pv_dict.get('y_pos',[])
            unique_x_positions = copy.deepcopy(self.pv_dict.get('unique_x_pos',[]))
            unique_y_positions = copy.deepcopy(self.pv_dict.get('unique_y_pos',[]))


            x_indices = np.searchsorted(unique_x_positions, x_positions)
            y_indices = np.searchsorted(unique_y_positions, y_positions)

            y_coords, x_coords = np.indices((np.shape(image_rois)[1], np.shape(image_rois)[2]))

            # Compute weighted sums
            weighted_sum_y = np.sum(image_rois[:, :, :] * y_coords[np.newaxis, :, :], axis=(1, 2))
            weighted_sum_x = np.sum(image_rois[:, :, :] * x_coords[np.newaxis, :, :], axis=(1, 2))

            # Calculate COM
            com_y = weighted_sum_y / intensity_values
            com_x = weighted_sum_x / intensity_values
            intensity_matrix = np.zeros((len(unique_y_positions), len(unique_x_positions)))
            com_x_matrix = np.zeros((len(unique_y_positions), len(unique_x_positions)))
            com_y_matrix = np.zeros((len(unique_y_positions), len(unique_x_positions)))
            # Populate the matrices using the indices
            intensity_matrix[y_indices, x_indices] = intensity_values
            com_x_matrix[y_indices, x_indices] = com_x
            com_y_matrix[y_indices, x_indices] = com_y

            # TESTING: CAST MATRIX INTO NUMPY ARRAY
            intensity_matrix = np.array(intensity_matrix)

            # intensity_matrix = np.log(intensity_matrix+1)

            # TESTING WITH RANDIOM MATRIX 
            # intensity_matrix = np.random.randn(1024, 1024)
            height, width = intensity_matrix.shape

            # TESTING QIMAGE:
            # img = QImage(intensity_matrix.data, height, width, width*intensity_matrix.itemsize , QImage.Format.Format_Grayscale8)
            # pixmap = QPixmap.fromImage(img)
            # self.label_a.setPixmap(pixmap)

            # height, width = intensity_matrix.shape[:2]
            # print(intensity_matrix.shape[:2])

            # TESTING IMAGEVIEWER
            coordinates = pg.QtCore.QRectF(0,0, width - 1, height - 1)
            self.plot_viewer.setImage(intensity_matrix, autoRange=False, autoLevels=True)
            self.plot_viewer.imageItem.setRect(rect=coordinates)
            
            # USING PLOT DOES NOT WORK OUT OF THE BOX

            

            # self.plot.plot(x=histx, y=histy, clear=True)

    def init_ui(self):
        # self.label_a = QLabel()
        # self.grid_a.addWidget(self.label_a,0,0)
        self.plot = pg.PlotItem()
        self.plot_viewer = pg.ImageView(view=self.plot,)
        self.grid_a.addWidget(self.plot_viewer,0,0)

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
