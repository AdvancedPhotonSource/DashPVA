import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QTimer
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
import numpy as np


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
        self.timer.start(1000/100)
        self.pv_dict = None

    def timer_poll_pipe(self):
        if self.pipe.poll():
            self.pv_dict = self.pipe.recv() # try to send uniqueID
            image = self.pv_dict[list(self.pv_dict.keys())[0]]
            pixmap = QPixmap()
            pixmap.convertFromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0],QImage.Format.Format_RGB888))
            self.label_a.setPixmap(pixmap)
    
    def init_ui(self):
        self.label_a = QLabel()
        self.grid_a.addWidget(self.label_a,0,0)

    def closeEvent(self, event):
        self.pipe.send('close')
        event.accept()
        # super(AnalysisWindow, self).closeEvent(event)

# Global function so it can be called without needing to be within a class and get around
# not being able to pass arguments to pyqt slots
def analysis_window_process(pipe):
    app = QApplication(sys.argv)
    window = AnalysisWindow(pipe)
    window.show()
    app.exec_()

if __name__ == '__main__':
    import multiprocessing as mp
    parent_pipe, child_pipe = mp.Pipe()
    p = mp.Process(target=analysis_window_process, args=(child_pipe,))
    
    p.start()
    parent_pipe.send({1000:np.zeros((100,100))})
    try:
        while True:
            # Handling messages from the main process if necessary
            if parent_pipe.poll():
                message = parent_pipe.recv()
                if message == 'close':
                    break
    finally:
        p.join()
