import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtCore import QTimer
from PyQt5 import uic, QtGui


# Define the second window as a class
class AnalysisWindow(QMainWindow):
    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe # used to share memory with another process
        uic.loadUi('gui/analysi_window', self)
        self.setWindowTitle('Analysis Window')
        self.show()
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_check_message)
        self.timer.start(1000/10)

    def timer_check_message(self):
        message = self.pipe.recv()
        self.label_a.setText(message)
    
    def init_ui(self):
        self.label_a = QLabel()
        self.grid_a.addWidget(self.label_a,0,0)

# Global function so it can be called without needing to be within a class and get around
# not being able to pass arguments to pyqt slots
def analysis_window_process(pipe):
    app = QApplication(sys.argv)
    window = AnalysisWindow(pipe)
    window.show()
    sys.exit(app.exec_())
