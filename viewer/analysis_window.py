import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QTimer
from PyQt5 import uic


# Define the second window as a class
class AnalysisWindow(QMainWindow):
    def __init__(self,pipe):
        super(AnalysisWindow, self).__init__()
        self.pipe = pipe # used to share memory with another process
        uic.loadUi('gui/analysis_window.ui', self)
        self.setWindowTitle('Analysis Window')
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_check_message)
        self.timer.start(1000)

    def timer_check_message(self):
        message = self.pipe.recv()
        self.label_a.setText(message)
    
    def init_ui(self):
        self.label_a = QLabel()
        self.grid_a.addWidget(self.label_a,0,0)

    def closeEvent(self, event):
        self.pipe.send('close')
        event.accept()
        # super(AnalysisWindow, self).closeEvent()

# Global function so it can be called without needing to be within a class and get around
# not being able to pass arguments to pyqt slots
def analysis_window_process(pipe):
    app = QApplication(sys.argv)
    window = AnalysisWindow(pipe)
    window.show()
    app.exec_()
    # pipe.send('close')
    pipe.close()

if __name__ == '__main__':
    import multiprocessing as mp
    parent_pipe, child_pipe = mp.Pipe()
    p = mp.Process(target=analysis_window_process, args=(child_pipe,))
    
    p.start()
    parent_pipe.send("Hello from the main process!")
    try:
        while True:
            # Handling messages from the main process if necessary
            if parent_pipe.poll():
                message = parent_pipe.recv()
                if message == 'close':
                    break
    finally:
        p.join()
