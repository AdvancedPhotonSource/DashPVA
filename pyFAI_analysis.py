import sys
import time
import numpy as np
import pyFAI
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PyFAIAnalysisWindow(QMainWindow):
    def __init__(self, poni_file="2022-3_calib.poni", parent=None):
        super(PyFAIAnalysisWindow, self).__init__(parent)
        self.setWindowTitle("pyFAI Diffraction Integration")
        self.poni_file = poni_file
        try:
            self.ai = pyFAI.load(self.poni_file)
        except Exception as e:
            raise RuntimeError(f"Error loading PONI file '{poni_file}': {e}")
        self._latest_image = None
        self._update_interval_ms = 100  # update at 10 Hz

        self._initUI()
        self.startTimer(self._update_interval_ms)

    def _initUI(self):
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel("q (Å⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Integrated Diffraction Pattern")
        self.ax.grid(True)

    def timerEvent(self, event):
        # This timer fires at a 10 Hz rate and updates the diffraction integration plot.
        if self._latest_image is not None:
            try:
                # integrate the image into a 1D diffraction profile; using 1000 bins here
                q, intensity = self.ai.integrate1d(self._latest_image, 1000, mask=None, unit="q_A^-1", method='bbox')
                self.line.set_data(q, intensity)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw_idle()
            except Exception as e:
                print("Error during integration:", e)

    def update_image(self, image):
        """
        This method is to be called from the simulation whenever a new
        image is produced.
        """
        self._latest_image = image

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyFAIAnalysisWindow()
    window.show()
    sys.exit(app.exec_()) 