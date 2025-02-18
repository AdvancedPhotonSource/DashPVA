import sys
import numpy as np
import pyFAI
from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QTextEdit, QFileDialog, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pvaccess as pva
import logging
import cv2  # for image/mask resizing if needed
import fabio  # for mask loading if needed
import time

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure at least one handler is defined.
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class PyFAIAnalysisWindow(QMainWindow):
    def __init__(self, expected_image_shape=(2048, 2048), parent=None):
        super(PyFAIAnalysisWindow, self).__init__(parent)
        self.setWindowTitle("pyFAI Diffraction Integration - Interactive PyQt")
        self.expected_image_shape = expected_image_shape  # expected shape for incoming images

        self.ai = None        # Calibration object (loaded from PONI file)
        self.poni_file = None  # PONI file path string
        self.mask = None      # Optional mask (if used)
        self.paused = False   # Pause flag for image updates
        self._latest_image = None
        self.frame_count = 0  # Initialize frame counter

        self.initUI()

        # Setup PV subscription (assuming it always delivers images)
        self.channel = pva.Channel("pvapy:image", pva.PVA)
        self.channel.subscribe("update", self.pva_callback)
        self.channel.startMonitor()
        logger.debug("Started PV channel monitor with interactive UI.")

    def initUI(self):
        """Initializes the UI with a left canvas for plotting and a right sidebar for controls and metadata."""
        # Create a central widget with horizontal layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: matplotlib FigureCanvas as the plotting area
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        # Initialize an empty line for the integration result
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel("Q (Å⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Azimuthal Integration on Each Frame")
        self.ax.grid(True)
        main_layout.addWidget(self.canvas, stretch=3)

        # Right side: Sidebar panel with buttons and metadata text box
        sidebar = QWidget(self)
        sidebar_layout = QVBoxLayout(sidebar)

        # Button to load the PONI file (will open a file dialog)
        self.load_poni_button = QPushButton("Load PONI File", self)
        self.load_poni_button.clicked.connect(self.load_poni)
        sidebar_layout.addWidget(self.load_poni_button)

        # Button to pause/resume the processing of images
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.toggle_pause)
        sidebar_layout.addWidget(self.pause_button)

        # Button to save the current plot into an image file
        self.save_button = QPushButton("Save Plot", self)
        self.save_button.clicked.connect(self.save_plot)
        sidebar_layout.addWidget(self.save_button)

        # Text area to show the metadata from the calibration file
        self.metadata_panel = QTextEdit(self)
        self.metadata_panel.setReadOnly(True)
        self.metadata_panel.setMinimumWidth(200)
        sidebar_layout.addWidget(self.metadata_panel, stretch=1)

        sidebar_layout.addStretch()  # push items to the top
        main_layout.addWidget(sidebar, stretch=1)

    def load_poni(self):
        """
        Opens a file dialog to choose a PONI file and loads it using pyFAI.
        On success, it updates the calibration object and prints metadata on the sidebar.
        """
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self,
                                                  "Select PONI File",
                                                  "",
                                                  "PONI Files (*.poni);;All Files (*)",
                                                  options=options)
        if filePath:
            try:
                self.ai = pyFAI.load(filePath)
                self.poni_file = filePath
                logger.debug(f"Loaded PONI file: {filePath}")
                self.update_metadata()
            except Exception as e:
                logger.error(f"Error loading PONI file '{filePath}': {e}")
                QMessageBox.critical(self, "Error", f"Failed to load PONI file:\n{e}")
        else:
            logger.info("PONI file selection canceled.")

    def update_metadata(self):
        """
        Gathers metadata from the calibration (pyFAI) object and displays it in the sidebar.
        This includes parameters like distance, detector offsets, rotations, pixel sizes, and wavelength.
        """
        if self.ai is None:
            self.metadata_panel.setPlainText("No calibration loaded.")
            return

        metadata_str = f"PONI File: {self.poni_file}\n"
        # List common parameters if available
        metadata_items = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3",
                          "pixel1", "pixel2", "centerX", "centerY", "wavelength"]
        for item in metadata_items:
            value = getattr(self.ai, item, None)
            if value is not None:
                metadata_str += f"{item}: {value}\n"
        self.metadata_panel.setPlainText(metadata_str)

    def toggle_pause(self):
        """
        Toggles the pause state. When paused, the PV callback will skip updating the plot.
        """
        self.paused = not self.paused
        if self.paused:
            self.pause_button.setText("Resume")
            logger.debug("Paused image updates.")
        else:
            self.pause_button.setText("Pause")
            logger.debug("Resumed image updates.")

    def save_plot(self):
        """
        Opens a file dialog to save the current plot as an image file (PNG by default).
        """
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self,
                                                  "Save Plot",
                                                  "",
                                                  "PNG Files (*.png);;All Files (*)",
                                                  options=options)
        if filePath:
            try:
                self.figure.savefig(filePath)
                logger.debug(f"Saved plot to {filePath}")
            except Exception as e:
                logger.error(f"Error saving plot: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")

    def pva_callback(self, pv_object, offset=None):
        """
        Callback function invoked on every PV update.
        It reads in the incoming frame, extracts the image data,
        and then performs pyFAI integration on it if not paused.
        """
        if self.paused:
            logger.debug("Image update is paused. Skipping frame.")
            return

        try:
            if 'value' in pv_object:
                value = pv_object['value']
                timestamp = time.time()
                logger.debug(f"Received PV object at {timestamp}: {value}")

                # Check if 'value' is a list/tuple-like and process the first element
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 1:
                    first_element = value[0]
                    if isinstance(first_element, dict) and 'floatValue' in first_element:
                        image_data = first_element['floatValue']
                        image = np.array(image_data, dtype=np.float32)
                        logger.debug(f"Extracted image data type: {image.dtype}, shape: {image.shape}, ndim: {image.ndim}")

                        # Log a short sample of the data for debugging
                        if image.size > 10:
                            logger.debug(f"Image data sample: {image.flatten()[:10]}...")
                        else:
                            logger.debug(f"Image data sample: {image.flatten()}")

                        self.update_image(image)
                    else:
                        logger.error("First element of 'value' does not contain 'floatValue'. Invalid data structure.")
                        self.handle_invalid_image()
                else:
                    logger.error("PV 'value' is not a list/tuple with at least one element. Invalid data structure.")
                    self.handle_invalid_image()
            else:
                logger.warning("PV object does not contain 'value' key.")
        except Exception as e:
            logger.error(f"Error processing PV object: {e}")
            self.handle_invalid_image()

    def update_image(self, image):
        """
        Performs azimuthal integration on the incoming frame and updates the plot.
        If the image is not 2D, attempts to reshape or convert accordingly.
        """
        if self.ai is None:
            logger.warning("No calibration loaded. Ignoring image update.")
            return

        try:
            logger.debug(f"Processing image with shape: {image.shape}, dtype: {image.dtype}, ndim: {image.ndim}")

            # Ensure the incoming image is 2D.
            if image.ndim != 2:
                logger.warning("Incoming image data is not 2D. Attempting to reshape or convert.")
                if image.ndim == 1:
                    # Check if the 1D image can be reshaped to the expected dimensions
                    if image.size == self.expected_image_shape[0] * self.expected_image_shape[1]:
                        image = image.reshape(self.expected_image_shape)
                        logger.debug(f"Reshaped 1D image to 2D with shape: {image.shape}")
                    else:
                        logger.error(f"Incoming 1D image data cannot be reshaped to expected shape {self.expected_image_shape}.")
                        self.handle_invalid_image()
                        return
                elif image.ndim == 3:
                    # For example: Convert colored (RGB) 3D images to grayscale by averaging channels
                    logger.debug("Incoming image is 3D. Converting to grayscale by averaging channels.")
                    image = np.mean(image, axis=2)
                    logger.debug(f"Converted 3D image to 2D with shape: {image.shape}")
                else:
                    logger.error(f"Unsupported image dimensions: {image.ndim}")
                    self.handle_invalid_image()
                    return
            else:
                logger.debug("Incoming image is already 2D.")

            # If a mask is provided, ensure its shape matches
            if self.mask is not None:
                if self.mask.shape != image.shape:
                    logger.warning(f"Mask shape {self.mask.shape} does not match image shape {image.shape}. Resizing mask.")
                    image_height, image_width = image.shape
                    resized_mask = cv2.resize(self.mask.astype(np.uint8),
                                              (image_width, image_height),
                                              interpolation=cv2.INTER_NEAREST)
                    self.mask = resized_mask.astype(bool)
                    logger.debug(f"Resized mask to {self.mask.shape}")
                mask = self.mask
                logger.debug("Applying mask to the image.")
            else:
                mask = None
                logger.debug("No mask applied to the image.")

            # Perform azimuthal integration with pyFAI
            q, intensity = self.ai.integrate1d(image, 1000, mask=mask, unit="q_A^-1", method='bbox')
            logger.debug(f"Azimuthal integration successful. Q range: {q.min()} to {q.max()}, "
                         f"Intensity range: {intensity.min()} to {intensity.max()}")

            # Update the matplotlib plot data
            self.line.set_data(q, intensity)
            self.ax.relim()
            self.ax.autoscale_view()

            # Increment frame counter and update the title with frame number and current time
            self.frame_count += 1
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.ax.set_title(f"Frame: {self.frame_count} @ {current_time}")

            self.canvas.draw_idle()
            logger.debug("Plot updated with new azimuthal integration data.")

        except ValueError as ve:
            logger.error(f"ValueError during image processing: {ve}")
            self.handle_invalid_image()
        except Exception as e:
            logger.error(f"Unexpected error during integration: {e}")
            self.handle_invalid_image()

    def handle_invalid_image(self):
        """
        Handles invalid images by logging warnings.
        Additional user feedback could be added here (e.g. in the metadata panel or statusbar).
        """
        logger.warning("Received invalid image data. Skipping this frame.")
        current_text = self.metadata_panel.toPlainText()
        new_text = current_text + "\nInvalid image received."
        self.metadata_panel.setPlainText(new_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyFAIAnalysisWindow()
    window.show()
    sys.exit(app.exec_())