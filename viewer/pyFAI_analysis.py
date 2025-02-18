import sys
import numpy as np
import pyFAI
from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QTextEdit, QFileDialog, QMessageBox, QLabel, QSpinBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import pvaccess as pva
import logging
import cv2  # for image/mask resizing if needed
import fabio  # for mask loading if needed
import time
import json

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

        # Initialize data storage for waterfall plot
        self.waterfall_data = []  # Will store intensity data for each frame
        self.q_values = None      # Will store Q values (x-axis)
        self.max_frames = 100     # Maximum number of frames to show in waterfall

        # IMPORTANT: Build the UI first so that all UI elements (metadata_panel, status bar, etc.) are available.
        self.initUI()

        # Load configuration data and automatically load the last used PONI file, if available.
        self.load_config()
        if "last_poni_file" in self.config:
            last_file = self.config["last_poni_file"]
            try:
                self.ai = pyFAI.load(last_file)
                self.poni_file = last_file
                logger.debug(f"Auto-loaded last PONI file: {last_file}")
                # Update metadata panel with calibration stats
                self.update_metadata()
                # Also update the status bar with an appropriate message.
                self.statusBar().showMessage(f"Calibration loaded from: {last_file}")
            except Exception as e:
                logger.error(f"Failed to auto-load last PONI file '{last_file}': {e}")

        # Setup PV subscription (assuming it always delivers images)
        self.channel = pva.Channel("pvapy:image", pva.PVA)
        self.channel.subscribe("update", self.pva_callback)
        self.channel.startMonitor()
        logger.debug("Started PV channel monitor with interactive UI.")

    def initUI(self):
        """Initializes the UI with a top waterfall plot and the existing components."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top section: Waterfall plot
        waterfall_widget = QWidget()
        waterfall_layout = QHBoxLayout(waterfall_widget)
        
        # Create waterfall plot
        self.waterfall_figure = Figure(figsize=(6, 2))
        self.waterfall_canvas = FigureCanvas(self.waterfall_figure)
        self.waterfall_ax = self.waterfall_figure.add_subplot(111)
        self.waterfall_ax.set_xlabel("Q (Å⁻¹)")
        self.waterfall_ax.set_ylabel("Frame Number")
        self.waterfall_image = None
        waterfall_layout.addWidget(self.waterfall_canvas)
        
        # Add waterfall plot to main layout
        main_layout.addWidget(waterfall_widget)
        
        # Middle section: Original plot area
        plot_widget = QWidget()
        plot_layout = QHBoxLayout(plot_widget)
        
        # Create the original line plot
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel("Q (Å⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Current Frame Integration")
        self.ax.grid(True)
        plot_layout.addWidget(self.canvas, stretch=3)
        
        # Add the original plot to main layout
        main_layout.addWidget(plot_widget)
        
        # Right side: Sidebar panel
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        
        # Create a widget for PONI file and max frames controls
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        # Button to load the PONI file
        self.load_poni_button = QPushButton("Load PONI File", self)
        self.load_poni_button.clicked.connect(self.load_poni)
        controls_layout.addWidget(self.load_poni_button)

        # Add max frames spinbox
        max_frames_label = QLabel("Max Frames:", self)
        self.max_frames_spinbox = QSpinBox(self)
        self.max_frames_spinbox.setRange(10, 10000)
        self.max_frames_spinbox.setValue(300)
        self.max_frames_spinbox.setSingleStep(50)
        self.max_frames_spinbox.valueChanged.connect(self.update_max_frames)
        
        controls_layout.addWidget(max_frames_label)
        controls_layout.addWidget(self.max_frames_spinbox)
        
        # Add controls widget to sidebar
        sidebar_layout.addWidget(controls_widget)

        # Button to pause/resume
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.toggle_pause)
        sidebar_layout.addWidget(self.pause_button)

        # Button to save plot
        self.save_button = QPushButton("Save Plot", self)
        self.save_button.clicked.connect(self.save_plot)
        sidebar_layout.addWidget(self.save_button)

        # Text area for metadata
        self.metadata_panel = QTextEdit(self)
        self.metadata_panel.setReadOnly(True)
        self.metadata_panel.setMinimumWidth(200)
        sidebar_layout.addWidget(self.metadata_panel, stretch=1)

        sidebar_layout.addStretch()  # push items to the top
        plot_layout.addWidget(sidebar, stretch=1)

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
                # Update configuration with the new PONI file path
                self.config["last_poni_file"] = filePath
                self.save_config()
                logger.debug(f"Loaded PONI file: {filePath}")
                self.update_metadata()
            except Exception as e:
                logger.error(f"Error loading PONI file '{filePath}': {e}")
                QMessageBox.critical(self, "Error", f"Failed to load PONI file:\n{e}")
        else:
            logger.info("PONI file selection canceled.")

    def update_metadata(self):
        """
        Gathers metadata from the calibration (pyFAI) object and displays it in the metadata panel.
        This includes parameters like distance, detector offsets, rotations, pixel sizes, and wavelength.
        """
        if self.ai is None:
            self.metadata_panel.setPlainText("No calibration loaded.")
            return

        # Prepare a string that summarizes the auto-loaded calibration.
        metadata_str = f"Calibration loaded from: {self.poni_file}\n\n"
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

            # Update both plots
            self.line.set_data(q, intensity)
            self.ax.relim()
            self.ax.autoscale_view()
            
            # Update waterfall plot
            self.update_waterfall_plot(q, intensity)
            
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

    def load_config(self):
        """
        Loads application configuration from a JSON file.
        Currently stores information such as the last used PONI file.
        """
        self.config_file = "config.json"
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
                logger.debug(f"Configuration loaded: {self.config}")
        except Exception as e:
            logger.debug(f"No configuration found or could not load config: {e}")
            self.config = {}

    def save_config(self):
        """
        Saves the current application configuration to a JSON file.
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
                logger.debug(f"Configuration saved: {self.config}")
        except Exception as e:
            logger.error(f"Could not save configuration: {e}")

    def update_waterfall_plot(self, q, intensity):
        """Updates the waterfall plot with new intensity data."""
        if self.q_values is None:
            self.q_values = q
            
        # Add new intensity data
        self.waterfall_data.append(intensity)
        
        # Keep only the last max_frames
        if len(self.waterfall_data) > self.max_frames:
            self.waterfall_data.pop(0)
            
        # Convert data to 2D array for plotting
        data_array = np.array(self.waterfall_data)
        
        # Clear previous plot
        self.waterfall_ax.clear()
        
        # Create the contour plot
        if len(self.waterfall_data) > 1:
            frame_numbers = np.arange(len(self.waterfall_data))
            X, Y = np.meshgrid(self.q_values, frame_numbers)
            
            # Create contour plot with log scale for better visualization
            self.waterfall_image = self.waterfall_ax.pcolormesh(
                X, Y, data_array, 
                shading='auto',
                cmap='viridis',
                norm=LogNorm(vmin=data_array.min(), vmax=data_array.max())
            )
            
            # Set labels and title
            self.waterfall_ax.set_xlabel("Q (Å⁻¹)")
            self.waterfall_ax.set_ylabel("Frame Number")
            
            # Add colorbar if it doesn't exist
            if not hasattr(self, 'waterfall_colorbar'):
                self.waterfall_colorbar = self.waterfall_figure.colorbar(
                    self.waterfall_image, 
                    ax=self.waterfall_ax
                )
                self.waterfall_colorbar.set_label('Intensity')
        
        self.waterfall_canvas.draw_idle()

    def update_max_frames(self):
        """Updates the max_frames attribute with the value from the spinbox."""
        self.max_frames = self.max_frames_spinbox.value()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyFAIAnalysisWindow()
    window.show()
    sys.exit(app.exec_())