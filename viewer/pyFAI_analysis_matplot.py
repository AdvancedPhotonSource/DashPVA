import sys
import numpy as np
import pyFAI
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pvaccess as pva
import logging
import cv2  # Ensure OpenCV is installed if you use it for image resizing
import fabio  # Ensure fabio is installed for mask loading
import time

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a StreamHandler if none exist
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class PyFAIAnalysisWindow(QMainWindow):
    def __init__(self, poni_file="pyFAI/2022-3_calib.poni", mask_file=None, expected_image_shape=(2048, 2048), parent=None):
        super(PyFAIAnalysisWindow, self).__init__(parent)
        self.setWindowTitle("pyFAI Diffraction Integration")
        self.poni_file = poni_file
        self.expected_image_shape = expected_image_shape  # Define expected image shape here

        # Load PONI file
        try:
            self.ai = pyFAI.load(self.poni_file)
            logger.debug(f"Loaded PONI file: {self.poni_file}")
        except Exception as e:
            logger.error(f"Error loading PONI file '{self.poni_file}': {e}")
            raise RuntimeError(f"Error loading PONI file '{self.poni_file}': {e}")

        # Load the mask if provided
        self.mask = None
        if mask_file:
            try:
                mask_data = fabio.open(mask_file).data
                self.mask = mask_data.astype(bool)  # Ensure mask is boolean
                logger.debug(f"Loaded mask file: {mask_file} with shape {self.mask.shape}")
            except Exception as e:
                logger.error(f"Error loading mask file '{mask_file}': {e}")
                self.mask = None  # Proceed without a mask or handle accordingly

        self._latest_image = None

        self.initUI()

        # Setup PV subscription
        self.channel = pva.Channel("pvapy:image", pva.PVA)
        self.channel.subscribe("update", self.pva_callback)
        self.channel.startMonitor()
        logger.debug("Started PV channel monitor.")

    def initUI(self):
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        layout = QVBoxLayout(self.main_widget)
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        # Initialize an empty line for the integration result
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel("Q (Å⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Azimuthal Integration on Each Frame")
        self.ax.grid(True)

    def pva_callback(self, pv_object, offset=None):
        """
        Callback function invoked on every PV update.
        It reads in the incoming frame, extracts the image data,
        and then performs pyFAI integration on it.
        """
        try:
            if 'value' in pv_object:
                value = pv_object['value']
                timestamp = time.time()
                logger.debug(f"Received PV object at {timestamp}: {value}")

                # Check if 'value' is a list or similar iterable
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 1:
                    first_element = value[0]
                    if isinstance(first_element, dict) and 'floatValue' in first_element:
                        image_data = first_element['floatValue']
                        image = np.array(image_data, dtype=np.float32)
                        logger.debug(f"Extracted image data type: {image.dtype}, shape: {image.shape}, ndim: {image.ndim}")

                        # Log a small sample of the data for inspection
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
        Performs azimuthal integration on the incoming frame and
        updates the plot accordingly.
        """
        try:
            # Log the image properties
            logger.debug(f"Processing image with shape: {image.shape}, dtype: {image.dtype}, ndim: {image.ndim}")

            # Ensure the incoming image is 2D.
            if image.ndim != 2:
                logger.warning("Incoming PV image data is not 2D. Attempting to reshape or convert.")
                if image.ndim == 1:
                    # Attempt to reshape to expected dimensions
                    if image.size == self.expected_image_shape[0] * self.expected_image_shape[1]:
                        image = image.reshape(self.expected_image_shape)
                        logger.debug(f"Reshaped 1D image to 2D with shape: {image.shape}")
                    else:
                        logger.error(f"Incoming 1D PV image data cannot be reshaped to the expected shape {self.expected_image_shape}.")
                        self.handle_invalid_image()
                        return
                elif image.ndim == 3:
                    # Example: Convert RGB to grayscale by averaging channels
                    logger.debug("Incoming image is 3D. Converting to grayscale by averaging channels.")
                    image = np.mean(image, axis=2)
                    logger.debug(f"Converted 3D image to 2D with shape: {image.shape}")
                else:
                    logger.error(f"Unsupported image dimensions: {image.ndim}")
                    self.handle_invalid_image()
                    return
            else:
                logger.debug("Incoming image is already 2D.")

            # Apply mask if available
            if self.mask is not None:
                if self.mask.shape != image.shape:
                    logger.warning(f"Mask shape {self.mask.shape} does not match image shape {image.shape}. Resizing mask.")
                    image_height, image_width = image.shape
                    resized_mask = cv2.resize(self.mask.astype(np.uint8), (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    self.mask = resized_mask.astype(bool)
                    logger.debug(f"Resized mask to {self.mask.shape}")
                mask = self.mask
                logger.debug("Applying mask to the image.")
            else:
                mask = None  # No mask applied
                logger.debug("No mask applied to the image.")

            # Perform azimuthal integration
            q, intensity = self.ai.integrate1d(image, 1000, mask=mask, unit="q_A^-1", method='bbox')
            logger.debug(f"Azimuthal integration successful. Q range: {q.min()} to {q.max()}, Intensity range: {intensity.min()} to {intensity.max()}")

            # Update the plot with the new azimuthal integration result.
            self.line.set_data(q, intensity)
            self.ax.relim()
            self.ax.autoscale_view()
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
        Handles invalid images by logging and optionally notifying the user.
        """
        logger.warning("Received invalid image data. Skipping this frame.")
        # Optionally, implement further actions like notifying the user via UI.
        # For example, you could display a temporary message or increment a missed frames counter.

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyFAIAnalysisWindow()
    window.show()
    sys.exit(app.exec_()) 