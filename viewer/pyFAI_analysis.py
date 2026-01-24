
import os
os.environ.pop('QT_PLUGIN_PATH', None)  # Unset QT_PLUGIN_PATH to prevent conflicts

import sys
# Fix namespace conflict: local pyFAI/ directory shadows the installed package
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root in sys.path:
    sys.path.remove(_project_root)

import numpy as np
import pyFAI

# Create pyFAI.load wrapper if it doesn't exist (for compatibility with dev_11ID code)
if not hasattr(pyFAI, 'load'):
    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        def _load_poni(poni_file):
            ai = AzimuthalIntegrator()
            ai.load(poni_file)
            return ai
        pyFAI.load = _load_poni
    except ImportError:
        try:
            from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
            def _load_poni(poni_file):
                ai = AzimuthalIntegrator()
                ai.load(poni_file)
                return ai
            pyFAI.load = _load_poni
        except ImportError:
            pass  # Will fail later with better error message

# Restore project root to path (needed for other local imports)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QTextEdit, QFileDialog, QMessageBox, QLabel, QSpinBox, QProgressDialog, QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import pvaccess as pva
import logging
# import cv2  # for image/mask resizing if needed
from skimage.transform import resize
import fabio  # for mask loading if needed
import time
import json
import h5py
import shutil
try:
    import hdf5plugin
    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False
    logger.warning("hdf5plugin not available. Compression options will be limited.")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Ensure at least one handler is defined.
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class SaveWorker(QThread):
    """Worker thread for saving HDF5 data to prevent UI freezing."""
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, filename, image_cache, integration_data, waterfall_data, 
                 q_values, pv_address, pv_metadata, ai, poni_file, frame_count, max_frames, compression='blosc lz4'):
        super(SaveWorker, self).__init__()
        self.filename = filename
        self.image_cache = image_cache
        self.integration_data = integration_data
        self.waterfall_data = waterfall_data
        self.q_values = q_values
        self.pv_address = pv_address
        self.pv_metadata = pv_metadata
        self.ai = ai
        self.poni_file = poni_file
        self.frame_count = frame_count
        self.max_frames = max_frames
        self.compression = compression
    
    def run(self):
        """Perform the save operation in background thread."""
        try:
            # Convert image cache to numpy array
            images_array = np.array(self.image_cache)
            
            # Extract q and intensity arrays from integration_data
            q_arrays = []
            intensity_arrays = []
            for q, intensity in self.integration_data:
                q_arrays.append(q)
                intensity_arrays.append(intensity)
            
            # Convert to numpy arrays
            q_array = np.array(q_arrays) if q_arrays else None
            intensity_array = np.array(intensity_arrays) if intensity_arrays else None
            
            # Get waterfall data as 2D array
            waterfall_array = np.array(self.waterfall_data) if self.waterfall_data else None
            
            # Create HDF5 file with standard structure
            with h5py.File(self.filename, 'w') as h5f:
                # Create entry group
                entry_grp = h5f.create_group("entry")
                
                # Save raw diffraction images with selected compression
                data_grp = entry_grp.create_group("data")
                
                # Apply compression based on selection (only to images dataset)
                compression_kwargs = {}
                if HDF5PLUGIN_AVAILABLE:
                    if self.compression == 'blosc':
                        compression_kwargs = hdf5plugin.Blosc(cname='blosclz', clevel=5, shuffle=True)
                    elif self.compression == 'blosc lz4':
                        compression_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True)
                    elif self.compression == 'lz4':
                        compression_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True)
                    else:
                        # Fallback to gzip for unknown compression
                        compression_kwargs = {'compression': 'gzip', 'compression_opts': 4}
                else:
                    # Fallback to gzip if hdf5plugin not available
                    compression_kwargs = {'compression': 'gzip', 'compression_opts': 4}
                    logger.warning(f"hdf5plugin not available. Using gzip compression instead of {self.compression}.")
                
                data_grp.create_dataset("images", data=images_array, **compression_kwargs)
                data_grp["images"].attrs['description'] = 'Raw diffraction images (cached based on max_frames)'
                data_grp["images"].attrs['shape'] = images_array.shape
                data_grp["images"].attrs['dtype'] = str(images_array.dtype)
                data_grp["images"].attrs['compression'] = self.compression
                
                # Save 1D integration data
                analysis_grp = entry_grp.create_group("analysis")
                if q_array is not None:
                    analysis_grp.create_dataset("q_values", data=q_array, compression='gzip', compression_opts=4)
                    analysis_grp["q_values"].attrs['description'] = 'Q values (Å⁻¹) for each frame'
                if intensity_array is not None:
                    analysis_grp.create_dataset("intensity", data=intensity_array, compression='gzip', compression_opts=4)
                    analysis_grp["intensity"].attrs['description'] = 'Integrated intensity for each frame'
                
                # Save waterfall plot as 2D array
                if waterfall_array is not None:
                    analysis_grp.create_dataset("waterfall", data=waterfall_array, compression='gzip', compression_opts=4)
                    analysis_grp["waterfall"].attrs['description'] = 'Waterfall plot data (2D array: frames x q_points)'
                    if self.q_values is not None:
                        analysis_grp.create_dataset("waterfall_q_values", data=self.q_values, compression='gzip', compression_opts=4)
                        analysis_grp["waterfall_q_values"].attrs['description'] = 'Q values for waterfall plot'
                
                # Save PV metadata
                metadata_grp = entry_grp.create_group("metadata")
                if self.pv_address:
                    metadata_grp.attrs['pv_address'] = self.pv_address
                if self.pv_metadata:
                    for key, value in self.pv_metadata.items():
                        try:
                            if isinstance(value, dict):
                                # Store nested dict as attributes with prefix
                                for subkey, subvalue in value.items():
                                    metadata_grp.attrs[f'{key}_{subkey}'] = subvalue
                            else:
                                metadata_grp.attrs[key] = value
                        except Exception as e:
                            logger.warning(f"Could not save metadata key {key}: {e}")
                
                # Save pyFAI calibration info
                if self.ai is not None:
                    calib_grp = metadata_grp.create_group("calibration")
                    calib_items = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3",
                                  "pixel1", "pixel2", "centerX", "centerY", "wavelength"]
                    for item in calib_items:
                        value = getattr(self.ai, item, None)
                        if value is not None:
                            calib_grp.attrs[item] = value
                    if self.poni_file:
                        calib_grp.attrs['poni_file_path'] = self.poni_file
                
                # Save frame information
                metadata_grp.attrs['total_frames'] = self.frame_count
                metadata_grp.attrs['cached_frames'] = len(self.image_cache)
                metadata_grp.attrs['max_frames'] = self.max_frames
                metadata_grp.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Copy PONI file to the same directory as the HDF5 file
            poni_message = ""
            if self.poni_file and os.path.exists(self.poni_file):
                poni_dest = os.path.join(os.path.dirname(self.filename), os.path.basename(self.poni_file))
                try:
                    shutil.copy2(self.poni_file, poni_dest)
                    logger.debug(f"Copied PONI file to {poni_dest}")
                    poni_message = f"\nPONI file copied."
                except Exception as e:
                    logger.warning(f"Could not copy PONI file: {e}")
                    poni_message = f"\nWarning: Could not copy PONI file: {e}"
            
            success_msg = (f"All data saved successfully to:\n{self.filename}\n\n"
                          f"Saved {len(self.image_cache)} frames.{poni_message}")
            self.finished.emit(True, success_msg)
            
        except Exception as e:
            error_msg = f"Failed to save all data:\n{e}"
            logger.error(error_msg)
            self.finished.emit(False, error_msg)

class PyFAIAnalysisWindow(QMainWindow):
    # Define signals for thread-safe communication from PV callback to GUI thread
    # Use 'object' type for numpy arrays to avoid registration issues
    image_received = pyqtSignal(object)  # Signal emitted when image is received (np.ndarray)
    error_occurred = pyqtSignal(str)  # Signal emitted when error occurs
    
    def __init__(self, expected_image_shape=(2048, 2048), parent=None, pv_address=None):
        try:
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
            
            # Initialize data storage for saving all data
            self.image_cache = []  # Will store raw diffraction images (cached based on max_frames)
            self.integration_data = []  # Will store (q, intensity) tuples for each frame
            self.pv_address = pv_address or "pvapy:image"  # PV address from parameter or default
            self.pv_metadata = {}  # Additional PV metadata
            self.save_worker = None  # Worker thread for saving
            self.error_count = 0  # Track consecutive errors
            self.last_error_message = None  # Store last error message for display
            self._processing_image = False  # Flag to throttle updates

            # IMPORTANT: Build the UI first so that all UI elements (metadata_panel, status bar, etc.) are available.
            self.initUI()
            
            # Connect signals to slots for thread-safe GUI updates (auto connection type)
            self.image_received.connect(self.update_image)
            self.error_occurred.connect(self._handle_error_signal)
        except Exception as e:
            # If initialization fails, set error message but don't crash
            self.last_error_message = f"Initialization error: {str(e)}"
            logger.critical(f"Error during PyFAIAnalysisWindow initialization: {e}", exc_info=True)
            # Try to at least set basic attributes
            if not hasattr(self, 'pv_address'):
                self.pv_address = pv_address or "pvapy:image"
            if not hasattr(self, 'error_count'):
                self.error_count = 0
            if not hasattr(self, 'last_error_message'):
                self.last_error_message = str(e)

        # Load configuration data and automatically load the last used PONI file, if available.
        self.load_config()
        poni_loaded = False
        if "last_poni_file" in self.config:
            last_file = self.config["last_poni_file"]
            if os.path.exists(last_file):
                try:
                    self.ai = pyFAI.load(last_file)
                    self.poni_file = last_file
                    logger.debug(f"Auto-loaded last PONI file: {last_file}")
                    # Update metadata panel with calibration stats
                    self.update_metadata()
                    # Also update the status bar with an appropriate message.
                    self.statusBar().showMessage(f"Calibration loaded from: {last_file}")
                    poni_loaded = True
                except Exception as e:
                    logger.error(f"Failed to auto-load last PONI file '{last_file}': {e}")
        
        # If no PONI loaded from config, try default PONI file
        if self.ai is None:
            default_poni = os.path.join(_project_root, 'pyFAI', '2022-3_calib.poni')
            if os.path.exists(default_poni):
                try:
                    self.ai = pyFAI.load(default_poni)
                    self.poni_file = default_poni
                    self.config["last_poni_file"] = default_poni
                    self.save_config()
                    logger.info(f"Auto-loaded default PONI file: {default_poni}")
                    self.update_metadata()
                    self.statusBar().showMessage(f"Calibration loaded from default: {os.path.basename(default_poni)}")
                except Exception as e:
                    logger.warning(f"Failed to auto-load default PONI file '{default_poni}': {e}")

        # Setup PV subscription (assuming it always delivers images)
        try:
            self.channel = pva.Channel(self.pv_address, pva.PVA)
            self.channel.subscribe("update", self.pva_callback)
            self.channel.startMonitor()
            logger.debug(f"Started PV channel monitor with interactive UI. PV address: {self.pv_address}")
            self.statusBar().showMessage(f"Connected to PV: {self.pv_address}")
        except Exception as e:
            error_msg = f"Failed to connect to PV channel '{self.pv_address}': {e}"
            logger.error(error_msg)
            self.statusBar().showMessage(f"ERROR: {error_msg}", 10000)  # Show for 10 seconds
            self._show_error_in_metadata(error_msg)

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
        max_frames_label = QLabel("Max Cached Frames:", self)
        self.max_frames_spinbox = QSpinBox(self)
        self.max_frames_spinbox.setRange(10, 10000)
        self.max_frames_spinbox.setValue(300)
        self.max_frames_spinbox.setSingleStep(50)
        self.max_frames_spinbox.valueChanged.connect(self.update_max_frames)
        # Sync max_frames with spinbox initial value
        self.update_max_frames()
        
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
        
        # Compression selection dropdown and Save All button
        compression_widget = QWidget()
        compression_layout = QHBoxLayout(compression_widget)
        compression_layout.setContentsMargins(0, 0, 0, 0)
        
        compression_label = QLabel("Compression:", self)
        self.compression_combo = QComboBox(self)
        self.compression_combo.addItems(["blosc", "blosc lz4", "lz4"])
        self.compression_combo.setCurrentText("lz4")  # Set default
        compression_layout.addWidget(compression_label)
        compression_layout.addWidget(self.compression_combo)
        sidebar_layout.addWidget(compression_widget)
        
        # Button to save all data
        self.save_all_button = QPushButton("Save All", self)
        self.save_all_button.clicked.connect(self.save_all)
        sidebar_layout.addWidget(self.save_all_button)

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

    def save_all(self):
        """
        Opens a file dialog to select save location and filename, then saves all data to HDF5 file.
        Uses a background thread to prevent UI freezing and shows a progress dialog.
        Saves:
        - All cached diffraction images
        - All 1D integration data (q, intensity) for each frame
        - Waterfall plot as 2D numpy array
        - PV address and metadata
        - Copies PONI file to the same directory
        """
        from pathlib import Path
        
        # Ask for filename (user can navigate to folder and specify name)
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self,
                                                 "Save All Data",
                                                 "",
                                                 "HDF5 Files (*.h5);;All Files (*)",
                                                 options=options)
        if not filename:
            return
        
        # Ensure filename ends with .h5
        if not filename.endswith('.h5'):
            filename += '.h5'
        
        # Check if we have data to save
        if len(self.image_cache) == 0:
            QMessageBox.warning(self, "No Data", "No data to save. Please collect some frames first.")
            return
        
        # Create and show progress dialog
        progress = QProgressDialog("Saving data to HDF5 file...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Saving")
        progress.setWindowModality(2)  # Qt::WindowModal
        progress.setCancelButton(None)  # Don't allow cancellation (saving is in progress)
        progress.setMinimumDuration(0)  # Show immediately
        progress.show()
        QApplication.processEvents()  # Update UI to show progress dialog
        
        # Get selected compression
        compression = self.compression_combo.currentText()
        
        # Create worker thread with copies of data (to avoid issues with thread safety)
        self.save_worker = SaveWorker(
            filename,
            list(self.image_cache),  # Make copies to avoid thread issues
            list(self.integration_data),
            list(self.waterfall_data),
            self.q_values.copy() if self.q_values is not None else None,
            self.pv_address,
            dict(self.pv_metadata),
            self.ai,
            self.poni_file,
            self.frame_count,
            self.max_frames,
            compression
        )
        
        # Connect finished signal to handle completion
        self.save_worker.finished.connect(lambda success, msg: self._on_save_finished(progress, success, msg))
        
        # Start the worker thread
        self.save_worker.start()
    
    def _on_save_finished(self, progress_dialog, success, message):
        """Handle completion of save operation."""
        # Close progress dialog
        progress_dialog.close()
        
        # Show result message
        if success:
            QMessageBox.information(self, "Success", message)
            logger.info(f"Saved all data successfully")
        else:
            QMessageBox.critical(self, "Error", message)
            logger.error(f"Failed to save all data")
        
        # Clean up worker
        self.save_worker = None

    def pva_callback(self, pv_object, offset=None):
        """
        Callback function invoked on every PV update from background thread.
        This runs in the PV monitor thread, so we must use signals to update GUI.
        """
        # Check if window still exists before doing anything
        try:
            if not hasattr(self, 'image_received') or not self.isVisible():
                return
        except RuntimeError:
            # Window has been deleted
            return
        
        if self.paused:
            logger.debug("Image update is paused. Skipping frame.")
            return

        try:
            # Extract PV metadata (store once, update if needed) - this is thread-safe for dict updates
            if self.pv_metadata == {}:
                # Try to extract metadata from PV object
                for key in ['timeStamp', 'alarm', 'display', 'control', 'valueAlarm']:
                    if key in pv_object:
                        try:
                            # Store metadata as JSON-serializable
                            if isinstance(pv_object[key], dict):
                                self.pv_metadata[key] = {k: str(v) for k, v in pv_object[key].items()}
                            else:
                                self.pv_metadata[key] = str(pv_object[key])
                        except Exception:
                            pass
            
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

                        # Emit signal to update GUI in main thread (thread-safe)
                        # Skip if still processing previous frame to avoid queue buildup
                        if not self._processing_image:
                            try:
                                self.image_received.emit(image)
                            except RuntimeError:
                                # Window deleted, ignore
                                return
                    else:
                        error_msg = "PV data structure invalid: 'floatValue' not found. This PV may not contain image data suitable for pyFAI analysis."
                        logger.error(error_msg)
                        try:
                            self.error_occurred.emit(error_msg)
                        except RuntimeError:
                            return
                else:
                    error_msg = "PV 'value' is not in expected format (list/tuple with image data). This PV may not contain diffraction images."
                    logger.error(error_msg)
                    try:
                        self.error_occurred.emit(error_msg)
                    except RuntimeError:
                        return
            else:
                error_msg = "PV object does not contain 'value' key. Check if this is the correct PV for image data."
                logger.warning(error_msg)
                try:
                    self.error_occurred.emit(error_msg)
                except RuntimeError:
                    return
        except RuntimeError as e:
            # Window deleted, ignore
            return
        except Exception as e:
            error_msg = f"Error processing PV object: {e}. Check PV address and data format."
            logger.error(error_msg)
            try:
                self.error_occurred.emit(error_msg)
            except RuntimeError:
                return
    
    def _handle_error_signal(self, error_message):
        """Thread-safe error handler called from GUI thread via signal."""
        self.handle_invalid_image(error_message)

    def update_image(self, image):
        """
        Performs azimuthal integration on the incoming frame and updates the plot.
        If the image is not 2D, attempts to reshape or convert accordingly.
        """
        self._processing_image = True
        try:
            if self.ai is None:
                error_msg = "No PONI calibration file loaded. Please load a PONI file to process images."
                logger.warning(error_msg)
                self.statusBar().showMessage(f"WARNING: {error_msg}", 5000)
                self._show_error_in_metadata(error_msg)
                return
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
                        error_msg = f"Image data cannot be reshaped. Expected shape: {self.expected_image_shape}, got size: {image.size}. This may not be a diffraction image."
                        logger.error(error_msg)
                        self.handle_invalid_image(error_msg)
                        return
                elif image.ndim == 3:
                    # For example: Convert colored (RGB) 3D images to grayscale by averaging channels
                    logger.debug("Incoming image is 3D. Converting to grayscale by averaging channels.")
                    image = np.mean(image, axis=2)
                    logger.debug(f"Converted 3D image to 2D with shape: {image.shape}")
                else:
                    error_msg = f"Unsupported image dimensions: {image.ndim}D. pyFAI requires 2D diffraction images."
                    logger.error(error_msg)
                    self.handle_invalid_image(error_msg)
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
                    
                    resized_mask = resize(self.mask.astype(np.uint8),
                                          (image_height, image_width),
                                          order=0,  # Nearest-neighbor interpolation
                                          preserve_range=True,
                                          anti_aliasing=framePublisher).astype(bool)
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

            # Cache the raw image and integration data
            self.image_cache.append(image.copy())
            self.integration_data.append((q.copy(), intensity.copy()))
            
            # Keep only the last max_frames in cache
            if len(self.image_cache) > self.max_frames:
                self.image_cache.pop(0)
                self.integration_data.pop(0)

            # Reset error count on successful processing
            self.error_count = 0
            self.last_error_message = None

            # Update both plots
            self.line.set_data(q, intensity)
            self.ax.relim()
            self.ax.autoscale_view()
            
            # Update waterfall plot
            self.update_waterfall_plot(q, intensity)
            
            # Store latest image for immediate access
            self._latest_image = image.copy()
            
            # Increment frame counter and update the title with frame number and current time
            self.frame_count += 1
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.ax.set_title(f"Frame: {self.frame_count} @ {current_time}")
            
            # Update status bar with success
            self.statusBar().showMessage(f"Processing frame {self.frame_count} - Connected to {self.pv_address}", 2000)

            self.canvas.draw_idle()
            logger.debug("Plot updated with new azimuthal integration data.")

        except ValueError as ve:
            error_msg = f"pyFAI integration failed (ValueError): {ve}. Check PONI file calibration and image format."
            logger.error(error_msg)
            self.handle_invalid_image(error_msg)
        except Exception as e:
            error_msg = f"pyFAI integration error: {e}. Verify PONI file is correct and images are valid diffraction patterns."
            logger.error(error_msg)
            self.handle_invalid_image(error_msg)
        finally:
            self._processing_image = False

    def handle_invalid_image(self, error_message=None):
        """
        Handles invalid images by logging warnings and showing user-friendly error messages.
        """
        self.error_count += 1
        if error_message:
            self.last_error_message = error_message
            logger.warning(f"Error processing image: {error_message}")
        else:
            logger.warning("Received invalid image data. Skipping this frame.")
            self.last_error_message = "Invalid image data received. Check if PV contains diffraction images."
        
        # Show error in status bar
        self.statusBar().showMessage(f"ERROR: {self.last_error_message}", 5000)
        
        # Update metadata panel with error information
        self._show_error_in_metadata(self.last_error_message)
    
    def _show_error_in_metadata(self, error_message):
        """Display error message in metadata panel with helpful suggestions."""
        error_text = f"""
⚠️ ERROR: {error_message}

Troubleshooting:
• Check if the PV address '{self.pv_address}' contains diffraction images
• Verify the PONI calibration file is loaded correctly
• Ensure images are 2D arrays (not single values or 1D)
• Check that the image dimensions match the expected detector size

Current Status:
• PV Address: {self.pv_address}
• PONI File: {self.poni_file if self.poni_file else 'Not loaded'}
• Calibration: {'Loaded' if self.ai is not None else 'NOT LOADED - Please load a PONI file'}
• Frames Processed: {self.frame_count}
• Consecutive Errors: {self.error_count}
"""
        # Append to existing metadata or replace if it's just an error
        current_text = self.metadata_panel.toPlainText()
        if "⚠️ ERROR" in current_text:
            # Replace existing error
            lines = current_text.split('\n')
            # Find where the error section starts and replace it
            new_lines = []
            skip_until_troubleshooting = False
            for i, line in enumerate(lines):
                if line.startswith("⚠️ ERROR"):
                    skip_until_troubleshooting = True
                    new_lines.append(error_text.strip())
                    break
                elif not skip_until_troubleshooting:
                    new_lines.append(line)
            self.metadata_panel.setPlainText('\n'.join(new_lines))
        else:
            # Prepend error to existing content
            self.metadata_panel.setPlainText(error_text + "\n\n" + current_text)

    def load_config(self):
        """
        Loads application configuration from a JSON file.
        Currently stores information such as the last used PONI file.
        Creates the file if it doesn't exist.
        """
        # Store config file in the same directory as this script
        config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(config_dir, "config.json")
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
                    logger.debug(f"Configuration loaded: {self.config}")
            else:
                # Initialize with empty config if file doesn't exist
                self.config = {}
                # Create the file immediately with empty config
                self.save_config()
        except Exception as e:
            logger.debug(f"No configuration found or could not load config: {e}")
            self.config = {}
            # Try to create empty config file
            try:
                self.save_config()
            except:
                pass

    def save_config(self):
        """
        Saves the current application configuration to a JSON file.
        Creates the file if it doesn't exist.
        """
        try:
            # Ensure directory exists
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
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
                cmap='magma',
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
    import argparse
    import traceback
    
    # Parse command-line arguments first, before any GUI code
    try:
        parser = argparse.ArgumentParser(description='pyFAI Analysis Window')
        parser.add_argument('--pv-address', type=str, default='pvapy:image',
                            help='PV address for image channel (default: pvapy:image)')
        args, unknown = parser.parse_known_args()
        pv_address = args.pv_address
    except Exception as e:
        # If argument parsing fails, use default
        pv_address = 'pvapy:image'
        unknown = []
        print(f"Warning: Failed to parse arguments: {e}, using default PV address")
    
    # Remove parsed arguments from sys.argv so QApplication doesn't see them
    sys.argv = [sys.argv[0]] + unknown
    
    # Create QApplication first - this is critical for GUI to work
    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print(f"CRITICAL: Failed to create QApplication: {e}")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Now try to create the window
    window = None
    try:
        window = PyFAIAnalysisWindow(pv_address=pv_address)
        # Make sure window is visible
        window.setVisible(True)
        window.show()
        window.raise_()  # Bring window to front
        window.activateWindow()  # Activate the window
        
        # Force window to appear by processing events
        QApplication.processEvents()
        
        # Always show window even if there were initialization errors
        if hasattr(window, 'last_error_message') and window.last_error_message:
            # Use QTimer to show message box after window is fully shown
            from PyQt5.QtCore import QTimer
            def show_warning():
                QMessageBox.warning(window, "Initialization Warning", 
                                  f"Window opened but encountered issues:\n\n{window.last_error_message}\n\n"
                                  "Please check:\n"
                                  "• PV address is correct\n"
                                  "• PONI file is loaded\n"
                                  "• PV contains diffraction image data")
            QTimer.singleShot(500, show_warning)  # Show after 500ms
    except Exception as e:
        # Create a minimal window to show error even if full initialization failed
        error_msg = f"Failed to initialize pyFAI Analysis Window:\n\n{str(e)}\n\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        logger.critical(f"Critical error during window initialization: {e}", exc_info=True)
        
        try:
            error_window = QMainWindow()
            error_window.setWindowTitle("pyFAI Analysis - Error")
            error_window.resize(600, 400)
            error_text = QTextEdit()
            error_text.setReadOnly(True)
            error_text.setPlainText(
                f"{error_msg}\n\n"
                f"PV Address: {pv_address}\n\n"
                "Please check:\n"
                "• PV address is correct and accessible\n"
                "• PONI calibration file can be loaded\n"
                "• Required dependencies are installed\n"
                "• Check console/terminal for detailed error messages"
            )
            error_window.setCentralWidget(error_text)
            error_window.show()
            error_window.raise_()
            error_window.activateWindow()
            window = error_window
        except Exception as e2:
            print(f"CRITICAL: Even error window failed to create: {e2}")
            print(traceback.format_exc())
            # Last resort - try to show a message box
            try:
                QMessageBox.critical(None, "Critical Error", 
                                    f"Failed to create pyFAI Analysis Window:\n\n{str(e)}\n\n"
                                    f"PV Address: {pv_address}\n\n"
                                    "Check console for details.")
            except:
                pass
            sys.exit(1)
    
    # Ensure window is visible
    if window:
        window.show()
        window.raise_()
        window.activateWindow()
        QApplication.processEvents()  # Process events to ensure window appears
    
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error during application execution: {e}")
        print(traceback.format_exc())
        sys.exit(1)