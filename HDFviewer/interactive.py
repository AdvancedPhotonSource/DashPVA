import sys
import os
import getpass

# Set HDF5 plugin path dynamically based on current environment
try:
    import hdf5plugin
    plugin_path = os.path.join(os.path.dirname(hdf5plugin.__file__), 'plugins')
    if os.path.exists(plugin_path):
        os.environ["HDF5_PLUGIN_PATH"] = plugin_path
    else:
        # Fallback to DashPVA venv if hdf5plugin is installed there
        dashpva_plugin_path = os.path.expanduser("~/DashPVA/.venv/lib/python3.11/site-packages/hdf5plugin/plugins")
        if os.path.exists(dashpva_plugin_path):
            os.environ["HDF5_PLUGIN_PATH"] = dashpva_plugin_path
except ImportError:
    # Fallback to DashPVA venv path
    dashpva_plugin_path = os.path.expanduser("~/DashPVA/.venv/lib/python3.11/site-packages/hdf5plugin/plugins")
    if os.path.exists(dashpva_plugin_path):
        os.environ["HDF5_PLUGIN_PATH"] = dashpva_plugin_path

import h5py
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QTreeWidget, 
                             QTreeWidgetItem, QSplitter, QLabel, QComboBox, QCheckBox, QTextEdit, QStackedWidget,
                             QSpinBox, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import json
import traceback
from pyqtgraph import RectROI
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
try:
    import crosscor  # Optional - for cross-correlation analysis
    CROSSCOR_AVAILABLE = True
except ImportError:
    CROSSCOR_AVAILABLE = False
    print("Warning: crosscor module not available. Cross-correlation features will be disabled.")
import glob
import re
from scipy.interpolate import griddata

# Configure matplotlib
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 'small',
    'axes.titlesize': 'medium'
})

# Disable OpenGL for pyqtgraph to avoid GL-related errors
pg.setConfigOption('useOpenGL', False)
pg.setConfigOptions(imageAxisOrder='row-major')

# Speckle_analyzer class, refactored to work with already loaded data
class Speckle_analyzer:
    def __init__(self, data, ROI=(830,920,70,50), frame_index=0, compare_frame_index=0):
        """
        Initialize the Speckle_analyzer with already loaded data
        
        Args:
            data: The already loaded and processed numpy array
            ROI: Region of Interest (x, y, width, height)
            frame_index: The frame index to use as reference
            compare_frame_index: The frame index to compare with the reference
                                If same as frame_index, performs self-correlation.
                                If different, performs cross-correlation between frames.
        """
        self.dataset = data
        self.frame_index = frame_index
        self.compare_frame_index = compare_frame_index
        profile_time = time.time()
        self.dector_ROI = ROI
        self.speckle_profile = self.cross_corr(self.frame_index, self.compare_frame_index, ROI=self.dector_ROI)     
        profile_time -= time.time()
        print(
            "Cross-correlation between frames:", self.frame_index, "and", self.compare_frame_index, "\n"
            "Contrast (Beta)                           :", self.speckle_profile[0]-1, "\n"
            "X-coordinate peak shift                   :", self.speckle_profile[1], "[pixels] \n"
            "Y-coordinate peak shift                   :", self.speckle_profile[2], "[pixels] \n"
            "FWHM of speckle along the x-axis          :", self.speckle_profile[3], "[pixels] \n"
            "FWHM of speckle along the y-axis          :", self.speckle_profile[4], "[pixels] \n"
            "Speckle profiling time                    :", -1*profile_time, "[s]"
        )
        
    def cross_corr(self, ref_frame_idx, compare_frame_idx, ROI):
        self.ref_frame = self.dataset[ref_frame_idx]
        center_col, center_row, col_roi_width, row_roi_width = ROI
        col_slice = slice(max(0, center_col - col_roi_width // 2), min(self.ref_frame.shape[1], center_col + col_roi_width // 2))
        row_slice = slice(max(0, center_row - row_roi_width // 2), min(self.ref_frame.shape[0], center_row + row_roi_width // 2))
        
        ref_frame = self.ref_frame[row_slice, col_slice]
        compare_frame = self.dataset[compare_frame_idx][row_slice, col_slice]
        
        if not CROSSCOR_AVAILABLE:
            raise ImportError("crosscor module is required for cross-correlation analysis")
        cross_corr = crosscor.crosscor(ref_frame.shape, mask=None, normalization="symavg")
        ccr = cross_corr(ref_frame, compare_frame)
        
        p = np.unravel_index(np.argmax(ccr, axis=None), ccr.shape)
        ax = (ccr[p[0] - 1, p[1]] + ccr[p[0] + 1, p[1]] - 2 * ccr[p[0], p[1]]) / 2.
        dx = (ccr[p[0] - 1, p[1]] - ccr[p[0] + 1, p[1]]) / 4. / ax
        ay = (ccr[p[0], p[1] - 1] + ccr[p[0], p[1] + 1] - 2 * ccr[p[0], p[1]]) / 2.
        dy = (ccr[p[0], p[1] - 1] - ccr[p[0], p[1] + 1]) / 4. / ay
        cy = ccr[p[0], p[1]] - ay * dy * dy
        cx = ccr[p[0], p[1]] - ax * dx * dx

        center_col, center_row, col_roi_width, row_roi_width = p[1], p[0], int(cy * 10), int(cx * 10)
        col_slice = slice(max(0, center_col - col_roi_width // 2), min(ref_frame.shape[1], center_col + col_roi_width // 2))
        row_slice = slice(max(0, center_row - row_roi_width // 2), min(ref_frame.shape[0], center_row + row_roi_width // 2))
        
        result = [(cx+cy)/2, dx, dy, np.sqrt(-cx / 2 / ax) * 2.3548, np.sqrt(-cy / 2 / ay) * 2.3548, ccr[row_slice, col_slice], ref_frame]
        return result
    
    def plot_report(self):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(4, 6, width_ratios=[1.0, 1.0, 1.2, 1.5, 1.2, 1.5])

        # Display reference frame with ROI
        ax1 = fig.add_subplot(gs[:2, 0])
        ax1.imshow(self.dataset[self.frame_index], cmap='viridis', aspect='auto', norm=LogNorm(vmin=0.9, vmax=100))
        ax1.set_title(f"Reference Frame: {self.frame_index}", fontsize=12)
        ax1.set_xlabel("X [pixels]", fontsize=10)
        ax1.set_ylabel("Y [pixels]", fontsize=10)
        center_col, center_row, col_roi_width, row_roi_width = self.dector_ROI
        roi_rect = Rectangle((center_col - col_roi_width/2, center_row - row_roi_width/2), 
                            col_roi_width, row_roi_width, linestyle='-', edgecolor='red', fill=False)
        ax1.add_patch(roi_rect)
        ax1.text(center_col, center_row + row_roi_width * 2, "ROI", fontsize=10, color='black', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.4, facecolor="white", edgecolor="black", linewidth=2))

        # Display comparison frame with ROI
        ax2 = fig.add_subplot(gs[:2, 1])
        ax2.imshow(self.dataset[self.compare_frame_index], cmap='viridis', aspect='auto', norm=LogNorm(vmin=0.9, vmax=100))
        ax2.set_title(f"Compare Frame: {self.compare_frame_index}", fontsize=12)
        ax2.set_xlabel("X [pixels]", fontsize=10)
        ax2.set_ylabel("Y [pixels]", fontsize=10)
        ax2.add_patch(Rectangle((center_col - col_roi_width/2, center_row - row_roi_width/2), 
                            col_roi_width, row_roi_width, linestyle='-', edgecolor='red', fill=False))

        # Correlation image
        ax3 = fig.add_subplot(gs[:2, 2:4])
        toshow = self.speckle_profile[5]
        im = ax3.imshow(toshow, cmap='viridis', aspect='auto', norm=LogNorm())
        ax3.set_title("Cross-Correlation", fontsize=12)
        ax3.set_xlabel("X [pixels]", fontsize=10)
        ax3.set_ylabel("Y [pixels]", fontsize=10)
        y_value = toshow.shape[0] // 2
        ax3.axhline(y=y_value, color='red', alpha=0.3, linestyle='--', label=f'Horizontal Cut')
        x_value = toshow.shape[1] // 2
        ax3.axvline(x=x_value, color='blue', alpha=0.3, linestyle='--', label=f'Vertical Cut')
        cbar = plt.colorbar(im, ax=ax3, format='%.1f')
        cbar.set_label('Correlation', fontsize=10)
        ax3.legend(fontsize=9)

        # Horizontal Line Cut of correlation with annotations (restored from original)
        ax4 = fig.add_subplot(gs[:2, 4:])
        ax4.plot(toshow[toshow.shape[0] // 2, :])
        ax4.set_title("Horizontal Line Cut", fontsize=12)
        ax4.set_xlabel("X [pixels]", fontsize=10)
        ax4.set_ylabel("Contrast + 1 baseline", fontsize=10)
        ax4.annotate(f"Speckle contrast={self.speckle_profile[0]-1:.4f}\n"
                    f"X-coordinate shift={self.speckle_profile[1]:.4f} pixels\n"
                    f"FWHM X-axis={self.speckle_profile[3]:.4f} pixels",
                    xy=(0.05, 0.5), xycoords='axes fraction', ha='left', va='center', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.3, facecolor="white", edgecolor="black", linewidth=2))

        # Reference and Compare ROIs
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.imshow(self.speckle_profile[6], cmap='viridis', aspect='auto', norm=LogNorm())
        ax5.set_title("Reference ROI", fontsize=12)
        ax5.set_xlabel("X [pixels]", fontsize=10)
        ax5.set_ylabel("Y [pixels]", fontsize=10)

        # Comparison ROI close-up (extract from dataset)
        comparison_roi = self.dataset[self.compare_frame_index][
            max(0, center_row - row_roi_width // 2):min(self.dataset.shape[1], center_row + row_roi_width // 2),
            max(0, center_col - col_roi_width // 2):min(self.dataset.shape[2], center_col + col_roi_width // 2)
        ]
        ax6 = fig.add_subplot(gs[2, 1])
        im = ax6.imshow(comparison_roi, cmap='viridis', aspect='auto', norm=LogNorm())
        ax6.set_title("Compare ROI", fontsize=12)
        ax6.set_xlabel("X [pixels]", fontsize=10)
        ax6.set_ylabel("Y [pixels]", fontsize=10)

        # Vertical Line Cut of correlation with annotations (restored from original)
        ax7 = fig.add_subplot(gs[2:, 2:4])
        ax7.plot(toshow[:, toshow.shape[1] // 2])
        ax7.set_title("Vertical Line Cut", fontsize=12)
        ax7.set_xlabel("Y [pixels]", fontsize=10)
        ax7.set_ylabel("Contrast + 1 baseline", fontsize=10)
        ax7.annotate(f"Speckle contrast={self.speckle_profile[0]-1:.4f}\n"
                    f"Y-coordinate shift={self.speckle_profile[2]:.4f} pixels\n"
                    f"FWHM Y-axis={self.speckle_profile[4]:.4f} pixels",
                    xy=(0.05, 0.5), xycoords='axes fraction', ha='left', va='center', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.3, facecolor="white", edgecolor="black", linewidth=2))

        # Frame difference
        ax8 = fig.add_subplot(gs[3, 0:2])
        if self.frame_index != self.compare_frame_index:
            diff = self.speckle_profile[6] - comparison_roi
            vmax = np.max(np.abs(diff))
            im = ax8.imshow(diff, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
            ax8.set_title("ROI Difference", fontsize=12)
            ax8.set_xlabel("X [pixels]", fontsize=10)
            ax8.set_ylabel("Y [pixels]", fontsize=10)
            cbar = plt.colorbar(im, ax=ax8)
            cbar.set_label('Diff. Intensity', fontsize=10)
        else:
            ax8.text(0.5, 0.5, "Self-correlation\n(frames identical)", 
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", alpha=0.3, facecolor="white", edgecolor="black", linewidth=2))
            ax8.axis('off')

        # Metrics summary panel
        ax9 = fig.add_subplot(gs[2:, 4:])
        ax9.axis('off')
        correlation_text = (
            f"Cross-Correlation Metrics\n\n"
            f"Frames {self.frame_index} ↔ {self.compare_frame_index}\n\n"
            f"Contrast (Beta): {self.speckle_profile[0]-1:.4f}\n\n"
            f"X-coordinate shift: {self.speckle_profile[1]:.4f} pixels\n\n"
            f"Y-coordinate shift: {self.speckle_profile[2]:.4f} pixels\n\n"
            f"FWHM X-axis: {self.speckle_profile[3]:.4f} pixels\n\n"
            f"FWHM Y-axis: {self.speckle_profile[4]:.4f} pixels"
        )
        ax9.text(0.5, 0.5, correlation_text, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=1.0", alpha=0.2, facecolor="white", edgecolor="black", linewidth=2))

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.savefig('speckle_profile.pdf', dpi=400, bbox_inches='tight')
        plt.show()

# Main GUI class
class HDF5ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 Viewer with Speckle Analysis")
        self.resize(1200, 800)

        # Use user's home directory for config file
        config_dir = os.path.expanduser("~/.dashpva")
        os.makedirs(config_dir, exist_ok=True)
        self.config_file = os.path.join(config_dir, 'hdfviewer_config.json')
        self.load_config()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # File selection and loading controls - compact single line
        file_controls = QGroupBox("File Controls")
        file_layout = QHBoxLayout(file_controls)
        file_layout.setSpacing(5)
        
        self.select_file_button = QPushButton("Select HDF5 File")
        self.select_file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.select_file_button)
        
        self.status_label = QLabel("No file selected")
        file_layout.addWidget(self.status_label)
        
        file_layout.addWidget(QLabel("Frames:"))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMinimum(0)
        self.start_frame_spin.setMaximum(9999)
        self.start_frame_spin.setMaximumWidth(110)
        file_layout.addWidget(self.start_frame_spin)
        
        file_layout.addWidget(QLabel("to"))
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMinimum(0)
        self.end_frame_spin.setMaximum(9999)
        self.end_frame_spin.setValue(100)
        self.end_frame_spin.setMaximumWidth(110)
        file_layout.addWidget(self.end_frame_spin)
        
        file_layout.addWidget(QLabel("Downsample:"))
        self.bin_combo = QComboBox()
        self.bin_combo.addItems(['1 (None)', '2', '4', '8', '16', '32', '64'])
        self.bin_combo.setMaximumWidth(90)
        file_layout.addWidget(self.bin_combo)
        
        self.enable_crop_checkbox = QCheckBox("Crop:")
        self.enable_crop_checkbox.setChecked(False)
        file_layout.addWidget(self.enable_crop_checkbox)
        
        file_layout.addWidget(QLabel("X:"))
        self.x_start_spin = QSpinBox()
        self.x_start_spin.setMinimum(0)
        self.x_start_spin.setMaximum(9999)
        self.x_start_spin.setValue(0)
        self.x_start_spin.setMaximumWidth(60)
        file_layout.addWidget(self.x_start_spin)
        
        file_layout.addWidget(QLabel("-"))
        self.x_end_spin = QSpinBox()
        self.x_end_spin.setMinimum(0)
        self.x_end_spin.setMaximum(9999)
        self.x_end_spin.setValue(9999)
        self.x_end_spin.setMaximumWidth(60)
        file_layout.addWidget(self.x_end_spin)
        
        file_layout.addWidget(QLabel("Y:"))
        self.y_start_spin = QSpinBox()
        self.y_start_spin.setMinimum(0)
        self.y_start_spin.setMaximum(9999)
        self.y_start_spin.setValue(0)
        self.y_start_spin.setMaximumWidth(60)
        file_layout.addWidget(self.y_start_spin)
        
        file_layout.addWidget(QLabel("-"))
        self.y_end_spin = QSpinBox()
        self.y_end_spin.setMinimum(0)
        self.y_end_spin.setMaximum(9999)
        self.y_end_spin.setValue(9999)
        self.y_end_spin.setMaximumWidth(60)
        file_layout.addWidget(self.y_end_spin)
        
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        self.load_button.setEnabled(False)
        file_layout.addWidget(self.load_button)
        
        file_layout.addStretch(1)
        
        main_layout.addWidget(file_controls)

        # Main splitter: Left (HDF tree) vs Right (ImageView + Metadata)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        # Left panel: HDF tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["HDF5 Structure"])
        self.tree_widget.itemClicked.connect(self.on_item_clicked)
        self.tree_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        left_layout.addWidget(self.tree_widget)
        splitter.addWidget(left_panel)

        # Right panel: ImageView and Metadata
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Display controls - first line
        display_layout = QHBoxLayout()
        display_layout.setSpacing(5)
        self.colormap_label = QLabel("Colormap:")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['magma', 'viridis', 'plasma', 'inferno', 'gray', 'hot'])
        self.colormap_combo.setCurrentText('magma')
        self.colormap_combo.setMaximumWidth(100)
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        self.log_scale_checkbox = QCheckBox("Log Scale")
        self.log_scale_checkbox.stateChanged.connect(self.update_display)
        self.auto_linear_scale_checkbox = QCheckBox("Auto Linear Scale")
        self.auto_linear_scale_checkbox.stateChanged.connect(self.apply_autoscale)
        
        self.vmin_label = QLabel("Vmin:")
        self.vmin_spin = QSpinBox()
        self.vmin_spin.setRange(0, 999999999)
        self.vmin_spin.setValue(0)
        self.vmin_spin.setMaximumWidth(80)
        self.vmin_spin.valueChanged.connect(self.update_image_levels)
        
        self.vmax_label = QLabel("Vmax:")
        self.vmax_spin = QSpinBox()
        self.vmax_spin.setRange(10, 999999999)
        self.vmax_spin.setValue(100)
        self.vmax_spin.setMaximumWidth(80)
        self.vmax_spin.valueChanged.connect(self.update_image_levels)
        
        self.chk_threshold = QCheckBox("Threshold")
        self.chk_threshold.stateChanged.connect(self.threshold_checked)
        self.chk_threshold_auto = QCheckBox("Auto")
        self.chk_threshold_auto.setChecked(True)
        self.chk_threshold_auto.stateChanged.connect(self.threshold_auto_checked)
        
        self.threshold_min_label = QLabel("Min:")
        self.threshold_min_spin = QSpinBox()
        self.threshold_min_spin.setRange(0, 2147483647)  # int32 max (QSpinBox limit - cannot exceed this)
        self.threshold_min_spin.setValue(0)
        self.threshold_min_spin.setMaximumWidth(70)
        self.threshold_min_spin.valueChanged.connect(self.threshold_values_changed)
        
        self.threshold_max_label = QLabel("Max:")
        self.threshold_max_spin = QSpinBox()
        self.threshold_max_spin.setRange(0, 2147483647)  # int32 max (QSpinBox limit - cannot exceed this)
        self.threshold_max_spin.setValue(0)
        self.threshold_max_spin.setMaximumWidth(70)
        self.threshold_max_spin.valueChanged.connect(self.threshold_values_changed)
        
        display_layout.addWidget(self.colormap_label)
        display_layout.addWidget(self.colormap_combo)
        display_layout.addWidget(self.log_scale_checkbox)
        display_layout.addWidget(self.auto_linear_scale_checkbox)
        display_layout.addWidget(self.vmin_label)
        display_layout.addWidget(self.vmin_spin)
        display_layout.addWidget(self.vmax_label)
        display_layout.addWidget(self.vmax_spin)
        display_layout.addWidget(self.chk_threshold)
        display_layout.addWidget(self.chk_threshold_auto)
        display_layout.addWidget(self.threshold_min_label)
        display_layout.addWidget(self.threshold_min_spin)
        display_layout.addWidget(self.threshold_max_label)
        display_layout.addWidget(self.threshold_max_spin)
        display_layout.addStretch(1)
        
        # Initially hide threshold controls until data is loaded
        self.chk_threshold_auto.hide()
        self.threshold_min_label.hide()
        self.threshold_min_spin.hide()
        self.threshold_max_label.hide()
        self.threshold_max_spin.hide()
        right_layout.addLayout(display_layout)
        
        # Analysis controls - second line
        analysis_layout = QHBoxLayout()
        analysis_layout.setSpacing(5)
        self.draw_roi_button = QPushButton("Draw ROI")
        self.draw_roi_button.clicked.connect(self.draw_roi)
        
        self.ref_frame_label = QLabel("Ref Frame:")
        self.ref_frame_spin = QSpinBox()
        self.ref_frame_spin.setMinimum(0)
        self.ref_frame_spin.setMaximum(9999)
        self.ref_frame_spin.setReadOnly(True)
        self.ref_frame_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.ref_frame_spin.setMaximumWidth(60)
        
        self.other_frame_label = QLabel("Other Frame:")
        self.other_frame_spin = QSpinBox()
        self.other_frame_spin.setMinimum(0)
        self.other_frame_spin.setMaximum(9999)
        self.other_frame_spin.setMaximumWidth(60)
        
        self.analyze_speckle_button = QPushButton("Analyze Speckle")
        self.analyze_speckle_button.clicked.connect(self.analyze_speckle)
        
        self.plot_motor_positions_button = QPushButton("Plot Motor Positions")
        self.plot_motor_positions_button.clicked.connect(self.plot_motor_positions)
        
        analysis_layout.addWidget(self.draw_roi_button)
        analysis_layout.addWidget(self.ref_frame_label)
        analysis_layout.addWidget(self.ref_frame_spin)
        analysis_layout.addWidget(self.other_frame_label)
        analysis_layout.addWidget(self.other_frame_spin)
        analysis_layout.addWidget(self.analyze_speckle_button)
        analysis_layout.addWidget(self.plot_motor_positions_button)
        analysis_layout.addStretch(1)
        
        right_layout.addLayout(analysis_layout)

        # Nested splitter: ImageView vs Metadata
        nested_splitter = QSplitter(Qt.Vertical)
        
        self.stacked_widget = QStackedWidget()
        self.blank_widget = QWidget()
        
        # Create a PlotItem and pass it to ImageView
        plot = pg.PlotItem()
        self.image_view = pg.ImageView(view=plot)
        # Set axis labels
        self.image_view.view.getAxis('left').setLabel(text='Row [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='Columns [pixels]')
        
        # Connect currentIndexChanged signal to update the ref and other frame spinboxes
        self.image_view.sigTimeChanged.connect(self.update_frame_indices)
        
        self.plot_widget = pg.PlotWidget()
        self.stacked_widget.addWidget(self.blank_widget)
        self.stacked_widget.addWidget(self.image_view)
        self.stacked_widget.addWidget(self.plot_widget)
        nested_splitter.addWidget(self.stacked_widget)

        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        nested_splitter.addWidget(self.metadata_text)

        # Set initial sizes: Make metadata smaller (120px) and ImageView larger (800px)
        nested_splitter.setSizes([800, 120])

        right_layout.addWidget(nested_splitter)

        splitter.addWidget(right_panel)
        # Adjust main splitter sizes: Left panel larger (400px), Right panel larger (1100px)
        splitter.setSizes([400, 1100])

        self.current_h5_obj = None
        self.h5_file = None
        self.selected_file_path = None
        self.current_roi = None
        self.selected_dataset_path = None
        self.binned_data = None
        
        # For tracking dataset dimensions
        self.original_dimensions = None
        self.current_dimensions = None
        self.cropping_applied = False
        
        # Motor position data
        self.motor_positions = None
        self.motor_names = []
        self.experiment_name = None

    def load_config(self):
        self.config = {}
        # Get current username with more robust fallback
        try:
            self.username = getpass.getuser()
            print(f"Current user identified as: {self.username}")
        except Exception as e:
            print(f"Error detecting username: {e}")
            self.username = os.environ.get('USER', 'unknown_user')
            print(f"Using fallback username: {self.username}")
        
        # Create config file if it doesn't exist
        if not os.path.exists(self.config_file):
            print(f"Config file not found, creating new one: {self.config_file}")
            try:
                with open(self.config_file, 'w') as f:
                    json.dump({self.username: {'last_dir': f'/gdata/dm/9IDD/'}}, f, indent=2)
            except Exception as e:
                print(f"Error creating config file: {e}")
        
        # Load config
        try:
            with open(self.config_file, 'r') as f:
                all_configs = json.load(f)
                print(f"Loaded config file with users: {list(all_configs.keys())}")
                
            # Get user-specific config if it exists, or create empty one
            if self.username in all_configs:
                self.config = all_configs[self.username]
                print(f"Found configuration for {self.username}: {self.config}")
            else:
                self.config = {'last_dir': f'/gdata/dm/9IDD/'}
                print(f"No configuration found for {self.username}, using defaults")
                
        except Exception as e:
            print(f"Error loading config file: {e}")
            self.config = {'last_dir': f'/gdata/dm/9IDD/'}

    def save_config(self):
        print(f"Attempting to save config for user: {self.username}")
        try:
            # Load existing configs for all users or create new dictionary
            all_configs = {}
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, 'r') as f:
                        all_configs = json.load(f)
                    print(f"Loaded existing config with users: {list(all_configs.keys())}")
                except Exception as e:
                    print(f"Error reading existing config: {e}")
                    all_configs = {}
            
            # Update the config for the current user
            all_configs[self.username] = self.config
            
            # Save all user configs with explicit file permissions check
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(all_configs, f, indent=2)
                print(f"Successfully saved configuration for user: {self.username}")
                
                # Ensure file has correct permissions
                try:
                    os.chmod(self.config_file, 0o666)  # Make writable by all users
                    print(f"Set permissions on {self.config_file} to 666")
                except Exception as perm_error:
                    print(f"Warning: Could not set file permissions: {perm_error}")
                    
            except Exception as write_error:
                print(f"Error writing to config file: {write_error}")
                
        except Exception as e:
            print(f"Error in save_config: {e}")
            traceback.print_exc()

    def select_file(self):
        last_dir = self.config.get('last_dir', '')
        file_path, _ = QFileDialog.getOpenFileName(self, "Select HDF5 File", last_dir, "HDF5 Files (*.h5 *.hdf *.nxs *.nx *.cxi);;All Files (*)")
        if not file_path:
            return

        self.config['last_dir'] = os.path.dirname(file_path)
        self.save_config()
        
        self.selected_file_path = file_path
        self.status_label.setText(f"Selected: {file_path} (click Load Data to load)")
        self.load_button.setEnabled(True)
        
        # Clear previous data
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
        self.tree_widget.clear()
        self.current_h5_obj = None
        self.selected_dataset_path = None
        self.stacked_widget.setCurrentWidget(self.blank_widget)
        self.metadata_text.clear()

    def load_data(self):
        if not self.selected_file_path:
            self.status_label.setText("No file selected!")
            return
            
        try:
            if self.h5_file is not None:
                self.h5_file.close()
            
            self.h5_file = h5py.File(self.selected_file_path, 'r', libver='latest', swmr=True)
            self.status_label.setText(f"Loaded: {self.selected_file_path}")
            self.tree_widget.clear()
            self.build_h5_tree('/', self.h5_file)
            self.tree_widget.expandToDepth(0)
            self.current_h5_obj = None
            self.binned_data = None
        except Exception as e:
            self.status_label.setText(f"Error opening file: {str(e)}")
            print(f"Error opening file: {str(e)}")

    def build_h5_tree(self, path, h5_obj, parent=None):
        if parent is None:
            item = QTreeWidgetItem(self.tree_widget, [path])
        else:
            name = path.split('/')[-1] if path != '/' else '/'
            item = QTreeWidgetItem(parent, [name])
        item.setData(0, Qt.UserRole, path)

        if isinstance(h5_obj, h5py.Group):
            for key in h5_obj.keys():
                child_path = f"{path}/{key}" if path != '/' else f"/{key}"
                self.build_h5_tree(child_path, h5_obj[key], item)
        elif isinstance(h5_obj, h5py.Dataset):
            shape_str = str(h5_obj.shape)
            dtype_str = str(h5_obj.dtype)
            item.setText(0, item.text(0) + f" {shape_str} {dtype_str}")
            if len(h5_obj.shape) >= 2:
                item.setForeground(0, pg.mkColor('blue'))

    def on_item_clicked(self, item, column):
        path = item.data(0, Qt.UserRole)
        if path in self.h5_file:
            self.current_h5_obj = self.h5_file[path]
            if isinstance(self.current_h5_obj, h5py.Dataset):
                self.selected_dataset_path = path
                # Update frame range spinners based on dataset dimensions
                if len(self.current_h5_obj.shape) == 3:
                    max_frames = self.current_h5_obj.shape[0] - 1
                    self.start_frame_spin.setMaximum(max_frames)
                    self.end_frame_spin.setMaximum(max_frames)
                    if self.end_frame_spin.value() > max_frames:
                        self.end_frame_spin.setValue(max_frames)
                    if self.start_frame_spin.value() > self.end_frame_spin.value():
                        self.start_frame_spin.setValue(0)
                    
                    # Update crop spinners based on image dimensions
                    height = self.current_h5_obj.shape[1]
                    width = self.current_h5_obj.shape[2]
                    
                    # Store original dimensions
                    self.original_dimensions = (height, width)
                    
                    # Update X spinner maximums but don't reset values if already set
                    self.x_start_spin.setMaximum(width - 2)
                    self.x_end_spin.setMaximum(width - 1)
                    
                    # Only initialize default values if they're at their initial values
                    if self.x_end_spin.value() >= 9999:
                        self.x_end_spin.setValue(width - 1)
                    
                    # Update Y spinner maximums but don't reset values if already set
                    self.y_start_spin.setMaximum(height - 2)
                    self.y_end_spin.setMaximum(height - 1)
                    
                    # Only initialize default values if they're at their initial values
                    if self.y_end_spin.value() >= 9999:
                        self.y_end_spin.setValue(height - 1)
                
                elif len(self.current_h5_obj.shape) == 2:
                    # For 2D datasets
                    height = self.current_h5_obj.shape[0]
                    width = self.current_h5_obj.shape[1]
                    
                    # Store original dimensions
                    self.original_dimensions = (height, width)
                    
                    # Update X spinner maximums but don't reset values if already set
                    self.x_start_spin.setMaximum(width - 2)
                    self.x_end_spin.setMaximum(width - 1)
                    
                    # Only initialize default values if they're at their initial values
                    if self.x_end_spin.value() >= 9999:
                        self.x_end_spin.setValue(width - 1)
                    
                    # Update Y spinner maximums but don't reset values if already set
                    self.y_start_spin.setMaximum(height - 2)
                    self.y_end_spin.setMaximum(height - 1)
                    
                    # Only initialize default values if they're at their initial values
                    if self.y_end_spin.value() >= 9999:
                        self.y_end_spin.setValue(height - 1)
                
                # Update threshold values if Auto is checked
                if hasattr(self, 'chk_threshold_auto') and self.chk_threshold_auto.isChecked():
                    self.update_threshold_values_for_dataset(self.current_h5_obj)
                
                # Just show metadata, don't load the data yet
                self.metadata_text.setText(self.get_attributes_text(self.current_h5_obj))
            else:
                self.selected_dataset_path = None
                self.metadata_text.setText(self.get_attributes_text(self.current_h5_obj))
                self.stacked_widget.setCurrentWidget(self.blank_widget)
        else:
            self.current_h5_obj = None
            self.selected_dataset_path = None
            self.stacked_widget.setCurrentWidget(self.blank_widget)
            self.metadata_text.clear()

    def on_item_double_clicked(self, item, column):
        path = item.data(0, Qt.UserRole)
        if path in self.h5_file:
            self.current_h5_obj = self.h5_file[path]
            if isinstance(self.current_h5_obj, h5py.Dataset):
                self.selected_dataset_path = path
                # Load and display the data with current frame range and downsampling settings
                self.load_and_display_dataset()
            else:
                # For non-datasets, just show metadata
                self.metadata_text.setText(self.get_attributes_text(self.current_h5_obj))
                self.stacked_widget.setCurrentWidget(self.blank_widget)
        else:
            self.current_h5_obj = None
            self.selected_dataset_path = None
            self.stacked_widget.setCurrentWidget(self.blank_widget)
            self.metadata_text.clear()

    def load_and_display_dataset(self):
        if not self.selected_dataset_path or not isinstance(self.current_h5_obj, h5py.Dataset):
            return
            
        dataset = self.current_h5_obj
        try:
            if len(dataset.shape) == 1:
                self.stacked_widget.setCurrentWidget(self.plot_widget)
                data = dataset[()]
                self.plot_widget.clear()
                self.plot_widget.plot(data)
                self.plot_widget.setLogMode(False, self.log_scale_checkbox.isChecked())
            elif len(dataset.shape) in [2, 3]:
                self.stacked_widget.setCurrentWidget(self.image_view)
                
                # Validate parameters first
                self.validate_parameters(dataset)
                
                # Load only the selected frame range for 3D datasets
                if len(dataset.shape) == 3:
                    start_frame = self.start_frame_spin.value()
                    end_frame = self.end_frame_spin.value()
                    
                    # Ensure valid range
                    if start_frame > end_frame:
                        start_frame, end_frame = 0, min(end_frame, dataset.shape[0]-1)
                        self.start_frame_spin.setValue(start_frame)
                        self.end_frame_spin.setValue(end_frame)
                    
                    # Load only selected frames
                    self.status_label.setText(f"Loading frames {start_frame} to {end_frame}...")
                    QApplication.processEvents()  # Update UI
                    
                    if self.enable_crop_checkbox.isChecked():
                        # Apply cropping when loading
                        x_start = self.x_start_spin.value()
                        x_end = self.x_end_spin.value()
                        y_start = self.y_start_spin.value()
                        y_end = self.y_end_spin.value()
                        
                        image = dataset[start_frame:end_frame+1, y_start:y_end+1, x_start:x_end+1]
                        self.cropping_applied = True
                        self.current_dimensions = (y_end - y_start + 1, x_end - x_start + 1)
                    else:
                        image = dataset[start_frame:end_frame+1]
                        self.cropping_applied = False
                        self.current_dimensions = (dataset.shape[1], dataset.shape[2])
                else:
                    # For 2D datasets
                    if self.enable_crop_checkbox.isChecked():
                        # Apply cropping
                        x_start = self.x_start_spin.value()
                        x_end = self.x_end_spin.value()
                        y_start = self.y_start_spin.value()
                        y_end = self.y_end_spin.value()
                        
                        image = dataset[y_start:y_end+1, x_start:x_end+1]
                        image = np.expand_dims(image, axis=0)  # Add dummy dimension
                        self.cropping_applied = True
                        self.current_dimensions = (y_end - y_start + 1, x_end - x_start + 1)
                    else:
                        image = dataset[()]
                        # Add dummy dimension for 2D data to be treated as one frame
                        image = np.expand_dims(image, axis=0)
                        self.cropping_applied = False
                        self.current_dimensions = (dataset.shape[0], dataset.shape[1])
                
                # Update threshold values based on actual data statistics (if auto is enabled)
                # Do this BEFORE applying threshold so values are calculated from raw data
                if hasattr(self, 'chk_threshold_auto') and self.chk_threshold_auto.isChecked():
                    self.update_threshold_values_from_data(image)
                
                # Apply thresholding if enabled
                if self.chk_threshold.isChecked():
                    image = self.apply_threshold(image)
                
                # Apply downsampling/binning if selected
                bin_factor = int(self.bin_combo.currentText().split(' ')[0])
                
                # Validate bin factor against image dimensions
                min_dimension = min(image.shape[1], image.shape[2])
                if bin_factor >= min_dimension:
                    bin_factor = max(1, min_dimension // 2)  # Auto-adjust to max safe value
                    self.bin_combo.setCurrentText(f"{bin_factor}" if bin_factor > 1 else "1 (None)")
                    self.status_label.setText(f"Warning: Bin factor reduced to {bin_factor} to fit image dimensions")
                    QApplication.processEvents()  # Update UI
                
                if bin_factor > 1:
                    # Simple nearest-neighbor downsampling
                    if len(dataset.shape) == 3:
                        # For 3D dataset: keep all frames, downsample each frame
                        # Calculate exact size after downsampling with stride
                        h_binned = len(range(0, image.shape[1], bin_factor))
                        w_binned = len(range(0, image.shape[2], bin_factor))
                        new_shape = (image.shape[0], h_binned, w_binned)
                        
                        self.binned_data = np.zeros(new_shape, dtype=image.dtype)
                        for i in range(image.shape[0]):
                            self.binned_data[i] = image[i, ::bin_factor, ::bin_factor]
                    else:
                        # For 2D dataset
                        h_binned = len(range(0, image.shape[1], bin_factor))
                        w_binned = len(range(0, image.shape[2], bin_factor))
                        new_shape = (1, h_binned, w_binned)
                        
                        self.binned_data = np.zeros(new_shape, dtype=image.dtype)
                        self.binned_data[0] = image[0, ::bin_factor, ::bin_factor]
                    
                    image = self.binned_data
                    self.current_dimensions = (h_binned, w_binned)
                else:
                    self.binned_data = image
                
                # Get pixel dimensions for the title
                if len(dataset.shape) == 3:
                    original_frames = dataset.shape[0]
                    height, width = image.shape[1], image.shape[2]
                    
                    title = f"Image Dimensions: {width}x{height} pixels (Frames {start_frame}-{end_frame} of {original_frames})"
                    
                    if self.cropping_applied:
                        orig_height, orig_width = self.original_dimensions
                        crop_text = f" [Cropped from {orig_width}x{orig_height}]"
                        title += crop_text
                        
                    if bin_factor > 1:
                        title += f" [Downsampled {bin_factor}x]"
                else:
                    height, width = image.shape[1], image.shape[2]
                    title = f"Image Dimensions: {width}x{height} pixels"
                    
                    if self.cropping_applied:
                        orig_height, orig_width = self.original_dimensions
                        crop_text = f" [Cropped from {orig_width}x{orig_height}]"
                        title += crop_text
                        
                    if bin_factor > 1:
                        title += f" [Downsampled {bin_factor}x]"
                
                self.image_view.view.setTitle(title)

                # Compute min/max levels
                min_level, max_level = np.nanmin(image), np.nanmax(image)
                if self.log_scale_checkbox.isChecked():
                    # Clip negative values to 0 for log1p (log1p requires x > -1)
                    image = np.clip(image, 0, None)
                    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                    image = np.log1p(image)
                    min_level = np.log1p(max(0, min_level)) if not np.isnan(min_level) else 0
                    max_level = np.log1p(max(0, max_level)) if not np.isnan(max_level) else 0
                
                # Use vmin/vmax from spin boxes if they are within valid range
                vmin = self.vmin_spin.value()
                vmax = self.vmax_spin.value()
                if vmin < vmax:
                    if self.log_scale_checkbox.isChecked():
                        vmin = np.log1p(max(0, vmin))
                        vmax = np.log1p(max(0, vmax))
                    min_level = vmin
                    max_level = vmax

                # Display the image
                self.image_view.setImage(image, autoRange=False, autoLevels=False, 
                                        levels=(min_level, max_level), autoHistogramRange=False)
                
                # Apply colormap (magma is default)
                self.update_colormap(self.colormap_combo.currentText())
                
                # Apply autoscale if enabled
                if self.auto_linear_scale_checkbox.isChecked():
                    self.apply_autoscale()
                
                load_info = f"Loaded {image.shape[0]} frames"
                if self.cropping_applied:
                    load_info += f", cropped to {width}x{height}"
                if bin_factor > 1:
                    load_info += f", downsampled {bin_factor}x"
                self.status_label.setText(f"{load_info} from {self.selected_file_path}")
            else:
                raise ValueError("Dataset shape not supported for plotting.")
        except Exception as e:
            self.status_label.setText(f"Error plotting dataset: {str(e)}")
            self.stacked_widget.setCurrentWidget(self.blank_widget)
            traceback.print_exc()

    def update_image_levels(self):
        """Update the image display levels based on current vmin/vmax values"""
        if self.stacked_widget.currentWidget() == self.image_view and self.binned_data is not None:
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()
            
            if vmin < vmax:
                if self.log_scale_checkbox.isChecked():
                    vmin = np.log1p(max(0, vmin))
                    vmax = np.log1p(max(0, vmax))
                self.image_view.setLevels(min=vmin, max=vmax)
    
    def apply_autoscale(self):
        """
        Calculates and applies autoscale based on 5th and 95th percentiles of the intensity histogram.
        Sets min to 5th percentile and max to 95th percentile.
        """
        if self.auto_linear_scale_checkbox.isChecked():
            if self.stacked_widget.currentWidget() == self.image_view and self.binned_data is not None:
                # Get current frame
                current_idx = self.image_view.currentIndex
                if 0 <= current_idx < self.binned_data.shape[0]:
                    image = self.binned_data[current_idx]
                    
                    # Flatten the image to get all intensity values
                    intensities = image.flatten()
                    
                    # Remove any NaN or infinite values
                    intensities = intensities[np.isfinite(intensities)]
                    
                    if len(intensities) > 0:
                        # Calculate 5th and 95th percentiles
                        min_percentile = np.percentile(intensities, 5)
                        max_percentile = np.percentile(intensities, 95)
                        
                        # Ensure min < max
                        if min_percentile >= max_percentile:
                            max_percentile = min_percentile + 1
                        
                        # Store original values for setLevels (can be floats)
                        min_level = min_percentile
                        max_level = max_percentile
                        
                        # Clamp values to spinbox range to prevent overflow (QSpinBox uses int32, max is 2147483647)
                        spinbox_max = 2147483647
                        min_percentile_int = min(int(min_percentile), spinbox_max)
                        max_percentile_int = min(int(max_percentile), spinbox_max)
                        
                        # Update the spinbox values (this will trigger update_min_max_setting)
                        # Temporarily block signals to prevent feedback loop
                        self.vmin_spin.blockSignals(True)
                        self.vmax_spin.blockSignals(True)
                        self.vmin_spin.setValue(min_percentile_int)
                        self.vmax_spin.setValue(max_percentile_int)
                        self.vmin_spin.blockSignals(False)
                        self.vmax_spin.blockSignals(False)
                        
                        # Apply the levels directly using original float values (without log scale if auto linear is enabled)
                        if not self.log_scale_checkbox.isChecked():
                            self.image_view.setLevels(min_level, max_level)
                        else:
                            # If log scale is on, apply log1p to the percentiles
                            min_level = np.log1p(max(0, min_level))
                            max_level = np.log1p(max(0, max_level))
                            self.image_view.setLevels(min_level, max_level)
    
    def get_threshold_range_for_dtype(self, dtype) -> tuple:
        """
        Determines the threshold range based on the data type.
        
        Args:
            dtype: numpy dtype or h5py dtype
            
        Returns:
            tuple: (min_threshold, max_threshold) based on data type
        """
        # Convert h5py dtype to numpy dtype string if needed
        dtype_str = str(dtype)
        
        dtype_map = {
            'uint8': (0, 2**8 - 2),      # uint8: 0 to 254
            'uint16': (0, 2**16 - 2),    # uint16: 0 to 65534
            'uint32': (0, 2**32 - 2),    # uint32: 0 to 4294967294
            'uint64': (0, 2**32 - 2),    # uint64: use uint32 limit
            'int8': (-2**7, 2**7 - 1),   # int8: -128 to 127
            'int16': (-2**15, 2**15 - 1), # int16: -32768 to 32767
            'int32': (-2**31, 2**31 - 1), # int32
            'int64': (-2**31, 2**31 - 1), # int64: use int32 limit
            'float32': (0, 2**16 - 1),    # float32: 0 to 65535
            'float64': (0, 2**16 - 1),   # float64: 0 to 65535
        }
        
        # Try to match dtype string
        for key, value in dtype_map.items():
            if key in dtype_str:
                return value
        
        # Default fallback
        return (0, 2**16 - 1)
    
    def update_threshold_values_from_data(self, image_data: np.ndarray) -> None:
        """
        Updates the threshold min/max spinboxes based on dtype range.
        Uses dtype-based min and max values (not percentiles).
        """
        if not hasattr(self, 'chk_threshold_auto') or not self.chk_threshold_auto.isChecked():
            return
        
        # Get dtype-based range
        if self.current_h5_obj is not None and isinstance(self.current_h5_obj, h5py.Dataset):
            min_thresh, max_thresh = self.get_threshold_range_for_dtype(self.current_h5_obj.dtype)
        else:
            # Default fallback
            min_thresh = 0
            max_thresh = 2**16 - 1
        
        if min_thresh == 0 and max_thresh == 0:
            return
        
        # Convert to int and clamp to spinbox range (QSpinBox uses int32, max is 2147483647)
        spinbox_max = 2147483647
        min_thresh_int = int(min_thresh)
        max_thresh_int = int(max_thresh)
        
        # Clamp to spinbox maximum (QSpinBox limitation)
        if max_thresh_int > spinbox_max:
            max_thresh_int = spinbox_max
        if min_thresh_int > spinbox_max:
            min_thresh_int = spinbox_max
        
        # Ensure min < max
        if min_thresh_int >= max_thresh_int:
            min_thresh_int = max(0, max_thresh_int - 1)
        
        # Block signals to prevent unchecking Auto
        self.threshold_min_spin.blockSignals(True)
        self.threshold_max_spin.blockSignals(True)
        self.threshold_min_spin.setValue(min_thresh_int)
        self.threshold_max_spin.setValue(max_thresh_int)
        self.threshold_min_spin.blockSignals(False)
        self.threshold_max_spin.blockSignals(False)
    
    def update_threshold_values_for_dataset(self, dataset) -> None:
        """
        Updates the threshold min/max spinboxes based on dataset dtype.
        Only updates if Auto checkbox is checked.
        This is a fallback when data-based calculation is not available.
        """
        if hasattr(self, 'chk_threshold_auto') and self.chk_threshold_auto.isChecked():
            min_thresh, max_thresh = self.get_threshold_range_for_dtype(dataset.dtype)
            if min_thresh == 0 and max_thresh == 0:
                return
            
            # Clamp values to spinbox range to prevent overflow (QSpinBox uses int32, max is 2147483647)
            spinbox_max = 2147483647
            min_thresh = min(int(min_thresh), spinbox_max)
            max_thresh = min(int(max_thresh), spinbox_max)
            
            # Block signals to prevent unchecking Auto
            self.threshold_min_spin.blockSignals(True)
            self.threshold_max_spin.blockSignals(True)
            self.threshold_min_spin.setValue(min_thresh)
            self.threshold_max_spin.setValue(max_thresh)
            self.threshold_min_spin.blockSignals(False)
            self.threshold_max_spin.blockSignals(False)
    
    def threshold_auto_checked(self) -> None:
        """
        Handles Auto checkbox state changes for threshold.
        When checked, fills min/max spinboxes with auto-calculated values from actual data.
        Reloads the image to apply threshold changes.
        """
        if hasattr(self, 'chk_threshold_auto') and self.chk_threshold_auto.isChecked():
            # Use loaded data if available, otherwise fall back to dataset dtype
            if self.binned_data is not None:
                self.update_threshold_values_from_data(self.binned_data)
            elif self.current_h5_obj is not None and isinstance(self.current_h5_obj, h5py.Dataset):
                self.update_threshold_values_for_dataset(self.current_h5_obj)
            
            # Reload the image to apply threshold changes
            if self.binned_data is not None and self.selected_dataset_path:
                self.load_and_display_dataset()
    
    def threshold_values_changed(self) -> None:
        """
        Handles manual changes to threshold min/max spinboxes.
        Unchecks Auto checkbox when user manually edits values.
        Reloads the image to apply threshold changes immediately.
        """
        if hasattr(self, 'chk_threshold_auto'):
            # Check if this is a user-initiated change (not from auto update)
            if not self.threshold_min_spin.signalsBlocked() and not self.threshold_max_spin.signalsBlocked():
                # User manually edited, so uncheck Auto
                self.chk_threshold_auto.blockSignals(True)
                self.chk_threshold_auto.setChecked(False)
                self.chk_threshold_auto.blockSignals(False)
                
                # Reload the image to apply threshold changes immediately
                if self.binned_data is not None and self.selected_dataset_path:
                    self.load_and_display_dataset()
    
    def threshold_checked(self) -> None:
        """
        Handles threshold checkbox state changes.
        Updates the visibility of threshold controls and reloads image.
        """
        if hasattr(self, 'chk_threshold_auto'):
            if self.chk_threshold.isChecked():
                self.chk_threshold_auto.show()
                self.threshold_min_label.show()
                self.threshold_min_spin.show()
                self.threshold_max_label.show()
                self.threshold_max_spin.show()
                # Update values if Auto is checked - use loaded data if available
                if self.chk_threshold_auto.isChecked():
                    if self.binned_data is not None:
                        self.update_threshold_values_from_data(self.binned_data)
                    elif self.current_h5_obj is not None and isinstance(self.current_h5_obj, h5py.Dataset):
                        self.update_threshold_values_for_dataset(self.current_h5_obj)
            else:
                self.chk_threshold_auto.hide()
                self.threshold_min_label.hide()
                self.threshold_min_spin.hide()
                self.threshold_max_label.hide()
                self.threshold_max_spin.hide()
            
            # Reload the image to apply/remove threshold changes immediately
            if self.binned_data is not None and self.selected_dataset_path:
                self.load_and_display_dataset()
    
    def apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Applies vectorized thresholding to the image based on threshold spinbox values.
        If auto threshold is enabled, uses the actual dtype maximum instead of spinbox value.
        Values below min_thresh are set to min_thresh, values above max_thresh are set to 0.
        
        Args:
            image: Input image array
            
        Returns:
            Thresholded image array
        """
        if not hasattr(self, 'threshold_min_spin') or not hasattr(self, 'threshold_max_spin'):
            return image
        
        min_thresh = self.threshold_min_spin.value()
        max_thresh = self.threshold_max_spin.value()
        
        # If auto threshold is enabled, use the actual dtype maximum instead of clamped spinbox value
        if hasattr(self, 'chk_threshold_auto') and self.chk_threshold_auto.isChecked():
            if self.current_h5_obj is not None and isinstance(self.current_h5_obj, h5py.Dataset):
                _, dtype_max_thresh = self.get_threshold_range_for_dtype(self.current_h5_obj.dtype)
                # Use dtype maximum if spinbox is at its max (indicating it was clamped)
                if max_thresh >= 2147483647:  # Spinbox max, likely clamped
                    max_thresh = dtype_max_thresh
        
        if min_thresh == 0 and max_thresh == 0:
            return image
        
        # Vectorized thresholding: 
        # - Values below min_thresh are set to min_thresh
        # - Values above max_thresh are set to 0
        result = image.copy()
        result[result < min_thresh] = min_thresh
        result[result > max_thresh] = 0
        return result
    
    def update_display(self):
        if self.current_h5_obj is None:
            self.stacked_widget.setCurrentWidget(self.blank_widget)
            self.metadata_text.clear()
            return

        self.metadata_text.setText(self.get_attributes_text(self.current_h5_obj))

        if isinstance(self.current_h5_obj, h5py.Dataset):
            # Don't automatically load data - just show that a dataset was selected
            # Data will only be loaded and displayed on double-click
            pass
        else:
            self.stacked_widget.setCurrentWidget(self.blank_widget)
            
        # Update image levels when log scale changes
        self.update_image_levels()

    def get_attributes_text(self, h5_obj):
        text = f"Path: {self.selected_dataset_path}\n\n"
        
        if isinstance(h5_obj, h5py.Dataset):
            text += f"Shape: {h5_obj.shape}\n"
            text += f"Type: {h5_obj.dtype}\n\n"
            
        if h5_obj.attrs:
            text += "Attributes:\n"
            for key, value in h5_obj.attrs.items():
                text += f"{key}: {value}\n"
        else:
            text += "No attributes."
        
        if isinstance(h5_obj, h5py.Dataset) and len(h5_obj.shape) >= 2:
            text += f"\n\nDouble-click to load"
            
            if len(h5_obj.shape) == 3:
                text += f" frames {self.start_frame_spin.value()}-{self.end_frame_spin.value()}"
            
            if self.enable_crop_checkbox.isChecked():
                x_start = self.x_start_spin.value()
                x_end = self.x_end_spin.value()
                y_start = self.y_start_spin.value()
                y_end = self.y_end_spin.value()
                text += f"\nCropping: X({x_start}-{x_end}), Y({y_start}-{y_end})"
                crop_width = x_end - x_start + 1
                crop_height = y_end - y_start + 1
                text += f" [{crop_width}x{crop_height} pixels]"
            
            bin_factor = int(self.bin_combo.currentText().split(' ')[0])
            if bin_factor > 1:
                text += f"\nDownsampling: {bin_factor}x"
                
                # If cropping is enabled, calculate final dimensions
                if self.enable_crop_checkbox.isChecked():
                    x_start = self.x_start_spin.value()
                    x_end = self.x_end_spin.value()
                    y_start = self.y_start_spin.value()
                    y_end = self.y_end_spin.value()
                    
                    crop_width = x_end - x_start + 1
                    crop_height = y_end - y_start + 1
                    
                    final_width = len(range(0, crop_width, bin_factor))
                    final_height = len(range(0, crop_height, bin_factor))
                    
                    text += f" [Final size: {final_width}x{final_height} pixels]"
        
        return text

    def update_colormap(self, colormap_name):
        try:
            if hasattr(pg.colormap, 'get'):
                cmap = pg.colormap.get(colormap_name)
                self.image_view.setColorMap(cmap)
            else:
                # Fallback for older pyqtgraph versions
                if colormap_name == 'magma':
                    # Magma colormap colors
                    colors = [[0, 0, 4], [28, 16, 68], [79, 18, 123], [126, 37, 124], [160, 64, 104], [188, 97, 83], [214, 131, 64], [243, 164, 44], [254, 219, 127], [254, 254, 255]]
                    positions = [i / (len(colors) - 1) for i in range(len(colors))]
                    self.image_view.setColorMap(pg.ColorMap(positions, colors))
                elif colormap_name == 'gray':
                    self.image_view.setColorMap(pg.ColorMap([0, 1], [[0, 0, 0], [255, 255, 255]]))
                elif colormap_name == 'hot':
                    colors = [[0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255]]
                    self.image_view.setColorMap(pg.ColorMap([0, 0.33, 0.66, 1], colors))
                else:
                    colors = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]
                    self.image_view.setColorMap(pg.ColorMap([0, 0.25, 0.5, 0.75, 1], colors))
        except Exception as e:
            print(f"Error updating colormap: {str(e)}")

    def draw_roi(self):
        if self.stacked_widget.currentWidget() == self.image_view:
            if self.current_roi:
                self.image_view.removeItem(self.current_roi)
            self.current_roi = RectROI([50, 50], [100, 100], pen=(0, 9))
            self.image_view.addItem(self.current_roi)
        else:
            self.status_label.setText("Please select an image dataset to draw ROI.")

    def update_frame_indices(self):
        """Update the reference frame and other frame spinboxes when current frame changes"""
        if self.stacked_widget.currentWidget() == self.image_view:
            current_idx = self.image_view.currentIndex
            self.ref_frame_spin.setValue(current_idx)
            self.other_frame_spin.setValue(current_idx)
            # Apply autoscale if enabled
            if self.auto_linear_scale_checkbox.isChecked():
                self.apply_autoscale()

    def analyze_speckle(self):
        if not self.selected_dataset_path or not isinstance(self.current_h5_obj, h5py.Dataset):
            self.status_label.setText("Please select a dataset first.")
            return
        if not self.current_roi:
            self.status_label.setText("Please draw an ROI first.")
            return
        if self.binned_data is None:
            self.status_label.setText("Please load dataset first by double-clicking on it.")
            return

        pos = self.current_roi.pos()
        size = self.current_roi.size()
        x, y = int(pos.x()), int(pos.y())
        w, h = int(size.x()), int(size.y())
        roi = (x, y, w, h)

        # Get reference and other frame indices
        ref_frame = self.ref_frame_spin.value()
        other_frame = self.other_frame_spin.value()
        
        # Ensure the frame indices are valid
        max_frame = self.binned_data.shape[0] - 1
        if ref_frame > max_frame or other_frame > max_frame:
            self.status_label.setText(f"Frame index out of range. Max frame is {max_frame}.")
            return
        
        try:
            # Use the already loaded data from memory
            self.status_label.setText(f"Analyzing speckle pattern: comparing frame {ref_frame} with frame {other_frame}...")
            QApplication.processEvents()  # Update UI
            
            # Create analyzer with already loaded data, passing both frame indices
            speckle_analyzer = Speckle_analyzer(
                data=self.binned_data, 
                ROI=roi,
                frame_index=ref_frame,
                compare_frame_index=other_frame  # New parameter for the other frame
            )
            speckle_analyzer.plot_report()
            self.status_label.setText(f"Speckle analysis complete: compared frame {ref_frame} with frame {other_frame}")
        except Exception as e:
            self.status_label.setText(f"Error during speckle analysis: {str(e)}")
            print(f"Error during speckle analysis: {str(e)}")
            traceback.print_exc()

    def validate_parameters(self, dataset):
        """Validate and correct loading parameters to prevent errors"""
        if self.enable_crop_checkbox.isChecked():
            # Validate X range
            x_start = self.x_start_spin.value()
            x_end = self.x_end_spin.value()
            
            # Get max width (last dimension for both 2D and 3D)
            if len(dataset.shape) == 3:
                max_width = dataset.shape[2] - 1
            else:
                max_width = dataset.shape[1] - 1
                
            # Ensure x_end is at least x_start + 1 and not beyond max
            if x_end <= x_start or x_end > max_width:
                x_end = min(x_start + 10, max_width)
                self.x_end_spin.setValue(x_end)
                
            # Ensure x_start is not negative
            if x_start < 0:
                x_start = 0
                self.x_start_spin.setValue(x_start)
                
            # Validate Y range
            y_start = self.y_start_spin.value()
            y_end = self.y_end_spin.value()
            
            # Get max height (2nd dimension for 3D, 1st for 2D)
            if len(dataset.shape) == 3:
                max_height = dataset.shape[1] - 1
            else:
                max_height = dataset.shape[0] - 1
                
            # Ensure y_end is at least y_start + 1 and not beyond max
            if y_end <= y_start or y_end > max_height:
                y_end = min(y_start + 10, max_height)
                self.y_end_spin.setValue(y_end)
                
            # Ensure y_start is not negative
            if y_start < 0:
                y_start = 0
                self.y_start_spin.setValue(y_start)
                
            # Check if resulting crop is too small (at least 2x2)
            if (x_end - x_start < 1) or (y_end - y_start < 1):
                self.status_label.setText("Warning: Crop region too small. Adjusting...")
                QApplication.processEvents()
                
                # Adjust to minimum 2x2 crop
                if x_end - x_start < 1:
                    if x_end < max_width:
                        x_end = x_start + 1
                    else:
                        x_start = x_end - 1
                        
                if y_end - y_start < 1:
                    if y_end < max_height:
                        y_end = y_start + 1
                    else:
                        y_start = y_end - 1
                        
                # Update UI
                self.x_start_spin.setValue(x_start)
                self.x_end_spin.setValue(x_end)
                self.y_start_spin.setValue(y_start)
                self.y_end_spin.setValue(y_end)
        
        # Validate frame range for 3D datasets
        if len(dataset.shape) == 3:
            start_frame = self.start_frame_spin.value()
            end_frame = self.end_frame_spin.value()
            max_frame = dataset.shape[0] - 1
            
            if start_frame < 0:
                start_frame = 0
                self.start_frame_spin.setValue(start_frame)
                
            if end_frame > max_frame:
                end_frame = max_frame
                self.end_frame_spin.setValue(end_frame)
                
            if start_frame > end_frame:
                start_frame = max(0, end_frame - 10)
                self.start_frame_spin.setValue(start_frame)
                
            # Check if trying to load too many frames (arbitrary limit of 500 for memory safety)
            if end_frame - start_frame > 500:
                original_start = start_frame
                original_end = end_frame
                end_frame = start_frame + 500
                self.end_frame_spin.setValue(end_frame)
                self.status_label.setText(f"Warning: Too many frames requested ({original_end-original_start+1}). Limiting to 500 frames.")
                QApplication.processEvents()
                
        # Validate bin factor
        bin_factor = int(self.bin_combo.currentText().split(' ')[0])
        if bin_factor <= 0:
            self.bin_combo.setCurrentText("1 (None)")
            
        # No need to check upper limit here as that's handled during actual loading
        # when we know the cropped dimensions

    def load_motor_positions(self, directory_path, experiment_name, cluster_number=None):
        """
        Load motor positions from JSON files.
        
        Args:
            directory_path: Directory containing motor position files
            experiment_name: Name of the experiment
            cluster_number: If specified, load only this cluster's motor positions.
                          If None, load all clusters or main file.
        
        Returns motor data dict or None if not found.
        """
        # If a specific cluster is requested, load only that cluster's motor positions
        if cluster_number is not None:
            cluster_file = os.path.join(directory_path, 
                                       f"{experiment_name}_motor_position_analysis_cluster{cluster_number:04d}.json")
            if os.path.exists(cluster_file):
                print(f"Loading motor positions for cluster {cluster_number}: {os.path.basename(cluster_file)}")
                with open(cluster_file, 'r') as f:
                    motor_data = json.load(f)
                print(f"  Loaded {len(motor_data.get('positions', []))} positions from cluster {cluster_number}")
                return motor_data
            else:
                print(f"Cluster motor position file not found: {cluster_file}")
                return None
        
        # If no specific cluster, look for main motor positions file
        motor_file = os.path.join(directory_path, f"{experiment_name}_motor_positions.json")
        
        if os.path.exists(motor_file):
            print(f"Loading motor positions from: {motor_file}")
            with open(motor_file, 'r') as f:
                motor_data = json.load(f)
            return motor_data
        
        # Look for individual cluster motor position files and combine them
        print(f"Main motor positions file not found: {motor_file}")
        print("Attempting to combine individual cluster motor position files...")
        
        cluster_motor_pattern = os.path.join(directory_path, f"{experiment_name}_motor_position_analysis_cluster*.json")
        cluster_motor_files = glob.glob(cluster_motor_pattern)
        
        if not cluster_motor_files:
            print(f"No cluster motor position files found matching pattern: {cluster_motor_pattern}")
            return None
        
        # Sort cluster files by cluster number
        def extract_cluster_number(filename):
            match = re.search(r'_cluster(\d+)\.json$', filename)
            return int(match.group(1)) if match else 0
        
        cluster_motor_files.sort(key=extract_cluster_number)
        print(f"Found {len(cluster_motor_files)} cluster motor position files")
        
        # Load the first file to get the base structure
        with open(cluster_motor_files[0], 'r') as f:
            base_data = json.load(f)
        
        # Initialize the combined data structure
        motor_data = {
            "experiment": experiment_name,
            "motors": base_data["motors"],
            "motor_pvs": base_data["motor_pvs"],
            "total_clusters": base_data["total_clusters"],
            "total_points": 0,
            "positions": []
        }
        
        # Combine all positions from all cluster files
        total_positions = 0
        for cluster_file in cluster_motor_files:
            print(f"Loading positions from {os.path.basename(cluster_file)}...")
            
            with open(cluster_file, 'r') as f:
                cluster_data = json.load(f)
            
            # Add all positions from this cluster
            cluster_positions = cluster_data.get("positions", [])
            motor_data["positions"].extend(cluster_positions)
            total_positions += len(cluster_positions)
            
            print(f"  Added {len(cluster_positions)} positions")
        
        motor_data["total_points"] = total_positions
        print(f"Combined {total_positions} total positions from {len(cluster_motor_files)} cluster files")
        
        return motor_data

    def plot_motor_positions(self):
        """
        Generate scatter and contour plots of ROI intensities as a function of motor positions.
        """
        if not self.selected_file_path:
            self.status_label.setText("No file selected!")
            return
        
        if self.binned_data is None:
            self.status_label.setText("Please load dataset first by double-clicking on it.")
            return
        
        if not self.current_roi:
            self.status_label.setText("Please draw an ROI first.")
            return
        
        # Extract experiment name, directory, and cluster number from the selected file path
        directory_path = os.path.dirname(self.selected_file_path)
        filename = os.path.basename(self.selected_file_path)
        
        # Try to extract experiment name and cluster number from filename
        # Pattern: experimentname_clusterXXXX_YYY.h5 or experimentname_clusterXXXX.h5
        cluster_match = re.match(r'(.+?)_cluster(\d+)', filename)
        if cluster_match:
            experiment_name = cluster_match.group(1)
            cluster_number = int(cluster_match.group(2))
            print(f"Detected cluster file: cluster {cluster_number}")
        else:
            # Fallback: use filename without extension, no specific cluster
            experiment_name = os.path.splitext(filename)[0]
            cluster_number = None
            print(f"Main file detected (no cluster number)")
        
        print(f"Looking for motor positions for experiment: {experiment_name}")
        print(f"In directory: {directory_path}")
        
        # Load motor positions
        self.status_label.setText("Loading motor positions...")
        QApplication.processEvents()
        
        motor_data = self.load_motor_positions(directory_path, experiment_name, cluster_number)
        
        if motor_data is None:
            self.status_label.setText("Could not find motor position files!")
            return
        
        self.motor_positions = motor_data
        self.motor_names = motor_data["motors"]
        self.experiment_name = experiment_name
        
        print(f"Loaded {len(motor_data['positions'])} motor positions")
        print(f"Motors: {self.motor_names}")
        
        # Extract ROI position
        pos = self.current_roi.pos()
        size = self.current_roi.size()
        x, y = int(pos.x()), int(pos.y())
        w, h = int(size.x()), int(size.y())
        
        print(f"ROI: x={x}, y={y}, w={w}, h={h}")
        
        # Extract ROI intensities for each frame
        self.status_label.setText("Extracting ROI intensities...")
        QApplication.processEvents()
        
        num_frames = self.binned_data.shape[0]
        num_positions = len(motor_data['positions'])
        
        # Check if number of frames matches number of positions
        if num_frames != num_positions:
            print(f"Warning: Number of frames ({num_frames}) does not match number of positions ({num_positions})")
            # Use the minimum
            num_points = min(num_frames, num_positions)
        else:
            num_points = num_frames
        
        # Extract motor positions (hexapod x and z) and ROI metrics
        x_positions = []
        y_positions = []
        roi_intensities = []
        roi_com_x = []
        roi_com_y = []
        
        for i in range(num_points):
            pos_data = motor_data['positions'][i]
            # Motor positions: "0" is first motor (hexapod1_z), "1" is second motor (hexapod1_x)
            actual = pos_data.get('actual', {})
            x_pos = actual.get("1", 0)  # hexapod1_x
            y_pos = actual.get("0", 0)  # hexapod1_z
            
            x_positions.append(x_pos)
            y_positions.append(y_pos)
            
            # Extract ROI region
            frame = self.binned_data[i]
            roi_region = frame[y:y+h, x:x+w]
            
            # Calculate total intensity
            roi_intensity = np.sum(roi_region)
            roi_intensities.append(roi_intensity)
            
            # Calculate center of mass within ROI
            if roi_intensity > 0:
                # Create coordinate grids for the ROI
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                # Calculate center of mass
                com_x = np.sum(x_coords * roi_region) / roi_intensity
                com_y = np.sum(y_coords * roi_region) / roi_intensity
                roi_com_x.append(com_x)
                roi_com_y.append(com_y)
            else:
                roi_com_x.append(w / 2.0)  # Default to center if no intensity
                roi_com_y.append(h / 2.0)
        
        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)
        roi_intensities = np.array(roi_intensities)
        roi_com_x = np.array(roi_com_x)
        roi_com_y = np.array(roi_com_y)
        
        print(f"Extracted {len(roi_intensities)} ROI measurements")
        print(f"Motor X range: {x_positions.min():.6f} to {x_positions.max():.6f}")
        print(f"Motor Y range: {y_positions.min():.6f} to {y_positions.max():.6f}")
        print(f"Intensity range: {roi_intensities.min():.2f} to {roi_intensities.max():.2f}")
        print(f"COM-X range: {roi_com_x.min():.2f} to {roi_com_x.max():.2f} pixels")
        print(f"COM-Y range: {roi_com_y.min():.2f} to {roi_com_y.max():.2f} pixels")
        
        # Generate plots
        self.status_label.setText("Generating plots...")
        QApplication.processEvents()
        
        self._generate_motor_position_plots(x_positions, y_positions, 
                                            roi_intensities, roi_com_x, roi_com_y,
                                            self.motor_names, experiment_name)
        
        self.status_label.setText(f"Motor position plots displayed")

    def _generate_motor_position_plots(self, x_positions, y_positions, 
                                       roi_intensities, roi_com_x, roi_com_y,
                                       motor_names, sample_name):
        """
        Generate scatter and contour plots from motor position and ROI data.
        Shows total intensity, center of mass X, and center of mass Y.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with 2 rows, 3 columns (scatter + contour for each metric)
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'ROI Analysis vs Motor Position - {sample_name}', fontsize=16)
            
            motor_x_name = motor_names[1] if len(motor_names) > 1 else "Motor X"
            motor_y_name = motor_names[0] if len(motor_names) > 0 else "Motor Y"
            
            # Normalize COM values for better visualization
            com_x_norm = (roi_com_x - np.min(roi_com_x)) / (np.max(roi_com_x) - np.min(roi_com_x) + 1e-10)
            com_y_norm = (roi_com_y - np.min(roi_com_y)) / (np.max(roi_com_y) - np.min(roi_com_y) + 1e-10)
            
            # Three metrics to plot (using normalized COM for better visualization)
            metrics = [
                {'data': roi_intensities, 'name': 'Integrated Intensity', 'unit': 'counts', 'cmap': 'viridis'},
                {'data': com_x_norm, 'name': 'Center of Mass X', 'unit': 'normalized', 'cmap': 'plasma'},
                {'data': com_y_norm, 'name': 'Center of Mass Y', 'unit': 'normalized', 'cmap': 'inferno'}
            ]
            
            # Calculate plot limits with padding (let matplotlib handle aspect ratio)
            x_min, x_max = np.min(x_positions), np.max(x_positions)
            y_min, y_max = np.min(y_positions), np.max(y_positions)
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Add 5% padding
            x_pad = x_range * 0.05
            y_pad = y_range * 0.05
            x_min, x_max = x_min - x_pad, x_max + x_pad
            y_min, y_max = y_min - y_pad, y_max + y_pad
            
            # Plot each metric
            for col, metric in enumerate(metrics):
                data = metric['data']
                name = metric['name']
                unit = metric['unit']
                cmap = metric['cmap']
                
                # Skip if all values are zero or invalid
                if np.all(data == 0) or not np.any(np.isfinite(data)):
                    axes[0, col].text(0.5, 0.5, f'No Data\n({name})', 
                                     ha='center', va='center', transform=axes[0, col].transAxes, fontsize=14)
                    axes[1, col].text(0.5, 0.5, f'No Data\n({name})', 
                                     ha='center', va='center', transform=axes[1, col].transAxes, fontsize=14)
                    axes[0, col].set_title(f'{name} Scatter', fontsize=12)
                    axes[1, col].set_title(f'{name} Contour', fontsize=12)
                    continue
                
                # Top row: Scatter plots
                scatter = axes[0, col].scatter(x_positions, y_positions, c=data, 
                                              cmap=cmap, s=30, alpha=0.7)
                axes[0, col].set_title(f'{name} Scatter', fontsize=12)
                axes[0, col].set_xlabel(f'{motor_x_name}', fontsize=10)
                axes[0, col].set_ylabel(f'{motor_y_name}', fontsize=10)
                axes[0, col].set_xlim(x_min, x_max)
                axes[0, col].set_ylim(y_min, y_max)
                axes[0, col].grid(True, linestyle='--', alpha=0.6)
                cbar = plt.colorbar(scatter, ax=axes[0, col])
                cbar.set_label(unit, fontsize=10)
                
                # Bottom row: Contour plots (if enough points)
                if len(data) >= 4:
                    try:
                        # Create regular grid for interpolation
                        xi = np.linspace(x_positions.min(), x_positions.max(), 100)
                        yi = np.linspace(y_positions.min(), y_positions.max(), 100)
                        XI, YI = np.meshgrid(xi, yi)
                        
                        # Interpolate data to regular grid
                        try:
                            ZI = griddata((x_positions, y_positions), data, (XI, YI), 
                                        method='cubic', fill_value=np.nan)
                        except:
                            # Fallback to linear if cubic fails
                            ZI = griddata((x_positions, y_positions), data, (XI, YI), 
                                        method='linear', fill_value=np.nan)
                        
                        # Get data range for consistent scaling
                        vmin, vmax = np.nanmin(data), np.nanmax(data)
                        
                        # Create filled contour plot
                        contour = axes[1, col].contourf(XI, YI, ZI, levels=20, cmap=cmap, 
                                                       vmin=vmin, vmax=vmax)
                        
                        # Add white contour lines for better definition
                        contour_lines = axes[1, col].contour(XI, YI, ZI, levels=10, colors='white', 
                                                            alpha=0.4, linewidths=0.5)
                        
                        axes[1, col].set_title(f'{name} Contour', fontsize=12)
                        axes[1, col].set_xlabel(f'{motor_x_name}', fontsize=10)
                        axes[1, col].set_ylabel(f'{motor_y_name}', fontsize=10)
                        axes[1, col].set_xlim(x_min, x_max)
                        axes[1, col].set_ylim(y_min, y_max)
                        axes[1, col].grid(True, linestyle='--', alpha=0.6)
                        cbar = plt.colorbar(contour, ax=axes[1, col])
                        cbar.set_label(unit, fontsize=10)
                        
                    except Exception as contour_error:
                        print(f"Could not generate contour for {name}: {contour_error}")
                        axes[1, col].text(0.5, 0.5, f'Contour Error\n{str(contour_error)}', 
                                        ha='center', va='center', transform=axes[1, col].transAxes, fontsize=10)
                else:
                    axes[1, col].text(0.5, 0.5, f'Need ≥4 points\nfor contour\n(have {len(data)})', 
                                    ha='center', va='center', transform=axes[1, col].transAxes, fontsize=10)
            
            # Adjust layout and show
            plt.tight_layout()
            plt.show()  # Show interactively like speckle profiler
            
            print(f"Motor position plots displayed")
            
        except Exception as e:
            print(f"Error generating plots from motor position data: {e}")
            traceback.print_exc()
            self.status_label.setText(f"Error generating plots: {str(e)}")

    def closeEvent(self, event):
        if self.h5_file is not None:
            self.h5_file.close()
        event.accept()

if __name__ == "__main__":
    def exception_hook(exctype, value, tb):
        print(''.join(traceback.format_exception(exctype, value, tb)))
        sys.exit(1)

    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    window = HDF5ImageViewer()
    window.show()
    sys.exit(app.exec_())