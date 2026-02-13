"""
Copyright (c) UChicago Argonne, LLC. All rights reserved.
See LICENSE file.
"""

# ==============================================================================

import csv
import h5py
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from pyqtgraph.Qt import QtGui, QtCore
from rsMap3D.config.rsmap3dconfigparser import RSMap3DConfigParser
from rsMap3D.datasource.Sector33SpecDataSource import Sector33SpecDataSource
from rsMap3D.datasource.DetectorGeometryForXrayutilitiesReader import DetectorGeometryForXrayutilitiesReader as detReader
from rsMap3D.datasource.InstForXrayutilitiesReader import InstForXrayutilitiesReader as instrReader
from rsMap3D.gui.rsm3dcommonstrings import BINARY_OUTPUT
from rsMap3D.mappers.gridmapper import QGridMapper
from rsMap3D.mappers.output.vtigridwriter import VTIGridWriter
from rsMap3D.transforms.unitytransform3d import UnityTransform3D
from rsMap3D.utils.srange import srange
from scipy import ndimage
import tifffile as tiff
import time
import vtk
from vtk.util import numpy_support as npSup
import warnings
import xrayutilities as xu

# ==============================================================================

class PostPlottingWidget(QtGui.QWidget):

    """
    Houses docked widgets components for Post Plotting widget
    """

    def __init__ (self):
        super().__init__()

        # Docks & Widgets ------------------------------------------------------
        self.dock_area = DockArea()
        self.createDocks()
        self.createWidgets()

        # Layout ---------------------------------------------------------------
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.dock_area)
        self.setLayout(self.layout)

        # Conversion parameter dialog ------------------------------------------
        # Instantiated for later use
        self.conversion_dialog = ConversionParametersDialog()
        self.vti_creation_dialog = VTICreationDialog()

    # --------------------------------------------------------------------------

    def createDocks(self):
        """
        - Creates docks
        - Adds docks to dock area in main widget
        """

        # Dock Creation --------------------------------------------------------
        self.data_selection_dock = Dock("Data Selection", size=(100, 100), hideTitle=True)
        self.analysis_dock = Dock("Analysis", size=(400, 100))
        self.roi_analysis_dock = Dock("ROI", size=(400, 100))
        self.data_dock = Dock("Data", size=(400, 100), hideTitle=True)
        self.line_roi_analysis_dock = Dock("Slicing", size=(400, 100))

        # Adding Docks to Area -------------------------------------------------
        self.dock_area.addDock(self.data_selection_dock)
        self.dock_area.addDock(self.line_roi_analysis_dock, "bottom", self.data_selection_dock)
        self.dock_area.addDock(self.analysis_dock, "above", self.line_roi_analysis_dock)
        self.dock_area.addDock(self.roi_analysis_dock, "above", self.analysis_dock)
        self.dock_area.addDock(self.data_dock, "right", self.data_selection_dock)
        self.dock_area.moveDock(self.analysis_dock, "above", self.roi_analysis_dock)

    # --------------------------------------------------------------------------

    def createWidgets(self):
        """
        - Creates instances of subwidgets
        - Adds each subwidget to its respective dock
        """

        # Widget Creation ------------------------------------------------------
        self.data_selection_widget = DataSelectionWidget(self)
        self.analysis_widget = AnalysisWidget(self)
        self.roi_analysis_widget = ROIAnalysisWidget(self)
        self.data_widget = DataWidget(self)
        self.line_roi_analysis_widget = LineROIAnalysisWidget(self)

        # Adding Widgets to Docks ----------------------------------------------
        self.data_selection_dock.addWidget(self.data_selection_widget)
        self.analysis_dock.addWidget(self.analysis_widget)
        self.roi_analysis_dock.addWidget(self.roi_analysis_widget)
        self.line_roi_analysis_dock.addWidget(self.line_roi_analysis_widget)
        self.data_dock.addWidget(self.data_widget)

# ==============================================================================

class DataSelectionWidget(QtGui.QWidget):

    """
    Allows user to select:
    - A project file
    - A SPEC file
    - Which scan in SPEC file to view
    - Parameters to convert scan to reciprocal space

    *** TODO: Add VTI Creation dialog and use experimental layout
    """

    def __init__ (self, parent):
        super(DataSelectionWidget, self).__init__(parent)
        self.main_widget = parent

        # Widget Creation ------------------------------------------------------
        self.project_btn = QtGui.QPushButton("Set Project")
        self.project_txtbox = QtGui.QLineEdit()
        self.project_txtbox.setReadOnly(True)
        self.spec_file_listbox = QtGui.QListWidget()
        self.vti_file_listbox = QtGui.QListWidget()
        self.select_vti_btn = QtGui.QPushButton("Select VTI File")
        self.create_vti_btn = QtGui.QPushButton("Create VTI File")
        self.vti_txtbox = QtGui.QLineEdit()
        self.vti_txtbox.setReadOnly(True)
        self.pixel_count_lbl = QtGui.QLabel("Pixel Count:")
        self.pixel_count_txtbox = QtGui.QLineEdit()
        self.pixel_count_txtbox.setReadOnly(True)
        self.h_lbl = QtGui.QLabel("H Range:")
        self.k_lbl = QtGui.QLabel("K Range:")
        self.l_lbl = QtGui.QLabel("L Range:")
        self.h_txtbox = QtGui.QLineEdit()
        self.h_txtbox.setReadOnly(True)
        self.k_txtbox = QtGui.QLineEdit()
        self.k_txtbox.setReadOnly(True)
        self.l_txtbox = QtGui.QLineEdit()
        self.l_txtbox.setReadOnly(True)
        self.vti_info_gbox = QtGui.QGroupBox("VTI");

        self.scan_directory_listbox = QtGui.QListWidget()
        self.conversion_btn = QtGui.QPushButton("Parameters")
        self.process_btn = QtGui.QPushButton("Display")
        self.slice_direction_lbl = QtGui.QLabel("Slice Direction:")
        self.slice_direction_cbox = QtGui.QComboBox()
        self.slice_direction_cbox.addItems(["X(H)", "Y(K)", "Z(L)"])

        # Layout ---------------------------------------------------------------
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)

        self.vti_info_layout = QtGui.QGridLayout()
        self.vti_info_gbox.setLayout(self.vti_info_layout)

        self.layout.addWidget(self.select_vti_btn, 0, 0)
        self.layout.addWidget(self.create_vti_btn, 0, 1)
        self.layout.addWidget(self.vti_txtbox, 1, 0, 1, 2)
        self.layout.addWidget(self.vti_info_gbox, 2, 0, 1, 2)
        self.layout.addWidget(self.slice_direction_lbl, 3, 0)
        self.layout.addWidget(self.slice_direction_cbox, 3, 1)
        self.layout.addWidget(self.process_btn, 4, 0, 1, 2)

        self.vti_info_layout.addWidget(self.pixel_count_lbl, 0, 0, 1, 2)
        self.vti_info_layout.addWidget(self.pixel_count_txtbox, 0, 2, 1, 2)
        self.vti_info_layout.addWidget(self.h_lbl, 1, 0)
        self.vti_info_layout.addWidget(self.h_txtbox, 1, 1, 1, 3)
        self.vti_info_layout.addWidget(self.k_lbl, 2, 0)
        self.vti_info_layout.addWidget(self.k_txtbox, 2, 1, 1, 3)
        self.vti_info_layout.addWidget(self.l_lbl, 3, 0)
        self.vti_info_layout.addWidget(self.l_txtbox, 3, 1, 1, 3)

        # Signals --------------------------------------------------------------
        self.select_vti_btn.clicked.connect(self.selectVTI)
        self.create_vti_btn.clicked.connect(self.showVTICreationDialog)
        self.process_btn.clicked.connect(self.createDataset)
        self.slice_direction_cbox.currentTextChanged.connect(self.changeSliceDirection)

    # --------------------------------------------------------------------------

    def selectVTI(self):

        """
        Selects VTI file from dialog and previews info about file.
        """

        self.vti_path = QtGui.QFileDialog.getOpenFileName(self, "", "", "VTI Files (*.vti)")[0]
        self.vti_txtbox.setText(self.vti_path)

        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(self.vti_path)
        reader.Update()

        # Preview information for VTI file -------------------------------------
        data = reader.GetOutput()
        dimensions = data.GetDimensions()
        bounds = data.GetBounds()

        h_count, k_count, l_count = dimensions
        h_min, h_max = bounds[0], bounds[1]
        k_min, k_max = bounds[2], bounds[3]
        l_min, l_max = bounds[4], bounds[5]

        self.pixel_count_txtbox.setText(f"{dimensions}")
        self.h_txtbox.setText(f"({round(h_min, 5)},{round(h_max, 5)})")
        self.k_txtbox.setText(f"({round(k_min, 5)},{round(k_max, 5)})")
        self.l_txtbox.setText(f"({round(l_min, 5)},{round(l_max, 5)})")

    # --------------------------------------------------------------------------

    def showVTICreationDialog(self):

        """
        Displays VTICreationDialog
        """

        self.main_widget.vti_creation_dialog.show()

    # --------------------------------------------------------------------------

    def createDataset(self):

        """
        Creates HKL-converted dataset and bounds for dataset
        """

        try:
            if self.vti_path == "":
                dataset = []
                dataset_rect = None

                # For VTI file creation
                scan_name = self.scan_directory_listbox.currentItem().text()
                scan_number = scan_name[1:]

                # Maps/interpolates data into reciprocal space
                self.vti_path = ConversionLogic.createVTIFile(self.project_path, \
                    self.spec_file_path, self.detector_path, self.instrument_path, \
                    scan_number, self.pixel_count_nx, self.pixel_count_ny, \
                    self.pixel_count_nz)

            # Creates axis limits and dataset ----------------------------------
            axes, dataset = ConversionLogic.loadData(self.vti_path)

            self.dataset = dataset
            self.h_values = np.array(axes[0])
            self.k_values = np.array(axes[1])
            self.l_values = np.array(axes[2])

            # IN PROGRESS ******************************************************
            # Creates 4D array to map pixels in dataset to HKL positions
            h, k, l = self.h_values, self.k_values, self.l_values
            h_map = np.reshape(np.tile(h, k.shape[0] * l.shape[0]), dataset.shape)
            k_map = np.reshape(np.tile(np.repeat(k, h.shape[0]), l.shape[0]), dataset.shape)
            l_map = np.reshape(np.repeat(l, h.shape[0] * k.shape[0]), dataset.shape)

            hkl_map = np.stack((h_map, k_map, l_map), axis=-1)
            
            axis_labels = ["H", "K", "L"]
            # MINI DEMO: Locating max pixel intensity ||||||||||||||||||||||||||
            max = np.unravel_index(np.argmax(dataset, axis=None), dataset.shape)
            print("\nLocating Max Intensity in Dataset")
            print("=================================\n")
            print(f"Axis Labels: {axis_labels}")
            print(f"{axis_labels[0]} Index: {max[0]}")
            print(f"{axis_labels[1]} Index: {max[1]}")
            print(f"{axis_labels[2]} Index: {max[2]}\n")
            print(f"{axis_labels[0]} Map[{max[0]}]: {h_map[max[0]][max[1]][max[2]]}")
            print(f"{axis_labels[1]} Map[{max[1]}]: {k_map[max[0]][max[1]][max[2]]}")
            print(f"{axis_labels[2]} Map[{max[2]}]: {l_map[max[0]][max[1]][max[2]]}\n")
            print(f"HKL Map[{max[0]}][{max[1]}][{max[2]}]: {hkl_map[max[0]][max[1]][max[2]]}\n")
            print(f"Max Intensity: {dataset[max[0]][max[1]][max[2]]}\n")
            print(f"Dataset Shape: {dataset.shape}")
            print(f"HKL Map Shape: {hkl_map.shape}")
            # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

            # ******************************************************************


            self.hkl_values = [self.h_values, self.k_values, self.l_values]

            dataset_rect = [self.h_values, self.k_values, self.l_values]

            # Loads dataset into datawidget image window
            self.main_widget.data_widget.displayDataset(new_dataset=True, \
                dataset_rect=dataset_rect)

        except Exception:
            msg_box = QtGui.QMessageBox()
            msg_box.setWindowTitle("Error")
            msg_box.setText("Error Loading Data")
            msg_box.exec_()

# --------------------------------------------------------------------------

    def changeSliceDirection(self):

        """
        Changes slice direction in data widget.
        (Function exists in this class because the spinbox is housed in the
            DataSelection layout)
        """

        direction = self.sender().currentText()
        self.main_widget.data_widget.displayDataset()

# ==============================================================================

class DataWidget(pg.ImageView):

    """
    Plots 3d dataset
    """

    def __init__ (self, parent):
        super(DataWidget, self).__init__(parent, view=pg.PlotItem(), imageItem=pg.ImageItem())
        self.main_widget = parent

        self.dataset = []
        self.color_dataset = None
        self.slice_direction = None
        self.dataset_rect = None
        self.scene_point = None

        self.view_box = self.view.getViewBox()
        self.view.setAspectLocked(False)

        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        # Crosshair ------------------------------------------------------------
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.v_line.setVisible(False)
        self.h_line.setVisible(False)
        self.view.addItem(self.v_line, ignoreBounds=True)
        self.view.addItem(self.h_line, ignoreBounds=True)

        #ROIs ------------------------------------------------------------------
        self.roi_1 = self.main_widget.roi_analysis_widget.roi_1.roi
        self.roi_2 = self.main_widget.roi_analysis_widget.roi_2.roi
        self.roi_3 = self.main_widget.roi_analysis_widget.roi_3.roi
        self.roi_4 = self.main_widget.roi_analysis_widget.roi_4.roi
        self.addItem(self.roi_1)
        self.addItem(self.roi_2)
        self.addItem(self.roi_3)
        self.addItem(self.roi_4)

    # --------------------------------------------------------------------------

    def displayDataset(self, slice_direction=None, new_dataset=False,
        dataset_rect=None):

        """
        Displays 3d dataset in plot

        TODO Fix colormap
        """

        self.dataset = self.main_widget.data_selection_widget.dataset
        self.hkl_values = self.main_widget.data_selection_widget.hkl_values
        self.slice_direction = self.main_widget.data_selection_widget. \
            slice_direction_cbox.currentText()

        if dataset_rect != None:
            self.dataset_rect = dataset_rect

        # Checking Slice Direction ---------------------------------------------
        # Decides how the dataset is oriented in the viewing window
        # x_dir: direction of x-axis
        # y_dir: direction of y-axis
        # t_dir: direction of timeline
        if self.slice_direction == "X(H)":
            x_dir, y_dir, t_dir = 2, 1, 0
            self.view.setLabel(axis="left", text="K")
            self.view.setLabel(axis="bottom", text="L")
        elif self.slice_direction == "Y(K)":
            x_dir, y_dir, t_dir = 2, 0, 1
            self.view.setLabel(axis="left", text="H")
            self.view.setLabel(axis="bottom", text="L")
        else: # "Z(L)"
            x_dir, y_dir, t_dir = 1, 0, 2
            self.view.setLabel(axis="left", text="H")
            self.view.setLabel(axis="bottom", text="K")

        # Sets colormap for new datasets ---------------------------------------
        if new_dataset == True:
            # Normalize image with logarithmic colormap
            colormap_max = np.amax(self.dataset)
            norm = colors.LogNorm(vmax=colormap_max)
            shape = self.dataset.shape
            temp_reshaped_dataset = np.reshape(self.dataset, -1)
            norm_dataset = np.reshape(norm(temp_reshaped_dataset), shape)
            self.color_dataset = plt.cm.jet(norm_dataset)

        # Sets Scaling, Postion, and Orientation of Dataset --------------------
        axes = {"t":t_dir, "x":x_dir, "y":y_dir, "c":3}
        pos = (self.hkl_values[x_dir][0], self.hkl_values[y_dir][0])
        x_scale = (self.hkl_values[x_dir][1] - self.hkl_values[x_dir][0])
        y_scale = (self.hkl_values[y_dir][1] - self.hkl_values[y_dir][0])
        scale = (x_scale, y_scale)
        t_values = self.hkl_values[t_dir]

        self.setImage(self.color_dataset, axes=axes, pos=pos, scale=scale, \
            xvals=t_values)
        self.setCurrentIndex(0)

        # Enables widgets
        self.main_widget.analysis_widget.setEnabled(True)
        self.main_widget.roi_analysis_widget.setEnabled(True)
        self.main_widget.line_roi_analysis_widget.setEnabled(True)

        self.view_box.scene().sigMouseMoved.connect(self.updateMouse)
        self.updateMouse()
        self.main_widget.analysis_widget.updateScanInfo(self.dataset)
        self.main_widget.analysis_widget.updateMaxInfo(self.dataset, self.dataset_rect)

    # --------------------------------------------------------------------------

    def updateMouse(self, scene_point=None):

        """
        Updates mouse/crosshair positions
        """
        if scene_point != None:
            self.scene_point = scene_point
        if self.scene_point != None:
            self.view_point = self.view_box.mapSceneToView(self.scene_point)

            # x and y values of mouse
            x, y = self.view_point.x(), self.view_point.y()
        else:
            return

        # Crosshair
        self.v_line.setPos(x)
        self.h_line.setPos(y)

        self.main_widget.analysis_widget.updateMouseInfo(self.dataset, \
            self.dataset_rect, x, y, self.currentIndex, self.slice_direction)

# ==============================================================================

class AnalysisWidget(pg.LayoutWidget):

    """
    Houses basic information about data in datawidget viewing window
    """

    def __init__ (self, parent):
        super(AnalysisWidget, self).__init__(parent)
        self.main_widget = parent

        self.setEnabled(False)

        # Widget Creation ------------------------------------------------------
        # Scan Info
        self.scan_pixel_count_x_lbl = QtGui.QLabel("Pixel Count (x):")
        self.scan_pixel_count_x_txtbox = QtGui.QLineEdit()
        self.scan_pixel_count_x_txtbox.setReadOnly(True)
        self.scan_pixel_count_y_lbl = QtGui.QLabel("Pixel Count (y):")
        self.scan_pixel_count_y_txtbox = QtGui.QLineEdit()
        self.scan_pixel_count_y_txtbox.setReadOnly(True)
        self.scan_pixel_count_z_lbl = QtGui.QLabel("Pixel Count (z):")
        self.scan_pixel_count_z_txtbox = QtGui.QLineEdit()
        self.scan_pixel_count_z_txtbox.setReadOnly(True)

        # Mouse Info
        self.mouse_x_lbl = QtGui.QLabel("x Pos:")
        self.mouse_x_txtbox = QtGui.QLineEdit()
        self.mouse_x_txtbox.setReadOnly(True)
        self.mouse_y_lbl = QtGui.QLabel("y Pos:")
        self.mouse_y_txtbox = QtGui.QLineEdit()
        self.mouse_y_txtbox.setReadOnly(True)
        self.mouse_intensity_lbl = QtGui.QLabel("Intensity:")
        self.mouse_intensity_txtbox = QtGui.QLineEdit()
        self.mouse_intensity_txtbox.setReadOnly(True)
        self.mouse_h_lbl = QtGui.QLabel("H:")
        self.mouse_h_txtbox = QtGui.QLineEdit()
        self.mouse_h_txtbox.setReadOnly(True)
        self.mouse_k_lbl = QtGui.QLabel("K:")
        self.mouse_k_txtbox = QtGui.QLineEdit()
        self.mouse_k_txtbox.setReadOnly(True)
        self.mouse_l_lbl = QtGui.QLabel("L:")
        self.mouse_l_txtbox = QtGui.QLineEdit()
        self.mouse_l_txtbox.setReadOnly(True)

        # Max Info
        self.max_intensity_lbl = QtGui.QLabel("Intensity:")
        self.max_intensity_txtbox = QtGui.QLineEdit()
        self.max_intensity_txtbox.setReadOnly(True)
        self.max_h_lbl = QtGui.QLabel("H:")
        self.max_h_txtbox = QtGui.QLineEdit()
        self.max_h_txtbox.setReadOnly(True)
        self.max_k_lbl = QtGui.QLabel("K:")
        self.max_k_txtbox = QtGui.QLineEdit()
        self.max_k_txtbox.setReadOnly(True)
        self.max_l_lbl = QtGui.QLabel("L:")
        self.max_l_txtbox = QtGui.QLineEdit()
        self.max_l_txtbox.setReadOnly(True)

        # GroupBoxes -----------------------------------------------------------
        self.scan_gbox = QtGui.QGroupBox("Scan")
        self.slice_gbox = QtGui.QGroupBox("Slice")
        self.mouse_gbox = QtGui.QGroupBox("Mouse")
        self.max_gbox = QtGui.QGroupBox("Max")

        self.addWidget(self.scan_gbox, row=0, col=0)
        self.addWidget(self.mouse_gbox, row=0, col=1)
        self.addWidget(self.max_gbox, row=0, col=2)

        # Layouts --------------------------------------------------------------
        self.scan_layout = QtGui.QGridLayout()
        self.mouse_layout = QtGui.QGridLayout()
        self.max_layout = QtGui.QGridLayout()
        self.scan_gbox.setLayout(self.scan_layout)
        self.mouse_gbox.setLayout(self.mouse_layout)
        self.max_gbox.setLayout(self.max_layout)

        # Scan
        self.scan_layout.addWidget(self.scan_pixel_count_x_lbl, 0, 0)
        self.scan_layout.addWidget(self.scan_pixel_count_x_txtbox, 0, 1)
        self.scan_layout.addWidget(self.scan_pixel_count_y_lbl, 1, 0)
        self.scan_layout.addWidget(self.scan_pixel_count_y_txtbox, 1, 1)
        self.scan_layout.addWidget(self.scan_pixel_count_z_lbl, 2, 0)
        self.scan_layout.addWidget(self.scan_pixel_count_z_txtbox, 2, 1)

        # Mouse
        self.mouse_layout.addWidget(self.mouse_x_lbl, 0, 0)
        self.mouse_layout.addWidget(self.mouse_x_txtbox, 0, 1)
        self.mouse_layout.addWidget(self.mouse_y_lbl, 1, 0)
        self.mouse_layout.addWidget(self.mouse_y_txtbox, 1, 1)
        self.mouse_layout.addWidget(self.mouse_intensity_lbl, 2, 0)
        self.mouse_layout.addWidget(self.mouse_intensity_txtbox, 2, 1)
        self.mouse_layout.addWidget(self.mouse_h_lbl, 3, 0)
        self.mouse_layout.addWidget(self.mouse_h_txtbox, 3, 1)
        self.mouse_layout.addWidget(self.mouse_k_lbl, 4, 0)
        self.mouse_layout.addWidget(self.mouse_k_txtbox, 4, 1)
        self.mouse_layout.addWidget(self.mouse_l_lbl, 5, 0)
        self.mouse_layout.addWidget(self.mouse_l_txtbox, 5, 1)

        # Max
        self.max_layout.addWidget(self.max_intensity_lbl, 0, 0)
        self.max_layout.addWidget(self.max_intensity_txtbox, 0, 1)
        self.max_layout.addWidget(self.max_h_lbl, 1, 0)
        self.max_layout.addWidget(self.max_h_txtbox, 1, 1)
        self.max_layout.addWidget(self.max_k_lbl, 2, 0)
        self.max_layout.addWidget(self.max_k_txtbox, 2, 1)
        self.max_layout.addWidget(self.max_l_lbl, 3, 0)
        self.max_layout.addWidget(self.max_l_txtbox, 3, 1)

    # --------------------------------------------------------------------------

    def updateScanInfo(self, dataset):

        """
        Updates textboxes with dataset shape
        """
        self.scan_pixel_count_x_txtbox.setText(str(dataset.shape[0]))
        self.scan_pixel_count_y_txtbox.setText(str(dataset.shape[1]))
        self.scan_pixel_count_z_txtbox.setText(str(dataset.shape[2]))

    # --------------------------------------------------------------------------

    def updateMouseInfo(self, dataset, rect, x, y, index, slice_direction):

        """
        Updates mouse location and HKL value textboxes
        """
        # Mouse location in pixels
        self.mouse_x_txtbox.setText(str(x))
        self.mouse_y_txtbox.setText(str(y))

        # Checks slice direction and sets HKL positions accordingly
        # H direction
        if slice_direction == None or slice_direction == "X(H)":
            self.mouse_h_txtbox.setText(str(rect[0][0] + (rect[0][-1] - rect[0][0]) * index / dataset.shape[0]))
            self.mouse_k_txtbox.setText(str(y))
            self.mouse_l_txtbox.setText(str(x))
            h_index = index
            k_index = int(dataset.shape[1] * (y - rect[1][0]) / (rect[1][-1] - rect[1][0]))
            l_index = int(dataset.shape[2] * (x - rect[2][0]) / (rect[2][-1] - rect[2][0]))
        # K direction
        elif slice_direction == "Y(K)":
            self.mouse_h_txtbox.setText(str(y))
            self.mouse_k_txtbox.setText(str(rect[1][0] + (rect[1][-1] - rect[1][0]) * index / dataset.shape[1]))
            self.mouse_l_txtbox.setText(str(x))
            h_index = int(dataset.shape[0] * (y - rect[0][0]) / (rect[0][-1] - rect[0][0]))
            k_index = index
            l_index = int(dataset.shape[2] * (x - rect[2][0]) / (rect[2][-1] - rect[2][0]))
        # L direction
        else:
            self.mouse_h_txtbox.setText(str(y))
            self.mouse_k_txtbox.setText(str(x))
            self.mouse_l_txtbox.setText(str(rect[2][0] + (rect[2][-1] - rect[2][0]) * index / dataset.shape[2]))
            h_index = int(dataset.shape[0] * (y - rect[0][0]) / (rect[0][-1] - rect[0][0]))
            k_index = int(dataset.shape[1] * (x - rect[1][0]) / (rect[1][-1] - rect[1][0]))
            l_index = index

        # Sets intensity based on HKL positions
        try:
            intensity = int(dataset[h_index][k_index][l_index])
            self.mouse_intensity_txtbox.setText(str(intensity))
        except IndexError:
            self.mouse_intensity_txtbox.setText("")

    # --------------------------------------------------------------------------

    def updateMaxInfo(self, dataset, rect):

        """
        Updates maximum pixel position and intensity
        """
        max = np.amax(dataset)
        h_index, k_index, l_index = np.unravel_index(dataset.argmax(), dataset.shape)

        self.max_intensity_txtbox.setText(str(int(max)))
        self.max_h_txtbox.setText(str(rect[0][0] + (rect[0][-1] - rect[0][0]) * h_index / dataset.shape[0]))
        self.max_k_txtbox.setText(str(rect[1][0] + (rect[1][-1] - rect[1][0]) * k_index / dataset.shape[1]))
        self.max_l_txtbox.setText(str(rect[2][0] + (rect[2][-1] - rect[2][0]) * l_index / dataset.shape[2]))

# ==============================================================================

class ROIAnalysisWidget(pg.LayoutWidget):

    """
    Widget that houses four ROIWidgets and an ROISubtractionWidget
    """

    def __init__ (self, parent):
        super(ROIAnalysisWidget, self).__init__(parent)
        self.main_widget = parent

        self.setEnabled(False)
        self.roi_tabs = QtGui.QTabWidget()

        self.roi_1 = ROIWidget(self)
        self.roi_2 = ROIWidget(self)
        self.roi_3 = ROIWidget(self)
        self.roi_4 = ROIWidget(self)
        self.roi_sub = ROISubtractionWidget(self)

        self.roi_tabs.addTab(self.roi_1, "ROI 1")
        self.roi_tabs.addTab(self.roi_2, "ROI 2")
        self.roi_tabs.addTab(self.roi_3, "ROI 3")
        self.roi_tabs.addTab(self.roi_4, "ROI 4")
        self.roi_tabs.addTab(self.roi_sub, "Subtraction")

        self.addWidget(self.roi_tabs)

# ==============================================================================

class ROIWidget(QtGui.QWidget):

    """
    - Enables user to add a rectangular ROI to the dataset viewing Window
    - Displays average intensity within ROI over a series of slices
    """

    def __init__ (self, parent):
        super(ROIWidget, self).__init__(parent)
        self.roi_analysis_widget = parent
        self.main_widget = self.roi_analysis_widget.main_widget

        self.avg_intensity = None

        # Widget Creation ------------------------------------------------------
        # ROI
        self.roi = pg.ROI([-0.25, -0.25], [0.5, 0.5])
        self.roi.hide()
        self.pen = pg.mkPen(width=3)
        self.roi.setPen(self.pen)
        self.roi.addScaleHandle([0.5, 0], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0], [0.5, 0.5])

        # Avg Intensity Plot
        self.plot_widget = pg.PlotWidget()

        # Info
        self.visible_chkbox = QtGui.QCheckBox("Visible")
        self.color_btn = pg.ColorButton()
        self.x_lbl = QtGui.QLabel("x Pos:")
        self.x_sbox = QtGui.QDoubleSpinBox()
        self.x_sbox.setMinimum(-1000)
        self.x_sbox.setMaximum(1000)
        self.x_sbox.setDecimals(6)
        self.y_lbl = QtGui.QLabel("y Pos:")
        self.y_sbox = QtGui.QDoubleSpinBox()
        self.y_sbox.setMinimum(-1000)
        self.y_sbox.setMaximum(1000)
        self.y_sbox.setDecimals(6)
        self.width_lbl = QtGui.QLabel("Width:")
        self.width_sbox = QtGui.QDoubleSpinBox()
        self.width_sbox.setMinimum(0)
        self.width_sbox.setMaximum(1000)
        self.width_sbox.setDecimals(6)
        self.height_lbl = QtGui.QLabel("Height:")
        self.height_sbox = QtGui.QDoubleSpinBox()
        self.height_sbox.setMinimum(0)
        self.height_sbox.setMaximum(1000)
        self.height_sbox.setDecimals(6)
        self.outline_btn = QtGui.QPushButton("Outline Image")

        # Groupboxes -----------------------------------------------------------
        self.info_gbox = QtGui.QGroupBox()

        # Layout ---------------------------------------------------------------
        self.layout = QtGui.QGridLayout()
        self.info_layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.info_gbox.setLayout(self.info_layout)

        self.layout.addWidget(self.info_gbox, 0, 0)
        self.layout.addWidget(self.plot_widget, 0, 1)

        # Info
        self.info_layout.addWidget(self.visible_chkbox, 0, 0)
        self.info_layout.addWidget(self.color_btn, 0, 1)
        self.info_layout.addWidget(self.x_lbl, 1, 0)
        self.info_layout.addWidget(self.x_sbox, 1, 1)
        self.info_layout.addWidget(self.y_lbl, 2, 0)
        self.info_layout.addWidget(self.y_sbox, 2, 1)
        self.info_layout.addWidget(self.width_lbl, 3, 0)
        self.info_layout.addWidget(self.width_sbox, 3, 1)
        self.info_layout.addWidget(self.height_lbl, 4, 0)
        self.info_layout.addWidget(self.height_sbox, 4, 1)
        self.info_layout.addWidget(self.outline_btn, 5, 0, 1, 2)

        # Signals ------------------------------------------------------------
        self.visible_chkbox.clicked.connect(self.toggleVisibility)
        self.color_btn.sigColorChanged.connect(self.changeColor)
        self.width_sbox.valueChanged.connect(self.updateSize)
        self.height_sbox.valueChanged.connect(self.updateSize)
        self.x_sbox.valueChanged.connect(self.updatePosition)
        self.y_sbox.valueChanged.connect(self.updatePosition)
        self.roi.sigRegionChanged.connect(self.updateAnalysis)
        self.roi.sigRegionChanged.connect(self.plotAverageIntensity)
        self.outline_btn.clicked.connect(self.center)

        # Keeps track of whether textboxes or roi was updated last
        # Helps avoid infinite loop of updating
        # Acts like a semaphore of sorts
        self.updating = ""

    # --------------------------------------------------------------------------

    def toggleVisibility(self, state):

        """
        Changes ROI visibility
        """

        if state:
            self.roi.show()
        else:
            self.roi.hide()

        # TODO: Find a place for this garbage
        self.roi.sigRegionChanged.connect(self.roi_analysis_widget.roi_sub.plotData)
        self.updateAnalysis()

    # --------------------------------------------------------------------------

    def changeColor(self):

        """
        Changes ROI color
        """

        color = self.color_btn.color()
        pen = pg.mkPen(color, width=3)
        self.roi.setPen(pen)

    # --------------------------------------------------------------------------

    def updateSize(self):

        """
        Updates ROI dimensions
        """

        if self.updating != "analysis":
            self.updating = "roi"
            width = self.width_sbox.value()
            height = self.height_sbox.value()
            self.roi.setSize((width, height))
            self.updatePosition()
            self.updating = ""

    # --------------------------------------------------------------------------

    def updatePosition(self):

        """
        Updates ROI position
        """

        if self.updating != "analysis":
            self.updating = "roi"
            # Bottom lefthand corner of roi
            x_origin = self.x_sbox.value() - self.roi.size()[0] / 2
            y_origin = self.y_sbox.value() - self.roi.size()[1] / 2
            self.roi.setPos((x_origin, y_origin))
            self.updating = ""

    # --------------------------------------------------------------------------

    def updateAnalysis(self):

        """
        Updates textboxes that contain ROI size/position
        """

        if self.updating != "roi":
            self.updating = "analysis"
            self.x_sbox.setValue(self.roi.pos()[0] + self.roi.size()[0] / 2)
            self.y_sbox.setValue(self.roi.pos()[1] + self.roi.size()[1] / 2)
            self.width_sbox.setValue(self.roi.size()[0])
            self.height_sbox.setValue(self.roi.size()[1])
            self.updating = ""

    # --------------------------------------------------------------------------

    def center(self):

        """
        Centers ROI around image
        """

        rect = self.main_widget.data_widget.dataset_rect
        slice_direction = self.main_widget.data_widget.slice_direction

        if slice_direction == None or slice_direction == "X(H)":
            x_dir, y_dir = 2, 1
        elif slice_direction == "Y(K)":
            x_dir, y_dir = 2, 0
        else:
            x_dir, y_dir = 1, 0

        self.roi.setPos((rect[x_dir][0], rect[y_dir][0]))
        self.roi.setSize((rect[x_dir][-1] - rect[x_dir][0], rect[y_dir][-1] - rect[y_dir][0]))

    # --------------------------------------------------------------------------

    def plotAverageIntensity(self):

        """
        Creates list of average intensities from each slice of dataset
        """

        avg_intensity = []
        dataset = self.main_widget.data_widget.dataset
        color_dataset = self.main_widget.data_widget.color_dataset
        rect = self.main_widget.data_widget.dataset_rect
        slice_direction = self.main_widget.data_widget.slice_direction

        # Sets x, y, and timeline (z) directions
        if slice_direction == None or slice_direction == "X(H)":
            x_dir, y_dir, t_dir = 2, 1, 0
        elif slice_direction == "Y(K)":
            x_dir, y_dir, t_dir = 2, 0, 1
        else:
             x_dir, y_dir, t_dir = 1, 0, 2

        # Shape and axis limits
        x_shape, x_rect = dataset.shape[x_dir], rect[x_dir]
        y_shape, y_rect = dataset.shape[y_dir], rect[y_dir]
        t_shape, t_rect = dataset.shape[t_dir], rect[t_dir]

        # Pixel indicies for ROI
        x_min = int((self.roi.pos()[0] - x_rect[0]) * x_shape / (x_rect[-1] - x_rect[0]))
        x_max = int((self.roi.pos()[0] + self.roi.size()[0] - x_rect[0]) * x_shape / (x_rect[-1] - x_rect[0]))
        y_min = int((self.roi.pos()[1] - y_rect[0]) * y_shape / (y_rect[-1] - y_rect[0]))
        y_max = int((self.roi.pos()[1] + self.roi.size()[1] - y_rect[0]) * y_shape / (y_rect[-1] - y_rect[0]))

        if x_min >= 0 and x_max <= x_shape and y_min >= 0 and y_max <= y_shape:
            if slice_direction == None or slice_direction == "X(H)":
                self.data_roi = dataset[:, y_min:y_max, x_min:x_max]
                self.color_data_roi = color_dataset[:, y_min:y_max, x_min:x_max]
                for i in range(self.data_roi.shape[0]):
                    avg = np.mean(self.data_roi[i, :, :])
                    avg_intensity.append(avg)
                self.plot_widget.setLabel(axis="bottom", text="H")

            elif slice_direction == "Y(K)":
                self.data_roi = dataset[y_min:y_max, :, x_min:x_max]
                self.color_data_roi = color_dataset[y_min:y_max, :, x_min:x_max]
                for i in range(self.data_roi.shape[1]):
                    avg = np.mean(self.data_roi[:, i, :])
                    avg_intensity.append(avg)
                self.plot_widget.setLabel(axis="bottom", text="K")

            else:
                self.data_roi = dataset[y_min:y_max, x_min:x_max, :]
                self.color_data_roi = color_dataset[y_min:y_max, x_min:x_max, :]
                for i in range(self.data_roi.shape[2]):
                    avg = np.mean(self.data_roi[:, :, i])
                    avg_intensity.append(avg)
                self.plot_widget.setLabel(axis="bottom", text="L")

            t_values = np.linspace(t_rect[0], t_rect[-1], t_shape)
            self.plot_widget.setLabel(axis="left", text="Average Intensity")
        else:
            self.plot_widget.clear()

        try:
            self.plot_widget.plot(t_values, avg_intensity, clear=True)
        except Exception:
            self.plot_widget.clear()

        self.avg_intensity = avg_intensity

# ==============================================================================

class ROISubtractionWidget(QtGui.QWidget):

    """
    A widget that displays the difference between average intensities of two ROI's
    """

    def __init__ (self, parent):
        super(ROISubtractionWidget, self).__init__(parent)
        self.roi_analysis_widget = parent
        self.main_widget = self.roi_analysis_widget.main_widget

        # Widget Creation ------------------------------------------------------
        self.first_roi_lbl = QtGui.QLabel("First ROI")
        self.first_roi_cbox = QtGui.QComboBox()
        self.first_roi_cbox.addItems(["ROI 1", "ROI 2", "ROI 3", "ROI 4"])
        self.second_roi_lbl = QtGui.QLabel("Second ROI")
        self.second_roi_cbox = QtGui.QComboBox()
        self.second_roi_cbox.addItems(["ROI 1", "ROI 2", "ROI 3", "ROI 4"])

        self.plot_widget = pg.PlotWidget()

        # GroupBoxes -----------------------------------------------------------
        self.info_gbox = QtGui.QGroupBox()

        # Layout ---------------------------------------------------------------
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.info_layout = QtGui.QGridLayout()
        self.info_gbox.setLayout(self.info_layout)

        self.layout.addWidget(self.info_gbox, 0, 0)
        self.layout.addWidget(self.plot_widget, 0, 1)

        self.info_layout.addWidget(self.first_roi_lbl, 0, 0)
        self.info_layout.addWidget(self.first_roi_cbox, 0, 1)
        self.info_layout.addWidget(self.second_roi_lbl, 1, 0)
        self.info_layout.addWidget(self.second_roi_cbox, 1, 1)


        # Singals --------------------------------------------------------------
        self.first_roi_cbox.currentTextChanged.connect(self.plotData)
        self.second_roi_cbox.currentTextChanged.connect(self.plotData)

        # Representing each ROI with a string
        self.roi_dict = {
            "ROI 1" : self.roi_analysis_widget.roi_1,
            "ROI 2" : self.roi_analysis_widget.roi_2,
            "ROI 3" : self.roi_analysis_widget.roi_3,
            "ROI 4" : self.roi_analysis_widget.roi_4
        }

    # --------------------------------------------------------------------------

    def plotData(self):

        """
        Displays average intensity difference
        """

        dataset = self.main_widget.data_widget.dataset
        slice_direction = self.main_widget.data_widget.slice_direction
        rect = self.main_widget.data_widget.dataset_rect

        try:
            avg_intensity_1 = self.roi_dict[self.first_roi_cbox.currentText()].avg_intensity
            avg_intensity_2 = self.roi_dict[self.second_roi_cbox.currentText()].avg_intensity
            avg_intensity_diff = np.subtract(avg_intensity_1, avg_intensity_2)

            self.plot_widget.setLabel(axis="left", text="Average Intensity")

            if slice_direction == None or slice_direction == "X(H)":
                t_values = np.linspace(rect[0][0], rect[0][-1], dataset.shape[0])
                self.plot_widget.setLabel(axis="bottom", text="H")
            elif slice_direction == "Y(K)":
                t_values = np.linspace(rect[1][0], rect[1][-1], dataset.shape[1])
                self.plot_widget.setLabel(axis="bottom", text="K")
            else:
                t_values = np.linspace(rect[2][0], rect[2][-1], dataset.shape[2])
                self.plot_widget.setLabel(axis="bottom", text="L")

            self.plot_widget.plot(t_values, avg_intensity_diff, clear=True)

        except Exception:
            self.plot_widget.clear()

# ==============================================================================

class LineROIAnalysisWidget(pg.LayoutWidget):

    """
    Widget that houses four LineROIWidgets
    """

    def __init__ (self, parent):
        super(LineROIAnalysisWidget, self).__init__(parent)
        self.main_widget = parent

        self.setEnabled(False)
        self.line_roi_tabs = QtGui.QTabWidget()

        self.line_roi_1 = LineROIWidget(self)
        self.line_roi_2 = LineROIWidget(self)
        self.line_roi_3 = LineROIWidget(self)
        self.line_roi_4 = LineROIWidget(self)

        self.line_roi_tabs.addTab(self.line_roi_1, "ROI 1")
        self.line_roi_tabs.addTab(self.line_roi_2, "ROI 2")
        self.line_roi_tabs.addTab(self.line_roi_3, "ROI 3")
        self.line_roi_tabs.addTab(self.line_roi_4, "ROI 4")

        self.addWidget(self.line_roi_tabs)

# ==============================================================================

class LineROIWidget(QtGui.QWidget):

    """
    - Creates a line segment ROI
    - Allows user to take arbitrary slices/line cuts of dataset
    """

    def __init__ (self, parent):
        super(LineROIWidget, self).__init__(parent)
        self.line_roi_analysis_widget = parent
        self.main_widget = self.line_roi_analysis_widget.main_widget

        self.scene_point = None

        # Widget Creation ------------------------------------------------------
        # Slice info
        self.visible_chkbox = QtGui.QCheckBox("Visible")
        self.color_btn = pg.ColorButton()
        self.center_btn = QtGui.QPushButton("Center")
        self.mouse_intensity_lbl = QtGui.QLabel("Intensity:")
        self.mouse_intensity_txtbox = QtGui.QLineEdit()
        self.mouse_intensity_txtbox.setReadOnly(True)
        self.mouse_h_lbl = QtGui.QLabel("H:")
        self.mouse_h_txtbox = QtGui.QLineEdit()
        self.mouse_h_txtbox.setReadOnly(True)
        self.mouse_k_lbl = QtGui.QLabel("K:")
        self.mouse_k_txtbox = QtGui.QLineEdit()
        self.mouse_k_txtbox.setReadOnly(True)
        self.mouse_l_lbl = QtGui.QLabel("L:")
        self.mouse_l_txtbox = QtGui.QLineEdit()
        self.mouse_l_txtbox.setReadOnly(True)
        self.export_slice_btn = QtGui.QPushButton("Export Slice")

        # Line Cut info
        self.line_cut_visible_chkbox = QtGui.QCheckBox("Visible")
        self.line_cut_color_btn = pg.ColorButton()
        self.line_cut_center_btn = QtGui.QPushButton("Center")
        self.line_cut_h_lbl = QtGui.QLabel("H:")
        self.line_cut_h_txtbox = QtGui.QLineEdit()
        self.line_cut_h_txtbox.setReadOnly(True)
        self.line_cut_k_lbl = QtGui.QLabel("K:")
        self.line_cut_k_txtbox = QtGui.QLineEdit()
        self.line_cut_k_txtbox.setReadOnly(True)
        self.line_cut_l_lbl = QtGui.QLabel("L:")
        self.line_cut_l_txtbox = QtGui.QLineEdit()
        self.line_cut_l_txtbox.setReadOnly(True)
        self.export_line_cut_btn = QtGui.QPushButton("Export Line Cut")

        # Slice ROI
        self.roi = pg.LineSegmentROI([[0, 0], [0.5, 0.5]])
        self.roi.hide()
        self.pen = pg.mkPen(width=3)
        self.roi.setPen(self.pen)
        self.main_widget.data_widget.addItem(self.roi)
        self.handle_1 = self.roi.getHandles()[0]
        self.handle_2 = self.roi.getHandles()[1]

        # Slice Image View
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.view.hideAxis('left')
        self.view_box = self.image_view.view.getViewBox()
        self.view_box.setAspectLocked(False)

        # Line Cut ROI
        self.line_cut_roi = pg.LineSegmentROI([[0, 0], [0.5, 0.5]])
        self.image_view.addItem(self.line_cut_roi)
        self.line_cut_roi.hide()
        self.line_cut_pen = pg.mkPen(width=3)
        self.line_cut_roi.setPen(self.line_cut_pen)
        self.line_cut_handle_1 = self.line_cut_roi.getHandles()[0]
        self.line_cut_handle_2 = self.line_cut_roi.getHandles()[1]

        # Intensity Plots
        self.slice_plot_widget = pg.PlotWidget(title="Slice")
        self.line_cut_plot_widget = pg.PlotWidget(title="Line Cut")
        self.line_cut_plot_widget.hideAxis('bottom')

        # GroupBoxes -----------------------------------------------------------
        self.info_gbox = QtGui.QGroupBox()
        self.info_gbox.setMinimumSize(100, 600)
        self.slice_info_gbox = QtGui.QGroupBox("Slice")
        self.line_cut_info_gbox = QtGui.QGroupBox("Line Cut")

        # ScrollArea -----------------------------------------------------------
        self.info_scroll_area = QtGui.QScrollArea()
        self.info_scroll_area.setWidget(self.info_gbox)
        self.info_scroll_area.setWidgetResizable(True)

        # Layouts --------------------------------------------------------------
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.info_layout = QtGui.QGridLayout()
        self.info_gbox.setLayout(self.info_layout)
        self.slice_info_layout = QtGui.QGridLayout()
        self.slice_info_gbox.setLayout(self.slice_info_layout)
        self.line_cut_info_layout = QtGui.QGridLayout()
        self.line_cut_info_gbox.setLayout(self.line_cut_info_layout)

        self.layout.addWidget(self.info_scroll_area, 0, 0)
        self.layout.addWidget(self.image_view, 0, 1)
        self.layout.addWidget(self.slice_plot_widget, 0, 2)
        self.layout.addWidget(self.line_cut_plot_widget, 0, 3)
        self.layout.setColumnStretch(0,2)
        self.layout.setColumnStretch(1,3)
        self.layout.setColumnStretch(2,3)
        self.layout.setColumnStretch(3,3)

        self.info_layout.addWidget(self.slice_info_gbox, 0, 0)
        self.info_layout.addWidget(self.line_cut_info_gbox, 1, 0)

        # Slice Info
        self.slice_info_layout.addWidget(self.visible_chkbox, 0, 0, 1, 2)
        self.slice_info_layout.addWidget(self.color_btn, 0, 2, 1, 2)
        self.slice_info_layout.addWidget(self.center_btn, 1, 0, 1, 4)
        self.slice_info_layout.addWidget(self.mouse_intensity_lbl, 2, 0, 1, 2)
        self.slice_info_layout.addWidget(self.mouse_intensity_txtbox, 2, 2, 1, 2)
        self.slice_info_layout.addWidget(self.mouse_h_lbl, 3, 0)
        self.slice_info_layout.addWidget(self.mouse_h_txtbox, 3, 1, 1, 3)
        self.slice_info_layout.addWidget(self.mouse_k_lbl, 4, 0)
        self.slice_info_layout.addWidget(self.mouse_k_txtbox, 4, 1, 1, 3)
        self.slice_info_layout.addWidget(self.mouse_l_lbl, 5, 0)
        self.slice_info_layout.addWidget(self.mouse_l_txtbox, 5, 1, 1, 3)
        self.slice_info_layout.addWidget(self.export_slice_btn, 6, 0, 1, 4)

        # Line Cut Info
        self.line_cut_info_layout.addWidget(self.line_cut_visible_chkbox, 0, 0, 1, 2)
        self.line_cut_info_layout.addWidget(self.line_cut_color_btn, 0, 2, 1, 3)
        self.line_cut_info_layout.addWidget(self.line_cut_center_btn, 1, 0, 1, 5)
        self.line_cut_info_layout.addWidget(self.line_cut_h_lbl, 2, 0)
        self.line_cut_info_layout.addWidget(self.line_cut_h_txtbox, 2, 1, 1, 4)
        self.line_cut_info_layout.addWidget(self.line_cut_k_lbl, 3, 0)
        self.line_cut_info_layout.addWidget(self.line_cut_k_txtbox, 3, 1, 1, 4)
        self.line_cut_info_layout.addWidget(self.line_cut_l_lbl, 4, 0)
        self.line_cut_info_layout.addWidget(self.line_cut_l_txtbox, 4, 1, 1, 4)
        self.line_cut_info_layout.addWidget(self.export_line_cut_btn, 5, 0, 1, 5)

        # Signals --------------------------------------------------------------
        self.roi.sigRegionChanged.connect(self.update)
        self.visible_chkbox.clicked.connect(self.toggleVisibility)
        self.color_btn.sigColorChanged.connect(self.changeColor)
        self.line_cut_roi.sigRegionChanged.connect(self.updateLineCut)
        self.line_cut_visible_chkbox.clicked.connect(self.toggleLineCutVisibility)
        self.line_cut_color_btn.sigColorChanged.connect(self.changeLineCutColor)
        self.center_btn.clicked.connect(self.center)
        self.line_cut_center_btn.clicked.connect(self.centerLineCut)
        self.view_box.scene().sigMouseMoved.connect(self.updateMouseInfo)
        self.export_slice_btn.clicked.connect(self.exportSlice)
        #self.export_line_cut_btn.clicked.connect(self.exportLineCut)

        self.updating = ""

    # --------------------------------------------------------------------------

    def update(self):

        """
        Updates slice image/plot to reflect current ROI position
        """

        dataset = self.main_widget.data_widget.dataset
        rect = self.main_widget.data_widget.dataset_rect
        image_item = self.main_widget.data_widget.imageItem
        axes = self.main_widget.data_widget.axes
        slice_direction = self.main_widget.data_widget.slice_direction
        self.slice_coords = []

        # Sets x, y, and timeline (z) directions
        if slice_direction == None or slice_direction == "X(H)":
            x_dir, y_dir, t_dir = 2, 1, 0
        elif slice_direction == "Y(K)":
            x_dir, y_dir, t_dir = 2, 0, 1
        else:
             x_dir, y_dir, t_dir = 1, 0, 2

        try:
            self.slice, self.slice_coords = self.roi.getArrayRegion(dataset, image_item, \
                axes=(axes.get("x"), axes.get("y")), returnMappedCoords=True)
            self.slice_coords = self.slice_coords.astype(int)

            colormap_max = np.amax(dataset)
            norm = colors.LogNorm(vmax=colormap_max)
            norm_slice = norm(self.slice)
            color_slice = plt.cm.jet(norm_slice)

            self.t_values = np.linspace(rect[t_dir][0], rect[t_dir][-1], dataset.shape[t_dir])
            self.image_x_coords = np.linspace(rect[x_dir][0], rect[x_dir][-1], dataset.shape[x_dir])
            self.image_y_coords = np.linspace(rect[y_dir][0], rect[y_dir][-1], dataset.shape[y_dir])
            pos = (rect[t_dir][0], 0)
            scale = ((rect[t_dir][-1] - rect[t_dir][0]) / dataset.shape[t_dir], 1)

            if slice_direction == None or slice_direction == "X(H)":
                intensities = np.mean(self.slice, axis=1)
                self.image_view.view.setLabel(axis="bottom", text="H")
                self.slice_plot_widget.setLabel(axis="bottom", text="H")

            elif slice_direction == "Y(K)":
                color_slice = np.swapaxes(color_slice, 0, 1)
                intensities = np.mean(self.slice, axis=0)
                self.image_view.view.setLabel(axis="bottom", text="K")
                self.slice_plot_widget.setLabel(axis="bottom", text="K")

            else:
                color_slice = np.swapaxes(color_slice, 0, 1)
                intensities = np.mean(self.slice, axis=0)
                self.image_view.view.setLabel(axis="bottom", text="L")
                self.slice_plot_widget.setLabel(axis="bottom", text="L")

            self.slice_plot_widget.setLabel(axis="left", text="Average Intensity")
            self.line_cut_plot_widget.setLabel(axis="left", text="Intensity")

            self.image_view.setImage(color_slice, pos=pos, scale=scale)
            self.slice_plot_widget.plot(self.t_values, intensities, clear=True)
            self.updateLineCut()

        except Exception:
            self.image_view.clear()
            self.slice_plot_widget.clear()
            self.line_cut_plot_widget.clear()

    # --------------------------------------------------------------------------

    def updateLineCut(self):

        """
        Updates line cut plot
        """

        rect = self.main_widget.data_widget.dataset_rect
        slice_direction = self.main_widget.data_widget.slice_direction

        try:
            if slice_direction == None or slice_direction == "X(H)":
                self.line_cut, self.line_cut_coords = self.line_cut_roi.getArrayRegion(self.slice, \
                        self.image_view.getImageItem(), returnMappedCoords=True)
                self.line_cut_coords = self.line_cut_coords.astype(int)
                h_1 = round(self.t_values[self.line_cut_coords[0][0]], 5)
                h_2 = round(self.t_values[self.line_cut_coords[0][-1]], 5)
                k_1 = round(self.image_y_coords[self.slice_coords[1][self.line_cut_coords[1][0]]], 5)
                k_2 = round(self.image_y_coords[self.slice_coords[1][self.line_cut_coords[1][-1]]], 5)
                l_1 = round(self.image_x_coords[self.slice_coords[0][self.line_cut_coords[1][0]]], 5)
                l_2 = round(self.image_x_coords[self.slice_coords[0][self.line_cut_coords[1][-1]]], 5)

            elif slice_direction == "Y(K)":
                self.line_cut, self.line_cut_coords = self.line_cut_roi.getArrayRegion(np.swapaxes(self.slice, 0, 1), \
                        self.image_view.getImageItem(), returnMappedCoords=True)
                self.line_cut_coords = self.line_cut_coords.astype(int)
                h_1 = round(self.image_y_coords[self.slice_coords[1][self.line_cut_coords[1][0]]], 5)
                h_2 = round(self.image_y_coords[self.slice_coords[1][self.line_cut_coords[1][-1]]], 5)
                k_1 = round(self.t_values[self.line_cut_coords[0][0]], 5)
                k_2 = round(self.t_values[self.line_cut_coords[0][-1]], 5)
                l_1 = round(self.image_x_coords[self.slice_coords[0][self.line_cut_coords[1][0]]], 5)
                l_2 = round(self.image_x_coords[self.slice_coords[0][self.line_cut_coords[1][-1]]], 5)

            else:
                self.line_cut, self.line_cut_coords = self.line_cut_roi.getArrayRegion(np.swapaxes(self.slice, 0, 1), \
                        self.image_view.getImageItem(), returnMappedCoords=True)
                self.line_cut_coords = self.line_cut_coords.astype(int)
                h_1 = round(self.image_y_coords[self.slice_coords[1][self.line_cut_coords[1][0]]], 5)
                h_2 = round(self.image_y_coords[self.slice_coords[1][self.line_cut_coords[1][-1]]], 5)
                k_1 = round(self.image_x_coords[self.slice_coords[0][self.line_cut_coords[1][0]]], 5)
                k_2 = round(self.image_x_coords[self.slice_coords[0][self.line_cut_coords[1][-1]]], 5)
                l_1 = round(self.t_values[self.line_cut_coords[0][0]], 5)
                l_2 = round(self.t_values[self.line_cut_coords[0][-1]], 5)

            self.line_cut_plot_widget.plot(self.line_cut, clear=True)
            h_interval = f"({h_1}, {h_2})"
            k_interval = f"({k_1}, {k_2})"
            l_interval = f"({l_1}, {l_2})"
            self.line_cut_h_txtbox.setText(h_interval)
            self.line_cut_k_txtbox.setText(k_interval)
            self.line_cut_l_txtbox.setText(l_interval)

        except Exception:
            self.line_cut_plot_widget.clear()

    # --------------------------------------------------------------------------

    def exportSlice(self):
        h_values = self.main_widget.data_selection_widget.h_values
        k_values = self.main_widget.data_selection_widget.k_values
        l_values = self.main_widget.data_selection_widget.l_values
        slice_direction = self.main_widget.data_widget.slice_direction

        #try:
        if slice_direction == None or slice_direction == "X(H)":
            h = self.t_values
            k = np.take(k_values, np.ndarray.tolist(self.slice_coords[1]))
            l = np.take(l_values, np.ndarray.tolist(self.slice_coords[0]))

            h = np.reshape(np.repeat(h, k.shape[0]), (h.shape[0], k.shape[0]))
            k = np.reshape(np.tile(k, h.shape[0]), (h.shape[0], k.shape[0]))
            l = np.reshape(np.tile(l, h.shape[0]), (h.shape[0], l.shape[0]))

        elif slice_direction == "Y(K)":
            h = np.take(h_values, np.ndarray.tolist(self.slice_coords[1]))
            k = self.t_values
            l = np.take(l_values, np.ndarray.tolist(self.slice_coords[0]))

            k = np.reshape(np.repeat(k, h.shape[0]), (k.shape[0], h.shape[0]))
            h = np.reshape(np.tile(h, k.shape[0]), (k.shape[0], h.shape[0]))
            l = np.reshape(np.tile(l, k.shape[0]), (k.shape[0], l.shape[0]))

        else:
            h = np.take(h_values, np.ndarray.tolist(self.slice_coords[1]))
            k = np.take(k_values, np.ndarray.tolist(self.slice_coords[0]))
            l = self.t_values

            l = np.reshape(np.repeat(l, h.shape[0]), (l.shape[0], h.shape[0]))
            h = np.reshape(np.tile(h, l.shape[0]), (l.shape[0], h.shape[0]))
            k = np.reshape(np.tile(k, l.shape[0]), (l.shape[0], k.shape[0]))

        hkl = np.dstack((h, k, l))

        file_path = QtGui.QFileDialog.getSaveFileName(self, "", "", "(*.hdf)")[0]
        file = h5py.File(file_path, 'a')
        file.create_group("data")
        file["data"].create_dataset("HKL", data=hkl)
        file["data"].create_dataset("Intensity", data=self.slice)
        """except:
            msg_box = QtGui.QMessageBox()
            msg_box.setWindowTitle("Error")
            msg_box.setText("Could Not Create File")
            msg_box.exec_()"""
    # --------------------------------------------------------------------------

    def exportLineCut(self):
        ...

    # --------------------------------------------------------------------------

    def toggleVisibility(self, state):

        """
        Changes visibility of slice ROI
        """

        if state:
            self.roi.show()
        else:
            self.roi.hide()

    # --------------------------------------------------------------------------

    def toggleLineCutVisibility(self, state):

        """
        Changes visibility of line cut ROI
        """

        if state:
            self.line_cut_roi.show()
        else:
            self.line_cut_roi.hide()

    # --------------------------------------------------------------------------

    def changeColor(self):
        color = self.color_btn.color()
        pen = pg.mkPen(color, width=3)
        self.roi.setPen(pen)

    # --------------------------------------------------------------------------

    def changeLineCutColor(self):
        color = self.line_cut_color_btn.color()
        pen = pg.mkPen(color, width=3)
        self.line_cut_roi.setPen(pen)

    # --------------------------------------------------------------------------

    def center(self):

        """
        Centers slice ROI diagonally across image
        """

        rect = self.main_widget.data_widget.dataset_rect
        slice_direction = self.main_widget.data_widget.slice_direction

        if slice_direction == None or slice_direction == "X(H)":
            x1, y1 = round(rect[2][0], 6), round(rect[1][0], 6)
            x2, y2 = round(rect[2][-1], 6), round(rect[1][-1], 6)
        elif slice_direction == "Y(K)":
            x1, y1 = round(rect[2][0], 6), round(rect[0][0], 6)
            x2, y2 = round(rect[2][-1], 6), round(rect[0][-1], 6)
        else:
            x1, y1 = round(rect[1][0], 6), round(rect[0][0], 6)
            x2, y2 = round(rect[1][-1], 6), round(rect[0][-1], 6)

        self.roi.movePoint(self.handle_1, (x1, y1))
        self.roi.movePoint(self.handle_2, (x2, y2))

    # --------------------------------------------------------------------------

    def centerLineCut(self):

        """
        Centers line cut ROI diagonally across slice image
        """

        rect = self.main_widget.data_widget.dataset_rect
        slice_direction = self.main_widget.data_widget.slice_direction

        if slice_direction == None or slice_direction == "X(H)":
            x1, y1 = round(rect[0][0], 6), 0
            x2, y2 = round(rect[0][-1], 6), self.slice.shape[1]
        elif slice_direction == "Y(K)":
            x1, y1 = round(rect[1][0], 6), 0
            x2, y2 = round(rect[1][-1], 6), self.slice.shape[1]
        else:
            x1, y1 = round(rect[2][0], 6), 0
            x2, y2 = round(rect[2][-1], 6), self.slice.shape[1]

        self.line_cut_roi.movePoint(self.line_cut_handle_1, (x1, y1))
        self.line_cut_roi.movePoint(self.line_cut_handle_2, (x2, y2))

    # --------------------------------------------------------------------------

    def updateMouseInfo(self, scene_point=None):

        """
        Updates mouse position on slice and converts mouse position to HKL position
        """

        dataset = self.main_widget.data_widget.dataset
        rect = self.main_widget.data_widget.dataset_rect
        slice_direction = self.main_widget.data_widget.slice_direction

        if scene_point != None:
            self.scene_point = scene_point
        if self.scene_point != None:
            self.view_point = self.view_box.mapSceneToView(self.scene_point)

            # x and y values of mouse
            x, y = self.view_point.x(), self.view_point.y()
        else:
            return

        try:
            if slice_direction == None or slice_direction == "X(H)":
                h_index = int(dataset.shape[0] * (x - rect[0][0]) / (rect[0][-1] - rect[0][0]))
                k_index = self.slice_coords[1][int(y)]
                l_index = self.slice_coords[0][int(y) + 1]
                self.mouse_h_txtbox.setText(str(round(x, 5)))
                self.mouse_k_txtbox.setText(str(round(self.image_y_coords[k_index], 5)))
                self.mouse_l_txtbox.setText(str(round(self.image_x_coords[l_index], 5)))

            elif slice_direction == "Y(K)":
                h_index = self.slice_coords[1][int(y)]
                k_index = int(dataset.shape[1] * (x - rect[1][0]) / (rect[1][-1] - rect[1][0]))
                l_index = self.slice_coords[0][int(y) + 1]
                self.mouse_h_txtbox.setText(str(round(self.image_y_coords[h_index], 5)))
                self.mouse_k_txtbox.setText(str(round(x, 5)))
                self.mouse_l_txtbox.setText(str(round(self.image_x_coords[l_index], 5)))

            else:
                h_index = self.slice_coords[1][int(y)]
                k_index = self.slice_coords[0][int(y) + 1]
                l_index = int(dataset.shape[2] * (x - rect[2][0]) / (rect[2][-1] - rect[2][0]))
                self.mouse_h_txtbox.setText(str(round(self.image_y_coords[h_index], 5)))
                self.mouse_k_txtbox.setText(str(round(self.image_x_coords[k_index], 5)))
                self.mouse_l_txtbox.setText(str(round(x, 5)))

            intensity = int(dataset[h_index][k_index][l_index])
            self.mouse_intensity_txtbox.setText(str(intensity))
        except Exception:
            self.mouse_intensity_txtbox.setText("")

# ==============================================================================

class ConversionParametersDialog(QtGui.QDialog):

    """
    Houses dialog where user selects pixel interpolation values and config files for conversion
    """

    def __init__ (self):
        super().__init__()

        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.detector_config_name = ""
        self.instrument_config_name = ""
        self.pixel_count_nx = 0
        self.pixel_count_ny = 0
        self.pixel_count_nz = 0

        # Widget Creation ------------------------------------------------------
        self.detector_lbl = QtGui.QLabel("Det. Config:")
        self.detector_txtbox = QtGui.QLineEdit()
        self.detector_txtbox.setReadOnly(True)
        self.detector_btn = QtGui.QPushButton("Browse")
        self.instrument_lbl = QtGui.QLabel("Instr. Config:")
        self.instrument_txtbox = QtGui.QLineEdit()
        self.instrument_txtbox.setReadOnly(True)
        self.instrument_btn = QtGui.QPushButton("Browse")
        self.pixel_count_lbl = QtGui.QLabel("Pixel Count:")
        self.pixel_count_nx_sbox = QtGui.QSpinBox(maximum=1000, minimum=1)
        self.pixel_count_nx_sbox.setValue(200)
        self.pixel_count_ny_sbox = QtGui.QSpinBox(maximum=1000, minimum=1)
        self.pixel_count_ny_sbox.setValue(200)
        self.pixel_count_nz_sbox = QtGui.QSpinBox(maximum=1000, minimum=1)
        self.pixel_count_nz_sbox.setValue(200)
        self.dialog_btnbox = QtGui.QDialogButtonBox()
        self.dialog_btnbox.addButton("OK", QtGui.QDialogButtonBox.AcceptRole)

        # Layout ---------------------------------------------------------------
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.detector_lbl, 1, 0)
        self.layout.addWidget(self.detector_txtbox, 1, 1, 1, 3)
        self.layout.addWidget(self.detector_btn, 1, 4)
        self.layout.addWidget(self.instrument_lbl, 2, 0)
        self.layout.addWidget(self.instrument_txtbox, 2, 1, 1, 3)
        self.layout.addWidget(self.instrument_btn, 2, 4)
        self.layout.addWidget(self.pixel_count_lbl, 3, 0)
        self.layout.addWidget(self.pixel_count_nx_sbox, 3, 1)
        self.layout.addWidget(self.pixel_count_ny_sbox, 3, 2)
        self.layout.addWidget(self.pixel_count_nz_sbox, 3, 3)
        self.layout.addWidget(self.dialog_btnbox, 4, 3, 1, 2)

        # Signals --------------------------------------------------------------

        self.detector_btn.clicked.connect(self.selectDetectorConfigFile)
        self.instrument_btn.clicked.connect(self.selectInstrumentConfigFile)
        self.dialog_btnbox.accepted.connect(self.accept)

    # --------------------------------------------------------------------------

    def selectDetectorConfigFile(self):

        """
        Allows user to select a detector configuration .xml file.
        """

        detector = QtGui.QFileDialog.getOpenFileName(self, "", "", "xml Files (*.xml)")
        self.detector_txtbox.setText(detector[0])

    # --------------------------------------------------------------------------

    def selectInstrumentConfigFile(self):

        """
        Allows user to select an instrument configuration .xml file.
        """

        instrument = QtGui.QFileDialog.getOpenFileName(self, "", "", "xml Files (*.xml)")
        self.instrument_txtbox.setText(instrument[0])

    # --------------------------------------------------------------------------

    def accept(self):

        """
        Sets class variables to values in dialog and closes the dialog window.
        """

        self.detector_config_name = self.detector_txtbox.text()
        self.instrument_config_name = self.instrument_txtbox.text()
        self.pixel_count_nx = self.pixel_count_nx_sbox.value()
        self.pixel_count_ny = self.pixel_count_ny_sbox.value()
        self.pixel_count_nz = self.pixel_count_nz_sbox.value()
        self.close()

# ==============================================================================

class VTICreationDialog(QtGui.QDialog):

    def __init__ (self):
        super().__init__()

        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.detector_config_name = ""
        self.instrument_config_name = ""
        self.pixel_count_nx = 0
        self.pixel_count_ny = 0
        self.pixel_count_nz = 0

        # Widget Creation ------------------------------------------------------
        self.project_lbl = QtGui.QLabel("Project Directory:")
        self.project_txtbox = QtGui.QLineEdit()
        self.project_txtbox.setReadOnly(True)
        self.project_btn = QtGui.QPushButton("Browse")
        self.data_source_lbl = QtGui.QLabel("Data Source (SPEC):")
        self.data_source_txtbox = QtGui.QLineEdit()
        self.data_source_txtbox.setReadOnly(True)
        self.data_source_btn = QtGui.QPushButton("Browse")
        self.selected_scan_lbl = QtGui.QLabel("Selected Scan:")
        self.selected_scan_cbox = QtGui.QComboBox()
        self.pixel_count_lbl = QtGui.QLabel("Interpolation (HKL):")
        self.h_count_sbox = QtGui.QSpinBox(maximum=1000, minimum=1)
        self.h_count_sbox.setValue(200)
        self.k_count_sbox = QtGui.QSpinBox(maximum=1000, minimum=1)
        self.k_count_sbox.setValue(200)
        self.l_count_sbox = QtGui.QSpinBox(maximum=1000, minimum=1)
        self.l_count_sbox.setValue(200)
        self.detector_lbl = QtGui.QLabel("Det. Config:")
        self.detector_txtbox = QtGui.QLineEdit()
        self.detector_txtbox.setReadOnly(True)
        self.detector_btn = QtGui.QPushButton("Browse")
        self.instrument_lbl = QtGui.QLabel("Instr. Config:")
        self.instrument_txtbox = QtGui.QLineEdit()
        self.instrument_txtbox.setReadOnly(True)
        self.instrument_btn = QtGui.QPushButton("Browse")
        self.dialog_btnbox = QtGui.QDialogButtonBox()
        self.dialog_btnbox.addButton("Create", QtGui.QDialogButtonBox.AcceptRole)

        # Layout ---------------------------------------------------------------
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.project_lbl, 0, 0, 1, 3)
        self.layout.addWidget(self.project_txtbox, 0, 3, 1, 3)
        self.layout.addWidget(self.project_btn, 0, 6, 1, 3)
        self.layout.addWidget(self.data_source_lbl, 1, 0, 1, 3)
        self.layout.addWidget(self.data_source_txtbox, 1, 3, 1, 3)
        self.layout.addWidget(self.data_source_btn, 1, 6, 1, 3)
        self.layout.addWidget(self.selected_scan_lbl, 2, 0, 1, 4)
        self.layout.addWidget(self.selected_scan_cbox, 2, 4, 1, 5)
        self.layout.addWidget(self.pixel_count_lbl, 3, 0, 1, 3)
        self.layout.addWidget(self.h_count_sbox, 3, 3, 1, 1)
        self.layout.addWidget(self.k_count_sbox, 3, 5, 1, 1)
        self.layout.addWidget(self.l_count_sbox, 3, 7, 1, 1)
        self.layout.addWidget(self.detector_lbl, 4, 0, 1, 3)
        self.layout.addWidget(self.detector_txtbox, 4, 3, 1, 3)
        self.layout.addWidget(self.detector_btn, 4, 6, 1, 3)
        self.layout.addWidget(self.instrument_lbl, 5, 0, 1, 3)
        self.layout.addWidget(self.instrument_txtbox, 5, 3, 1, 3)
        self.layout.addWidget(self.instrument_btn, 5, 6, 1, 3)

        self.layout.addWidget(self.dialog_btnbox, 6, 8)
        self.layout.setColumnStretch(0,1)
        self.layout.setColumnStretch(1,1)
        self.layout.setColumnStretch(2,1)
        self.layout.setColumnStretch(3,1)
        self.layout.setColumnStretch(4,1)
        self.layout.setColumnStretch(5,1)
        self.layout.setColumnStretch(6,1)
        self.layout.setColumnStretch(7,1)
        self.layout.setColumnStretch(8,1)


        # Signals --------------------------------------------------------------

        self.project_btn.clicked.connect(self.selectProjectDirectory)
        self.data_source_btn.clicked.connect(self.selectDataSource)
        self.detector_btn.clicked.connect(self.selectDetectorConfigFile)
        self.instrument_btn.clicked.connect(self.selectInstrumentConfigFile)
        self.dialog_btnbox.accepted.connect(self.accept)

    # --------------------------------------------------------------------------

    def selectProjectDirectory(self):

        """
        - Opens directory dialog
        - Sets project directory
        """

        # Selecting a Project Directory ----------------------------------------
        project_path = QtGui.QFileDialog.getExistingDirectory(self,
            "Open Project Directory")

        # Adding SPEC Files to a ListBox ---------------------------------------
        if project_path != "":
            self.spec_files = []

            # Adds SPEC file basenames in directory to list
            for file in os.listdir(project_path):
                if file.endswith(".spec"):
                    self.project_path = project_path
                    self.project_txtbox.setText(self.project_path)

    # --------------------------------------------------------------------------

    def selectDataSource(self):

        """
        Allows user to select a detector configuration .xml file.
        """

        self.data_source_path = QtGui.QFileDialog.getOpenFileName(self, "", "", "SPEC Files (*.spec)")[0]
        self.data_source_txtbox.setText(self.data_source_path)

        self.scan_list = []
        file = open(self.data_source_path,"r")
        for line in file:
            if line.startswith("#S"):
                scan = line.split()[1]
                if len(scan) < 3:
                    scan = ("00" + scan)[-3:]
                self.scan_list.append(scan)

        self.selected_scan_cbox.clear()
        self.selected_scan_cbox.addItems(self.scan_list)

    # --------------------------------------------------------------------------

    def selectDetectorConfigFile(self):

        """
        Allows user to select a detector configuration .xml file.
        """

        self.detector_path = QtGui.QFileDialog.getOpenFileName(self, "", "", "xml Files (*.xml)")[0]
        self.detector_txtbox.setText(self.detector_path)

    # --------------------------------------------------------------------------

    def selectInstrumentConfigFile(self):

        """
        Allows user to select an instrument configuration .xml file.
        """

        self.instrument_path = QtGui.QFileDialog.getOpenFileName(self, "", "", "xml Files (*.xml)")[0]
        self.instrument_txtbox.setText(self.instrument_path)

    # --------------------------------------------------------------------------

    def accept(self):

        scan = self.selected_scan_cbox.currentText()
        h_count = self.h_count_sbox.value()
        k_count = self.k_count_sbox.value()
        l_count = self.l_count_sbox.value()

        file_name = QtGui.QFileDialog.getSaveFileName(self,"", "", "VTI Files (*.vti)")[0]

        ConversionLogic.createVTIFile(self.project_path, self.data_source_path,
            self.detector_path, self.instrument_path, scan, h_count, k_count, l_count, file_name)

        self.close()

# ==============================================================================

class ConversionLogic():

    def createVTIFile(project_dir, spec_file, detector_config_name, instrument_config_name,
        scan, nx, ny, nz, file_name=None):

        # Necessary subfunctions for function to run smoothly
        # See rsMap3D source code
        def updateDataSourceProgress(value1, value2):
            print("DataSource Progress %s/%s" % (value1, value2))

        def updateMapperProgress(value1):
            print("Mapper Progress %s" % (value1))

        d_reader = detReader(detector_config_name)
        detector_name = "Pilatus"
        detector = d_reader.getDetectorById(detector_name)
        n_pixels = d_reader.getNpixels(detector)
        roi = [1, n_pixels[0], 1, n_pixels[1]]
        bin = [1,1]

        spec_name, spec_ext = os.path.splitext(os.path.basename(spec_file))
        # Set destination file for gridmapper
        if file_name == None:
            output_file_name = os.path.join(project_dir, spec_name + "_" + scan + ".vti")
        else:
            output_file_name = file_name

        if os.path.exists(output_file_name):
            return output_file_name

        app_config = RSMap3DConfigParser()
        max_image_memory = app_config.getMaxImageMemory()

        scan_dir = os.path.join(project_dir, "images", spec_name, f"S{scan}")
        
        for file in os.listdir(scan_dir):
            old_point_number = file.split("_")[-1]
            new_point_number = f"0000{old_point_number}"[-9:]
            os.rename(os.path.join(scan_dir, file), 
                os.path.join(scan_dir, file.replace(old_point_number, new_point_number)))

        scan_range = srange(scan).list()
        data_source = Sector33SpecDataSource(project_dir, spec_name, spec_ext,
            instrument_config_name, detector_config_name, roi=roi, pixelsToAverage=bin,
            scanList=scan_range, appConfig=app_config)
        data_source.setCurrentDetector(detector_name)
        data_source.setProgressUpdater(updateDataSourceProgress)
        data_source.loadSource(mapHKL=True)
        data_source.setRangeBounds(data_source.getOverallRanges())
        image_tbu = data_source.getImageToBeUsed()
        image_size = np.prod(data_source.getDetectorDimensions())

        grid_mapper = QGridMapper(data_source, output_file_name, nx=nx, ny=ny, nz=nz,
            outputType=BINARY_OUTPUT, transform=UnityTransform3D(),
            gridWriter=VTIGridWriter(), appConfig=app_config)
        grid_mapper.setProgressUpdater(updateMapperProgress)
        grid_mapper.doMap()

        return output_file_name

    # --------------------------------------------------------------------------

    def loadData(vti_file):

        """
        Converts information from .vti file into an array in HKL.
        """

        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(vti_file)
        reader.Update()

        data = reader.GetOutput()
        dim = data.GetDimensions()

        vec = list(dim)

        vec = [i for i in dim]
        vec.reverse()

        u = npSup.vtk_to_numpy(data.GetPointData().GetArray('Scalars_'))

        max_value = np.nanmax(u)
        min_value = np.nanmin(u)

        u = u.reshape(vec)

        # Swaps H and L
        ctrdata = np.swapaxes(u, 0, 2)

        origin = data.GetOrigin()
        spacing = data.GetSpacing()
        extent = data.GetExtent()

        x = []
        y = []
        z = []

        for point in range(extent[0], extent[1] + 1):
            x.append(origin[0] + point * spacing[0])
        for point in range(extent[2], extent[3] + 1):
            y.append(origin[1] + point * spacing[1])
        for point in range(extent[4], extent[5] + 1):
            z.append(origin[2] + point * spacing[2])

        axes = [x, y, z]

        return axes, ctrdata

# ==============================================================================
