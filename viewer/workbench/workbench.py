#!/usr/bin/env python3
"""
Workbench Window
A PyQt-based application for analyzing HDF5 data with 2D visualization capabilities.
Inherits from BaseWindow for consistent functionality across the application.
"""

import sys
import os
from pathlib import Path

try:
    # workbench.py lives at <project_root>/viewer/workbench/workbench.py
    # parents[2] => <project_root> when file path is .../<project_root>/viewer/workbench/workbench.py
    _project_root = Path(__file__).resolve().parents[2]
    # Put this worktree's project root at the front of sys.path
    sys.path.insert(0, str(_project_root))
    # Expose project_root for other code (e.g., excepthook below)
    project_root = _project_root
except Exception:
    # Fallback: current working directory
    project_root = Path.cwd()

from PyQt5.QtWidgets import QApplication, QMessageBox, QTreeWidgetItem, QFileDialog, QMenu, QAction, QVBoxLayout, QDockWidget, QListWidget, QListWidgetItem, QInputDialog, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QWidget
from PyQt5.QtCore import QTimer, Qt, pyqtSlot, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QCursor
import h5py
import hdf5plugin  # Import hdf5plugin for decompression support
import glob
import numpy as np
import time
import pyqtgraph as pg
from viewer.workbench.workspace.workspace_3d import Workspace3D


# Add the project root to the Python path
# project_root = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(project_root))

from viewer.base_window import BaseWindow
from utils.hdf5_loader import HDF5Loader

# Dimension-specific controls
from viewer.controls.controls_1d import Controls1D
from viewer.controls.controls_2d import Controls2D
from viewer.workbench.managers.roi_manager import ROIManager

# Docks
from viewer.workbench.dock_window import DockWindow
from viewer.workbench.docks.data_structure import DataStructureDock
from viewer.workbench.docks.info_2d_dock import Info2DDock
from viewer.workbench.docks.info_3d_dock import Info3DDock
from viewer.workbench.docks.slice_plane import SlicePlaneDock
#from viewer.workbench.docks.dash_ai import DashAI
from viewer.workbench.docks.dock_win import DockWinDock


class WorkbenchWindow(BaseWindow):
    """
    Workbench window for data analysis.
    Inherits from BaseWindow and adds specific functionality for HDF5 analysis.
    """

    # === Initialization & UI Setup ===
    def __init__(self):
        """Initialize the Workbench window."""
        super().__init__(ui_file_name="workbench/workbench.ui", viewer_name="Workbench")
        self.setup_window_properties("Workbench - Data Analysis", 1600, 1000)

        # ====== DOCKS START ====== #
        # 2d
        #self.dash_sam_dock = DashAI(main_window=self)

        # 3d

        # other
        self.data_structure_dock = DataStructureDock(main_window=self, segment_name="other", dock_area=Qt.LeftDockWidgetArea)
        # info dock (2D)
        self.info_2d_dock = Info2DDock(main_window=self, title="2D Info", segment_name="2d", dock_area=Qt.RightDockWidgetArea)
        # info dock (3D)
        self.info_3d_dock = Info3DDock(main_window=self, title="3D Info", segment_name="3d", dock_area=Qt.RightDockWidgetArea)
        # Add Window control dock (right side under 'other')
        try:
            self.add_window_dock = DockWinDock(main_window=self, segment_name="other", dock_area=Qt.RightDockWidgetArea)
        except Exception:
            self.add_window_dock = None
        # roi 

        # Alias Workbench's tree to the dock's tree widget
        self.tree_data = self.data_structure_dock.tree_data
        # Hide any fixed left panel from the UI and give space to analysis
        if hasattr(self, 'leftPanel') and self.leftPanel is not None:
            self.leftPanel.hide()
        if hasattr(self, 'mainSplitter') and self.mainSplitter is not None:
            self.mainSplitter.setSizes([0, self.width()])
        # ======= DOCKS END ======== #


        # ======= TABS START ======= #
        self.tab_1d = None
        self.tab_2d = None
        self.tab_3d = Workspace3D(parent=self, main_window=self)
        # Slice Controls dock (left, under Data Structure)
        try:
            self.slice_plane_dock = SlicePlaneDock(main_window=self, segment_name="3d", dock_area=Qt.LeftDockWidgetArea)
            # Position below Data Structure dock
            try:
                self.splitDockWidget(self.data_structure_dock, self.slice_plane_dock, Qt.Vertical)
            except Exception:
                pass
        except Exception:
            pass
        # ======= TABS END ========= #


        # ===== CONTROLS START ===== #
        self.controls_1d = Controls1D(self)
        self.controls_2d = Controls2D(self)
        # ======== CONTROLS END ======= #

        # ROI manager to centralize ROI logic
        self.roi_manager = ROIManager(self)
        # Track secondary dock windows (modeless)
        self._dock_windows = []

        self.setup_2d_workspace()
        self.setup_1d_workspace()
        self.setup_workbench_connections()

        # Use shared HDF5 loader utility
        self.h5loader = HDF5Loader()

        # == FILE PATH INFO START ====== #
        self.current_file_path = None
        self.selected_dataset_path = None
        # ==== FILE PATH INFO END ====== #


        # ROI state
        self.rois = []
        self.current_roi = None
        # ROI dock mappings
        self.roi_by_item = {}
        self.item_by_roi_id = {}
        self.roi_names = {}
        self.stats_row_by_roi_id = {}
        self.roi_plot_docks_by_roi_id = {}
        # Setup dock to track ROIs and stats
        try:
            self.roi_manager.setup_docks()
        except Exception:
            pass
        # Initialize 2D axis variables
        try:
            self.axis_2d_x = "Columns"
            self.axis_2d_y = "Row"
        except Exception:
            pass

    def setup_roi_dock(self):
        try:
            self.roi_dock = QDockWidget("ROIs", self)
            self.roi_dock.setAllowedAreas(Qt.RightDockWidgetArea)
            self.roi_list = QListWidget()
            try:
                self.roi_list.itemClicked.connect(self.on_roi_list_item_clicked)
                self.roi_list.itemDoubleClicked.connect(self.on_roi_list_item_double_clicked)
                # Enable right-click context menu on ROI list
                self.roi_list.setContextMenuPolicy(Qt.CustomContextMenu)
                self.roi_list.customContextMenuRequested.connect(self.show_roi_list_context_menu)
            except Exception:
                pass
            self.roi_dock.setWidget(self.roi_list)
            self.addDockWidget(Qt.RightDockWidgetArea, self.roi_dock)
            try:
                self.roi_dock.visibilityChanged.connect(self.on_rois_dock_visibility_changed)
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error setting up ROI dock: {e}")

    def format_roi_text(self, roi):
        try:
            pos = roi.pos(); size = roi.size()
            x = int(pos.x()); y = int(pos.y())
            w = int(size.x()); h = int(size.y())
            name = self.get_roi_name(roi)
            return f"{name}: x={x}, y={y}, w={w}, h={h}"
        except Exception:
            return "ROI"

    def add_roi_to_dock(self, roi):
        try:
            if not hasattr(self, 'roi_list') or self.roi_list is None:
                return
            text = self.format_roi_text(roi)
            item = QListWidgetItem(text)
            self.roi_list.addItem(item)
            self.roi_by_item[item] = roi
            self.item_by_roi_id[id(roi)] = item
        except Exception as e:
            self.update_status(f"Error adding ROI to dock: {e}")

    def update_roi_item(self, roi):
        try:
            item = self.item_by_roi_id.get(id(roi))
            if item is not None:
                item.setText(self.format_roi_text(roi))
        except Exception:
            pass

    def on_roi_list_item_clicked(self, item):
        try:
            roi = self.roi_by_item.get(item)
            if roi:
                self.set_active_roi(roi)
        except Exception as e:
            self.update_status(f"Error selecting ROI from dock: {e}")

    def on_roi_list_item_double_clicked(self, item):
        try:
            roi = self.roi_by_item.get(item)
            if roi:
                self.show_roi_stats_for_roi(roi)
        except Exception as e:
            self.update_status(f"Error showing ROI stats from dock: {e}")

    def show_roi_list_context_menu(self, position):
        """Show context menu for ROI list items with an option to open a PyQtGraph view of the ROI."""
        try:
            if not hasattr(self, 'roi_list') or self.roi_list is None:
                return
            item = self.roi_list.itemAt(position)
            if item is None:
                return
            roi = self.roi_by_item.get(item)
            if roi is None:
                return
            menu = QMenu(self)
            action_plot = QAction("Open ROI Plot", self)
            action_plot.triggered.connect(lambda: self.open_roi_plot_dock(roi))
            menu.addAction(action_plot)
            # Open ROI Plot dock in DockWindow
            action_plot_dock_window = QAction("Open ROI Plot Dock (Window)", self)
            action_plot_dock_window.triggered.connect(lambda: self.open_roi_plot_dock_in_window(roi))
            menu.addAction(action_plot_dock_window)
            # Also provide ROI Math dock
            action_math_dock = QAction("Open ROI Math Dock", self)
            action_math_dock.triggered.connect(lambda: self.open_roi_math_dock(roi))
            menu.addAction(action_math_dock)
            # Open ROI Math dock in DockWindow
            action_math_dock_window = QAction("Open ROI Math Dock (Window)", self)
            action_math_dock_window.triggered.connect(lambda: self.open_roi_math_dock_in_window(roi))
            menu.addAction(action_math_dock_window)
            # Potential future actions can be added here
            menu.exec_(self.roi_list.mapToGlobal(position))
        except Exception as e:
            self.update_status(f"Error showing ROI context menu: {e}")

    def open_roi_plot(self, roi):
        """Open a modeless window displaying a 1D plot of the selected ROI region."""
        try:
            frame_data = self.get_current_frame_data()
            if frame_data is None:
                QMessageBox.information(self, "ROI Plot", "No image data available.")
                return
            # Compute ROI bounds
            pos = roi.pos(); size = roi.size()
            x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
            w = max(1, int(size.x())); h = max(1, int(size.y()))
            height, width = frame_data.shape
            x1 = min(width, x0 + w); y1 = min(height, y0 + h)
            if x0 >= x1 or y0 >= y1:
                QMessageBox.information(self, "ROI Plot", "ROI area is empty or out of bounds.")
                return
            sub = frame_data[y0:y1, x0:x1]
            # Create and show the 1D plot dialog (modeless)
            try:
                from viewer.workbench.roi_plot_dialog import ROIPlotDialog
            except Exception:
                ROIPlotDialog = None
            if ROIPlotDialog is None:
                QMessageBox.warning(self, "ROI Plot", "ROIPlotDialog not available.")
                return
            # Keep a reference to avoid GC
            if not hasattr(self, '_roi_plot_dialogs'):
                self._roi_plot_dialogs = []
            dlg = ROIPlotDialog(self, sub)
            dlg.setWindowTitle(f"ROI: {self.get_roi_name(roi)}")
            dlg.resize(600, 500)
            # Wire ROI & frame changes to update dialog data
            def _update_dialog_data():
                try:
                    frame = self.get_current_frame_data()
                except Exception:
                    frame = None
                if frame is None:
                    return
                sub_img = None
                try:
                    image_item = getattr(self.image_view, 'imageItem', None) if hasattr(self, 'image_view') else None
                    if image_item is not None:
                        sub_img = roi.getArrayRegion(frame, image_item)
                        if sub_img is not None and hasattr(sub_img, 'ndim') and sub_img.ndim > 2:
                            sub_img = np.squeeze(sub_img)
                except Exception:
                    sub_img = None
                if sub_img is None or int(getattr(sub_img, 'size', 0)) == 0:
                    # Fallback to axis-aligned bbox
                    pos = roi.pos(); size = roi.size()
                    x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
                    w = max(1, int(size.x())); h = max(1, int(size.y()))
                    hgt, wid = frame.shape
                    x1 = min(wid, x0 + w); y1 = min(hgt, y0 + h)
                    if x0 < x1 and y0 < y1:
                        sub_img = frame[y0:y1, x0:x1]
                if sub_img is not None and int(getattr(sub_img, 'size', 0)) > 0:
                    try:
                        dlg.update_roi_data(sub_img)
                    except Exception:
                        pass
            try:
                if hasattr(roi, 'sigRegionChanged'):
                    roi.sigRegionChanged.connect(_update_dialog_data)
                if hasattr(roi, 'sigRegionChangeFinished'):
                    roi.sigRegionChangeFinished.connect(_update_dialog_data)
            except Exception:
                pass
            try:
                if hasattr(self, 'frame_spinbox'):
                    self.frame_spinbox.valueChanged.connect(lambda _: _update_dialog_data())
            except Exception:
                pass
            dlg.show()
            # Track alive dialogs
            self._roi_plot_dialogs.append(dlg)
        except Exception as e:
            self.update_status(f"Error opening ROI plot: {e}")

    def open_roi_math_dock(self, roi):
        """Open a dockable ROI Math window on the right dock area."""
        try:
            # Ensure ROI exists
            if roi is None:
                QMessageBox.information(self, "ROI Math", "No ROI selected.")
                return
            # Import the ROIMathDock
            try:
                from viewer.workbench.roi_math_dock import ROIMathDock
            except Exception:
                ROIMathDock = None
            if ROIMathDock is None:
                QMessageBox.warning(self, "ROI Math", "ROIMathDock not available.")
                return
            # Create and add the dock widget
            dock_title = f"ROI Math: {self.get_roi_name(roi)}"
            dock = ROIMathDock(self, dock_title, self, roi)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            # Register toggle under Windows->2d submenu
            try:
                self.add_dock_toggle_action(dock, dock_title, segment_name="2d")
            except Exception:
                pass
            dock.show()
            # Track alive docks
            if not hasattr(self, '_roi_math_dock_widgets'):
                self._roi_math_dock_widgets = []
            self._roi_math_dock_widgets.append(dock)
            try:
                if not hasattr(self, 'roi_math_docks_by_roi_id') or self.roi_math_docks_by_roi_id is None:
                    self.roi_math_docks_by_roi_id = {}
                self.roi_math_docks_by_roi_id.setdefault(id(roi), []).append(dock)
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error opening ROI Math dock: {e}")

    def open_roi_plot_dock(self, roi):
        """Open a dockable 1D plot of the selected ROI region on the right dock area."""
        try:
            frame_data = self.get_current_frame_data()
            if frame_data is None:
                QMessageBox.information(self, "ROI Plot", "No image data available.")
                return
            # Compute ROI bounds
            pos = roi.pos(); size = roi.size()
            x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
            w = max(1, int(size.x())); h = max(1, int(size.y()))
            height, width = frame_data.shape
            x1 = min(width, x0 + w); y1 = min(height, y0 + h)
            if x0 >= x1 or y0 >= y1:
                QMessageBox.information(self, "ROI Plot", "ROI area is empty or out of bounds.")
                return
            sub = frame_data[y0:y1, x0:x1]
            # Create and add the dock widget
            try:
                from viewer.workbench.roi_plot_dock import ROIPlotDock
            except Exception:
                ROIPlotDock = None
            if ROIPlotDock is None:
                QMessageBox.warning(self, "ROI Plot", "ROIPlotDock not available.")
                return
            dock_title = f"ROI: {self.get_roi_name(roi)}"
            dock = ROIPlotDock(self, dock_title, self, roi)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            # Register toggle under Windows->2d submenu
            try:
                self.add_dock_toggle_action(dock, dock_title, segment_name="2d")
            except Exception:
                pass
            dock.show()
            # Track alive docks
            if not hasattr(self, '_roi_plot_dock_widgets'):
                self._roi_plot_dock_widgets = []
            self._roi_plot_dock_widgets.append(dock)
            try:
                if not hasattr(self, 'roi_plot_docks_by_roi_id') or self.roi_plot_docks_by_roi_id is None:
                    self.roi_plot_docks_by_roi_id = {}
                self.roi_plot_docks_by_roi_id.setdefault(id(roi), []).append(dock)
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error opening ROI plot dock: {e}")

    def create_dock_window_and_show(self):
        """Create a new modeless empty window to host dockables later."""
        try:
            # Enforce a cap of 2 additional windows
            try:
                if len(self._dock_windows) >= 2:
                    QMessageBox.information(self, "Add Window", "Maximum of 2 additional windows is reached.")
                    return
            except Exception:
                # If something goes wrong reading the count, continue to attempt creation
                pass

            win = DockWindow(self, title="Dock Window", width=1000, height=700)
            # Keep reference to prevent garbage collection while open
            self._dock_windows.append(win)

            # Wire destroyed hook: remove from list and notify dock to refresh count/labels
            try:
                def _on_win_destroyed(_obj=None):
                    try:
                        if win in self._dock_windows:
                            self._dock_windows.remove(win)
                    except Exception:
                        pass
                    # Notify dock about count change
                    try:
                        self._notify_add_window_count_changed()
                    except Exception:
                        pass
                win.destroyed.connect(_on_win_destroyed)
            except Exception:
                pass

            # Show modeless and bring to foreground
            win.show()
            try:
                win.raise_()
                win.activateWindow()
            except Exception:
                pass

            # Notify dock after creation
            try:
                self._notify_add_window_count_changed()
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error creating Dock Window: {e}")

    # --- DockWindow helpers (no BaseDock/global menu changes) ---
    def get_first_dock_window(self):
        """Return the first available DockWindow if one exists, else None."""
        try:
            wins = getattr(self, "_dock_windows", None)
            if isinstance(wins, (list, tuple)) and len(wins) > 0:
                # Prefer a visible/valid window
                for w in wins:
                    try:
                        if w is not None and not w.isHidden():
                            return w
                    except Exception:
                        # Fallback: return the first non-None even if we cannot query visibility
                        if w is not None:
                            return w
                # If all hidden or checks failed, return the most recent one
                return wins[-1]
        except Exception:
            pass
        return None

    def get_or_create_dock_window(self):
        """Return an existing DockWindow or create one using create_dock_window_and_show()."""
        try:
            win = self.get_first_dock_window()
            if win is not None:
                try:
                    # Bring to front if possible
                    if hasattr(win, "show_and_focus"):
                        win.show_and_focus()
                    else:
                        win.show()
                        win.raise_()
                        win.activateWindow()
                except Exception:
                    pass
                return win
            # None found -> create one on demand
            self.create_dock_window_and_show()
            try:
                wins = getattr(self, "_dock_windows", None)
                if isinstance(wins, (list, tuple)) and len(wins) > 0:
                    return wins[-1]
            except Exception:
                pass
            return None
        except Exception:
            return None

    def _notify_add_window_count_changed(self):
        """Notify the Add Window dock (if present) that the window count changed."""
        try:
            if hasattr(self, 'add_window_dock') and self.add_window_dock is not None:
                try:
                    self.add_window_dock.refresh_counts()
                except Exception:
                    pass
        except Exception:
            pass

    # --- ROI docks in separate DockWindow ---
    def open_roi_plot_dock_in_window(self, roi):
        """Open the ROI Plot dock inside a DockWindow (creates one on demand)."""
        try:
            if roi is None:
                QMessageBox.information(self, "ROI Plot", "No ROI selected.")
                return
            win = self.get_or_create_dock_window()
            if win is None:
                QMessageBox.information(self, "ROI Plot", "Could not open or create a Dock Window.")
                return
            try:
                from viewer.workbench.roi_plot_dock import ROIPlotDock
            except Exception:
                ROIPlotDock = None
            if ROIPlotDock is None:
                QMessageBox.warning(self, "ROI Plot", "ROIPlotDock not available.")
                return
            dock_title = f"ROI: {self.get_roi_name(roi)}"
            dock = ROIPlotDock(win, dock_title, self, roi)
            try:
                win.addDockWidget(Qt.RightDockWidgetArea, dock)
            except Exception:
                # Best effort: show without docking if addDockWidget fails
                pass
            dock.show()
            # Track alive docks for title/refresh updates
            try:
                if not hasattr(self, '_roi_plot_dock_widgets') or self._roi_plot_dock_widgets is None:
                    self._roi_plot_dock_widgets = []
                self._roi_plot_dock_widgets.append(dock)
                if not hasattr(self, 'roi_plot_docks_by_roi_id') or self.roi_plot_docks_by_roi_id is None:
                    self.roi_plot_docks_by_roi_id = {}
                self.roi_plot_docks_by_roi_id.setdefault(id(roi), []).append(dock)
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error opening ROI Plot dock in window: {e}")

    def open_roi_math_dock_in_window(self, roi):
        """Open the ROI Math dock inside a DockWindow (creates one on demand)."""
        try:
            if roi is None:
                QMessageBox.information(self, "ROI Math", "No ROI selected.")
                return
            win = self.get_or_create_dock_window()
            if win is None:
                QMessageBox.information(self, "ROI Math", "Could not open or create a Dock Window.")
                return
            try:
                from viewer.workbench.roi_math_dock import ROIMathDock
            except Exception:
                ROIMathDock = None
            if ROIMathDock is None:
                QMessageBox.warning(self, "ROI Math", "ROIMathDock not available.")
                return
            dock_title = f"ROI Math: {self.get_roi_name(roi)}"
            dock = ROIMathDock(win, dock_title, self, roi)
            try:
                win.addDockWidget(Qt.RightDockWidgetArea, dock)
            except Exception:
                pass
            dock.show()
            # Track alive docks
            try:
                if not hasattr(self, '_roi_math_dock_widgets') or self._roi_math_dock_widgets is None:
                    self._roi_math_dock_widgets = []
                self._roi_math_dock_widgets.append(dock)
                if not hasattr(self, 'roi_math_docks_by_roi_id') or self.roi_math_docks_by_roi_id is None:
                    self.roi_math_docks_by_roi_id = {}
                self.roi_math_docks_by_roi_id.setdefault(id(roi), []).append(dock)
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error opening ROI Math dock in window: {e}")

    def setup_roi_stats_dock(self):
        try:
            self.roi_stats_dock = QDockWidget("ROI", self)
            self.roi_stats_dock.setAllowedAreas(Qt.RightDockWidgetArea)
            self.roi_stats_table = QTableWidget(0, 11, self.roi_stats_dock)
            self.roi_stats_table.setHorizontalHeaderLabels(["Name","sum","min","max","mean","std","count","x","y","w","h"])
            self.roi_stats_dock.setWidget(self.roi_stats_table)
            self.addDockWidget(Qt.RightDockWidgetArea, self.roi_stats_dock)
            try:
                self.roi_stats_dock.visibilityChanged.connect(self.on_roi_stats_dock_visibility_changed)
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error setting up ROI stats dock: {e}")

    def get_roi_name(self, roi):
        try:
            # Prefer ROIManager's naming to keep everything in sync (including renames)
            if hasattr(self, 'roi_manager') and self.roi_manager is not None:
                try:
                    return self.roi_manager.get_roi_name(roi)
                except Exception:
                    pass
            # Fallback to local mapping
            name = self.roi_names.get(id(roi))
            if name:
                return name
            idx = 1
            if hasattr(self, 'rois') and roi in self.rois:
                idx = self.rois.index(roi) + 1
            name = f"ROI {idx}"
            self.roi_names[id(roi)] = name
            return name
        except Exception:
            return "ROI"

    def rename_roi(self, roi):
        """Delegate to ROIManager."""
        try:
            self.roi_manager.rename_roi(roi)
        except Exception as e:
            self.update_status(f"Error renaming ROI: {e}")

    def update_roi_plot_dock_title(self, roi):
        try:
            name = self.get_roi_name(roi)
            title = f"ROI: {name}"
            try:
                docks = self.roi_plot_docks_by_roi_id.get(id(roi), []) if hasattr(self, 'roi_plot_docks_by_roi_id') else []
            except Exception:
                docks = []
            for dock in list(docks):
                try:
                    dock.setWindowTitle(title)
                except Exception:
                    pass
        except Exception:
            pass

    def ensure_stats_row_for_roi(self, roi):
        try:
            if id(roi) in self.stats_row_by_roi_id:
                return self.stats_row_by_roi_id[id(roi)]
            if not hasattr(self, 'roi_stats_table') or self.roi_stats_table is None:
                return None
            row = self.roi_stats_table.rowCount()
            self.roi_stats_table.insertRow(row)
            self.stats_row_by_roi_id[id(roi)] = row
            # set name cell
            name = self.get_roi_name(roi)
            self.roi_stats_table.setItem(row, 0, QTableWidgetItem(name))
            return row
        except Exception:
            return None

    def update_stats_table_for_roi(self, roi, stats):
        try:
            row = self.ensure_stats_row_for_roi(roi)
            if row is None:
                return
            # name cell keep in sync
            self.roi_stats_table.setItem(row, 0, QTableWidgetItem(self.get_roi_name(roi)))
            # fill numeric cells with xywh at the end
            self.roi_stats_table.setItem(row, 1, QTableWidgetItem(f"{stats['sum']:.3f}"))
            self.roi_stats_table.setItem(row, 2, QTableWidgetItem(f"{stats['min']:.3f}"))
            self.roi_stats_table.setItem(row, 3, QTableWidgetItem(f"{stats['max']:.3f}"))
            self.roi_stats_table.setItem(row, 4, QTableWidgetItem(f"{stats['mean']:.3f}"))
            self.roi_stats_table.setItem(row, 5, QTableWidgetItem(f"{stats['std']:.3f}"))
            self.roi_stats_table.setItem(row, 6, QTableWidgetItem(str(stats['count'])))
            self.roi_stats_table.setItem(row, 7, QTableWidgetItem(str(stats['x'])))
            self.roi_stats_table.setItem(row, 8, QTableWidgetItem(str(stats['y'])))
            self.roi_stats_table.setItem(row, 9, QTableWidgetItem(str(stats['w'])))
            self.roi_stats_table.setItem(row, 10, QTableWidgetItem(str(stats['h'])))
        except Exception:
            pass

    def on_rois_dock_visibility_changed(self, visible):
        try:
            if hasattr(self, 'action_show_rois_dock'):
                self.action_show_rois_dock.setChecked(bool(visible))
        except Exception:
            pass

    def on_roi_stats_dock_visibility_changed(self, visible):
        try:
            if hasattr(self, 'action_show_roi_stats_dock'):
                self.action_show_roi_stats_dock.setChecked(bool(visible))
        except Exception:
            pass

    def setup_workbench_connections(self):
        """Set up connections specific to the workbench."""
        # Tree widget connections
        if hasattr(self, 'tree_data'):
            self.tree_data.itemClicked.connect(self.on_tree_item_clicked)
            self.tree_data.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
            self.tree_data.setContextMenuPolicy(Qt.CustomContextMenu)
            self.tree_data.customContextMenuRequested.connect(self.show_context_menu)

        # View menu actions
        if hasattr(self, 'actionCollapseAll'):
            self.actionCollapseAll.triggered.connect(self.collapse_all)
        if hasattr(self, 'actionExpandAll'):
            self.actionExpandAll.triggered.connect(self.expand_all)

        # Windows menu: add toggles to show/hide docks, with room for future items
        # try:
        #     windows_menu = None
        #     if hasattr(self, 'menuBar') and self.menuBar is not None:
        #         try:
        #             windows_menu = self.menuBar.addMenu("Windows")
        #         except Exception:
        #             windows_menu = QMenu("Windows", self)
        #             try:
        #                 self.menuBar().addMenu(windows_menu)
        #             except Exception:
        #                 pass
        #     else:
        #         windows_menu = QMenu("Windows", self)
        #         try:
        #             self.menuBar().addMenu(windows_menu)
        #         except Exception:
        #             pass

        #     # ROI dock toggle (renamed from 'ROI Stats' to 'ROI')
        #         self.action_show_roi_stats_dock = QAction("ROI", self)
        #         self.action_show_roi_stats_dock.setCheckable(True)
        #         self.action_show_roi_stats_dock.setChecked(True if hasattr(self, 'roi_stats_dock') and self.roi_stats_dock.isVisible() else True)
        #         self.action_show_roi_stats_dock.toggled.connect(lambda checked: hasattr(self, 'roi_stats_dock') and self.roi_stats_dock.setVisible(checked))
        #         windows_menu.addAction(self.action_show_roi_stats_dock)

        #         # Open ROI Math dock for the active ROI
        #         self.action_open_roi_math_dock = QAction("ROI Math (Active ROI)", self)
        #         self.action_open_roi_math_dock.setToolTip("Open ROI Math dock for the currently active ROI")
        #         self.action_open_roi_math_dock.triggered.connect(lambda: hasattr(self, 'current_roi') and self.open_roi_math_dock(self.current_roi))
        #         windows_menu.addAction(self.action_open_roi_math_dock)

        #         # Add Window: open an empty, modeless window for dockables
        #         self.action_add_window = QAction("Add Window", self)
        #         self.action_add_window.setToolTip("Open a new empty window for dockable tools")
        #         self.action_add_window.triggered.connect(self.create_dock_window_and_show)
        #         windows_menu.addAction(self.action_add_window)
        # except Exception:
        #     pass

        # Set up default splitter sizes
        self.setup_default_splitter_sizes()

        # Initialize file info text box
        self.initialize_file_info_display()

    def setup_default_splitter_sizes(self):
        """Set default splitter sizes for the horizontal splitter."""
        if hasattr(self, 'mainSplitter'):
            # Calculate 15% of window width for data structure panel
            window_width = self.width()
            data_panel_width = int(window_width * 0.15)
            analysis_panel_width = window_width - data_panel_width

            # Set the horizontal splitter sizes
            self.mainSplitter.setSizes([data_panel_width, analysis_panel_width])

    def initialize_file_info_display(self):
        """Initialize the file information display."""
        if hasattr(self, 'file_info_text'):
            self.update_file_info_display("No file loaded", {})

    def setup_2d_workspace(self):
        """Set up the 2D workspace with PyQtGraph plotitem functionality."""
        try:
            # Setup the 2D plot viewer with PyQtGraph
            self.setup_2d_plot_viewer()

            # Setup 2D controls connections (delegated)
            self.controls_2d.setup()

        except Exception as e:
            self.update_status(f"Error setting up 2D workspace: {e}")
            # Fallback to keeping the placeholder if setup fails

    # === File type helpers ===
    def _is_hdf5_path(self, file_path: str) -> bool:
        try:
            ext = str(Path(file_path).suffix or "").lower()
            return ext in (".h5", ".hdf5")
        except Exception:
            return False

    def _is_image_path(self, file_path: str) -> bool:
        try:
            ext = str(Path(file_path).suffix or "").lower()
            return ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
        except Exception:
            return False

    def setup_2d_plot_viewer(self):
        """Set up the 2D plot viewer with PyQtGraph PlotItem and ImageView."""
        try:
            # Create the plot item and image view similar to HKL slice 2D viewer
            self.plot_item = pg.PlotItem()
            self.image_view = pg.ImageView(view=self.plot_item)

            # Set axis labels
            self.plot_item.setLabel('bottom', 'Columns [pixels]')
            self.plot_item.setLabel('left', 'Row [pixels]')

            # Lock aspect ratio for square pixels
            try:
                self.image_view.view.setAspectLocked(True)
            except Exception:
                pass

            # Add the image view directly to the plot host
            if hasattr(self, 'layoutPlotHost'):
                self.layoutPlotHost.addWidget(self.image_view)
            else:
                print("Warning: layoutPlotHost not found, 2D plot may not display correctly")


            # Initialize with empty data
            self.clear_2d_plot()

            # Setup hover overlays and mouse tracking
            self._setup_2d_hover()

            # Set default hover enabled and preserve default context menu
            try:
                self._hover_enabled = True
                if hasattr(self, 'image_view') and self.image_view is not None:
                    # Restore default context menu (do not override with custom)
                    self.image_view.setContextMenuPolicy(Qt.DefaultContextMenu)
            except Exception:
                pass

        except Exception as e:
            self.update_status(f"Error setting up 2D plot viewer: {e}")

    # === File loading override ===
    def load_file_content(self, file_path: str):
        """Load content for the selected file.

        Supports HDF5 files (adds to tree) and common image formats (loads directly into 2D view).
        """
        try:
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                QMessageBox.critical(self, "Open File", "Selected file does not exist.")
                return

            # Store current path
            self.current_file_path = file_path

            # Clear previous visualizations
            try:
                self.clear_2d_plot()
            except Exception:
                pass
            try:
                self.clear_3d_plot()
            except Exception:
                pass
            self.selected_dataset_path = None

            # Branch by file type
            if self._is_hdf5_path(file_path):
                # Add HDF5 to tree and show default info
                try:
                    self.load_single_h5_file(file_path)
                except Exception as e:
                    QMessageBox.critical(self, "Open HDF5", f"Failed to load HDF5 file: {e}")
                    self.update_status("Failed to load file", level='error')
                    return

                if hasattr(self, 'file_status_label'):
                    self.file_status_label.setText(f"HDF5 file loaded: {os.path.basename(file_path)}")
                if hasattr(self, 'dataset_info_text'):
                    self.dataset_info_text.setPlainText("Select a dataset from the tree to view detailed information.")

                # Update file info panel
                self.update_file_info_display(file_path)
                self.update_status("HDF5 File Loaded Successfully")
                return

            if self._is_image_path(file_path):
                data = self._load_image_file(file_path)
                if data is None or int(getattr(data, 'size', 0)) == 0:
                    QMessageBox.warning(self, "Open Image", "Failed to load image data from file.")
                    self.update_status("Failed to load image", level='error')
                    return
                # Add image to tree for visibility in data structure
                try:
                    self.add_single_image_file(file_path, data)
                except Exception:
                    # Non-fatal if tree add fails
                    pass
                # Display the image or stack in 2D viewer
                self.display_2d_data(data)
                if hasattr(self, 'tabWidget_analysis'):
                    try:
                        self.tabWidget_analysis.setCurrentIndex(0)
                    except Exception:
                        pass
                # Basic info text
                if hasattr(self, 'dataset_info_text'):
                    try:
                        shp = tuple(map(int, getattr(data, 'shape', ())))
                        info = [f"Image file: {os.path.basename(file_path)}",
                                f"Shape: {shp}",
                                f"Data Type: {getattr(data, 'dtype', '')}",
                                f"Size: {int(getattr(data, 'size', 0)):,} elements"]
                        self.dataset_info_text.setPlainText("\n".join(info))
                    except Exception:
                        pass
                if hasattr(self, 'file_status_label'):
                    self.file_status_label.setText(f"Image loaded: {os.path.basename(file_path)}")
                # Update generic file info without forcing HDF5 read
                self.update_file_info_display(file_path)
                self.update_status("Image loaded successfully")
                return

            # Unsupported type: just show generic info
            self.update_file_info_display(file_path)
            QMessageBox.information(self, "Open File", "Unsupported file type for visualization.")
            self.update_status("Unsupported file type", level='warning')
        except Exception as e:
            QMessageBox.critical(self, "Open File", f"Failed to open file: {e}")
            self.update_status(f"Failed to open file: {e}", level='error')

    def _load_image_file(self, file_path: str):
        """Load an image file (.tif/.tiff/.png/.jpg/.jpeg/.bmp) into a numpy array.

        For multi-page TIFF, returns a 3D array (frames, H, W) if applicable.
        Color images (H, W, C) are converted to grayscale for 2D visualization.
        """
        try:
            ext = str(Path(file_path).suffix or "").lower()
            data = None
            # Prefer tifffile for TIFF
            if ext in (".tif", ".tiff"):
                try:
                    import tifffile as tiff
                    data = tiff.imread(file_path)
                except Exception:
                    data = None
            # Fallbacks for other formats
            if data is None:
                try:
                    from PIL import Image
                    img = Image.open(file_path)
                    # If RGB(A), convert to grayscale
                    if hasattr(img, 'mode') and img.mode in ("RGB", "RGBA", "P", "CMYK"):
                        try:
                            img = img.convert("L")
                        except Exception:
                            pass
                    data = np.asarray(img)
                except Exception:
                    data = None
            if data is None:
                try:
                    import imageio.v2 as iio
                    data = iio.imread(file_path)
                except Exception:
                    data = None

            if data is None:
                return None

            arr = np.asarray(data)
            # If color image HxWxC, convert to luminance
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                try:
                    # Use ITU-R BT.709 luma coefficients
                    rgb = arr[..., :3].astype(np.float32)
                    arr = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                except Exception:
                    # Fallback to single channel
                    arr = arr[..., 0]
            # Ensure float32 for visualization consistency
            try:
                arr = np.asarray(arr, dtype=np.float32)
            except Exception:
                pass
            # If single frame TIFF returned as (H,W), fine; if multi-page (N,H,W), fine.
            return arr
        except Exception as e:
            self.update_status(f"Error loading image file: {e}", level='error')
            return None

    def add_single_image_file(self, file_path: str, data=None):
        """Add a single image file as a root item in the data structure tree.

        Optionally provide loaded data to annotate shape.
        """
        try:
            if not hasattr(self, 'tree_data') or self.tree_data is None:
                return
            # Create root item
            name = os.path.basename(file_path)
            root_item = QTreeWidgetItem([name])
            root_item.setData(0, Qt.UserRole + 1, file_path)  # Store path
            root_item.setData(0, Qt.UserRole + 2, "file_root")
            # Mark as renderable by setting a child node with a label
            try:
                shape_str = ""
                if data is not None and hasattr(data, 'shape'):
                    shp = tuple(map(int, data.shape))
                    shape_str = f" (Shape: {shp})"
                child = QTreeWidgetItem([f"image{shape_str}"])
                # Store a synthetic path so double-click can detect image load
                child.setData(0, 32, "__image__")
                child.setData(0, Qt.UserRole + 3, True)  # renderable hint
                root_item.addChild(child)
            except Exception:
                pass
            # Insert at top
            self.tree_data.insertTopLevelItem(0, root_item)
            root_item.setExpanded(False)
        except Exception:
            pass

    # === Controls: 2D ===
    def setup_controls_2d(self):
        """Set up connections for the 2D viewer controls."""
        try:
            # Connect colormap selection
            if hasattr(self, 'cbColorMapSelect_2d'):
                self.cbColorMapSelect_2d.currentTextChanged.connect(self.on_colormap_changed)

            # Connect auto levels checkbox
            if hasattr(self, 'cbAutoLevels'):
                self.cbAutoLevels.toggled.connect(self.on_auto_levels_toggled)

            # Connect frame navigation controls from UI
            if hasattr(self, 'btn_prev_frame'):
                self.btn_prev_frame.clicked.connect(self.previous_frame)
            if hasattr(self, 'btn_next_frame'):
                self.btn_next_frame.clicked.connect(self.next_frame)
            if hasattr(self, 'frame_spinbox'):
                self.frame_spinbox.valueChanged.connect(self.on_frame_spinbox_changed)

            # Connect new speckle analysis controls
            if hasattr(self, 'cbLogScale'):
                self.cbLogScale.toggled.connect(self.on_log_scale_toggled)

            if hasattr(self, 'sbVmin'):
                self.sbVmin.valueChanged.connect(self.on_vmin_changed)

            if hasattr(self, 'sbVmax'):
                self.sbVmax.valueChanged.connect(self.on_vmax_changed)

            if hasattr(self, 'btnDrawROI'):
                self.btnDrawROI.clicked.connect(self.on_draw_roi_clicked)

            if hasattr(self, 'sbRefFrame'):
                self.sbRefFrame.valueChanged.connect(self.on_ref_frame_changed)

            if hasattr(self, 'sbOtherFrame'):
                self.sbOtherFrame.valueChanged.connect(self.on_other_frame_changed)





            # Playback controls for 3D stacks in 2D viewer (UI-defined or created programmatically)
            try:
                # Ensure playback timer exists
                if not hasattr(self, 'play_timer') or self.play_timer is None:
                    self.play_timer = QTimer(self)
                    try:
                        self.play_timer.timeout.connect(self._advance_frame_playback)
                        print("[PLAYBACK] Created play_timer and wired timeout")
                    except Exception as e:
                        print(f"[PLAYBACK] ERROR wiring timer: {e}")

                # Wire controls if present in UI
                if hasattr(self, 'btn_play'):
                    try:
                        self.btn_play.clicked.connect(self.start_playback)
                        print("[PLAYBACK] Wired btn_play -> start_playback")
                    except Exception as e:
                        print(f"[PLAYBACK] ERROR wiring btn_play: {e}")
                if hasattr(self, 'btn_pause'):
                    try:
                        self.btn_pause.clicked.connect(self.pause_playback)
                        print("[PLAYBACK] Wired btn_pause -> pause_playback")
                    except Exception as e:
                        print(f"[PLAYBACK] ERROR wiring btn_pause: {e}")
                if hasattr(self, 'sb_fps'):
                    try:
                        self.sb_fps.valueChanged.connect(self.on_fps_changed)
                        print("[PLAYBACK] Wired sb_fps -> on_fps_changed")
                    except Exception as e:
                        print(f"[PLAYBACK] ERROR wiring sb_fps: {e}")
                # cb_auto_replay is read in _advance_frame_playback; no signal wiring needed

                # Default disabled; enabled when 3D data with >3 frames is loaded
                try:
                    if hasattr(self, 'btn_play'):
                        self.btn_play.setEnabled(False)
                    if hasattr(self, 'btn_pause'):
                        self.btn_pause.setEnabled(False)
                    if hasattr(self, 'sb_fps'):
                        self.sb_fps.setEnabled(False)
                    if hasattr(self, 'cb_auto_replay'):
                        self.cb_auto_replay.setEnabled(False)
                        try:
                            # Select auto replay by default
                            self.cb_auto_replay.setChecked(True)
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass



        except Exception as e:
            self.update_status(f"Error setting up 2D connections: {e}")

    def setup_1d_workspace(self):
        """Set up the 1D workspace with PyQtGraph PlotItem."""
        try:
            self.plot_item_1d = pg.PlotItem()
            self.plot_widget_1d = pg.PlotWidget(plotItem=self.plot_item_1d)
            self.plot_item_1d.setLabel('bottom', 'Index')
            self.plot_item_1d.setLabel('left', 'Value')
            if hasattr(self, 'layout1DPlotHost'):
                self.layout1DPlotHost.addWidget(self.plot_widget_1d)
            else:
                print("Warning: layout1DPlotHost not found, 1D plot may not display correctly")
            self.clear_1d_plot()
            # Setup 1D controls connections (delegated)
            self.controls_1d.setup()
        except Exception as e:
            self.update_status(f"Error setting up 1D workspace: {e}")

    # === Controls: 1D ===
    def setup_controls_1d(self):
        """Set up connections for the 1D controls."""
        try:
            # Placeholder for future 1D controls (e.g., levels, scale, etc.)
            pass
        except Exception as e:
            self.update_status(f"Error setting up 1D connections: {e}")

    # === Controls: 3D ===
    def setup_controls_3d(self):
        pass

    def setup_2d_file_display(self):
        """Set up the 2D file information display in the main workspace."""
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QGroupBox, QTabWidget, QSplitter
        from PyQt5.QtCore import Qt

        # Create a vertical splitter for the workspace
        self.workspace_splitter = QSplitter(Qt.Vertical)

        # Create top section for main workspace
        self.main_workspace_widget = QGroupBox()
        self.main_workspace_layout = QVBoxLayout(self.main_workspace_widget)

        # File status label at the top
        self.file_status_label = QLabel("No HDF5 file loaded")
        self.file_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; padding: 10px;")
        self.file_status_label.setAlignment(Qt.AlignCenter)
        self.main_workspace_layout.addWidget(self.file_status_label)

        # Add a spacer to push content to top
        from PyQt5.QtWidgets import QSpacerItem, QSizePolicy
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_workspace_layout.addItem(spacer)

        # Create bottom section for tabs (compact)
        self.info_tabs_widget = QGroupBox()
        self.info_tabs_layout = QVBoxLayout(self.info_tabs_widget)
        self.info_tabs_layout.setContentsMargins(6, 6, 6, 6)

        # Create tab widget with compact size
        self.info_tabs = QTabWidget()
        self.info_tabs.setMaximumHeight(150)  # Limit height to make it compact
        self.info_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                border: 1px solid #dee2e6;
                padding: 6px 12px;
                margin-right: 2px;
                font-size: 9pt;
            }
            QTabBar::tab:selected {
                background-color: #f8f9fa;
                border-bottom: 1px solid #f8f9fa;
            }
        """)

        # Dataset Information Tab
        self.dataset_info_text = QTextEdit()
        self.dataset_info_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: none;
                padding: 6px;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
            }
        """)
        self.dataset_info_text.setReadOnly(True)
        self.dataset_info_text.setPlainText("Select a dataset from the tree to view detailed information.")

        # File Information Tab
        self.file_info_text = QTextEdit()
        self.file_info_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: none;
                padding: 6px;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
            }
        """)
        self.file_info_text.setReadOnly(True)
        self.file_info_text.setPlainText("Load an HDF5 file to view file information.")

        # Add tabs
        self.info_tabs.addTab(self.dataset_info_text, "Dataset Info")
        self.info_tabs.addTab(self.file_info_text, "File Info")

        # Add tab widget to bottom container
        self.info_tabs_layout.addWidget(self.info_tabs)

        # Add widgets to splitter
        self.workspace_splitter.addWidget(self.main_workspace_widget)
        self.workspace_splitter.addWidget(self.info_tabs_widget)

        # Set splitter sizes (85% for main workspace, 15% for tabs)
        self.workspace_splitter.setSizes([850, 150])

        # Add the splitter to the analysis layout
        self.analysisLayout.addWidget(self.workspace_splitter)

    # === Supers: BaseWindow overrides ===
    def get_file_filters(self):
        """
        Get file filters for Workbench open/save dialogs.

        Returns:
            str: File filter string for QFileDialog
        """
        # Allow both HDF5 and common image formats in the file browser
        # so users can select .tif/.tiff and other images directly.
        return (
            "Data Files (*.h5 *.hdf5 *.tif *.tiff *.png *.jpg *.jpeg *.bmp);;"
            "HDF5 Files (*.h5 *.hdf5);;"
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;"
            "All Files (*)"
        )

    # def load_file_content(self, file_path):
    #     """
    #     Load HDF5 file content and add it to the top of the data tree.

    #     Args:
    #         file_path (str): Path to the HDF5 file to load
    #     """
    #     try:
    #         # Update UI to show loading state
    #         self.update_status(f"Loading: {os.path.basename(file_path)}")

    #         # Store the current file path
    #         self.current_file_path = file_path

    #         # Update file info display
    #         self.update_file_info_display(file_path)

    #         # Check if this file is already loaded to avoid duplicates
    #         if hasattr(self, 'tree_data'):
    #             for i in range(self.tree_data.topLevelItemCount()):
    #                 existing_item = self.tree_data.topLevelItem(i)
    #                 existing_path = existing_item.data(0, Qt.UserRole + 1)
    #                 if existing_path == file_path:
    #                     # File already loaded, just select it and return
    #                     self.tree_data.setCurrentItem(existing_item)
    #                     self.update_status(f"File already loaded: {os.path.basename(file_path)}")
    #                     return

    #         # Clear any existing visualizations
    #         self.clear_2d_plot()
    #         self.clear_3d_plot()

    #         # Reset selected dataset path
    #         self.selected_dataset_path = None

    #         # Open and read HDF5 file
    #         with h5py.File(file_path, 'r') as h5file:
    #             # Create root item
    #             root_item = QTreeWidgetItem([os.path.basename(file_path)])
    #             root_item.setData(0, Qt.UserRole + 1, file_path)  # Store file path
    #             root_item.setData(0, Qt.UserRole + 2, "file_root")  # Mark as file root

    #             # Insert at the top (index 0) instead of adding to the end
    #             self.tree_data.insertTopLevelItem(0, root_item)

    #             # Recursively populate tree
    #             self._populate_tree_recursive(h5file, root_item)

    #             # Keep the root item collapsed by default
    #             root_item.setExpanded(False)

    #             # Select the newly added item
    #             self.tree_data.setCurrentItem(root_item)

    #         # Update workspace displays
    #         if hasattr(self, 'file_status_label'):
    #             self.file_status_label.setText(f"HDF5 file loaded: {os.path.basename(file_path)}")
    #         if hasattr(self, 'dataset_info_text'):
    #             self.dataset_info_text.setPlainText("Select a dataset from the tree to view detailed information.")

    #         self.update_status("HDF5 File Loaded Successfully")

    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Failed to load HDF5 file: {str(e)}")
    #         self.update_status("Failed to load file")

    def save_file_content(self, file_path):
        """
        Save analysis results.

        Args:
            file_path (str): Path to save the analysis to
        """
        try:
            self.update_status(f"Saving: {os.path.basename(file_path)}")

            # TODO: Implement actual save functionality
            # This would involve:
            # 1. Collecting current analysis state
            # 2. Saving results to HDF5 or other format
            # 3. Saving workspace configuration

            # Simulate save delay
            QTimer.singleShot(1000, lambda: self.update_status("Analysis Saved Successfully"))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save analysis: {str(e)}")
            self.update_status("Failed to save file")

    def load_folder_content(self, folder_path):
        """
        Load all HDF5 files from a folder and organize them under a folder section.

        Args:
            folder_path (str): Path to the folder containing HDF5 files
        """
        try:
            # Update UI to show loading state
            folder_name = os.path.basename(folder_path)
            self.update_status(f"Loading folder: {folder_name}")

            # Check if this folder is already loaded to avoid duplicates
            if hasattr(self, 'tree_data'):
                for i in range(self.tree_data.topLevelItemCount()):
                    existing_item = self.tree_data.topLevelItem(i)
                    existing_type = existing_item.data(0, Qt.UserRole + 2)
                    existing_path = existing_item.data(0, Qt.UserRole + 1)
                    if existing_type == "folder_section" and existing_path == folder_path:
                        # Folder already loaded, just select it and return
                        self.tree_data.setCurrentItem(existing_item)
                        self.update_status(f"Folder already loaded: {folder_name}")
                        return

            # Find all HDF5 and common image files in the folder
            h5_patterns = ['*.h5', '*.hdf5', '*.H5', '*.HDF5']
            img_patterns = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp',
                            '*.TIF', '*.TIFF', '*.PNG', '*.JPG', '*.JPEG', '*.BMP']
            h5_files = []
            img_files = []
            for pattern in h5_patterns:
                h5_files.extend(glob.glob(os.path.join(folder_path, pattern)))
            for pattern in img_patterns:
                img_files.extend(glob.glob(os.path.join(folder_path, pattern)))

            if not h5_files and not img_files:
                QMessageBox.information(self, "No Files", "No supported files found in the selected folder.")
                self.update_status("No supported files found")
                return

            # Sort files for consistent ordering
            h5_files.sort(); img_files.sort()

            # Create folder section header at the top
            total_files = len(h5_files) + len(img_files)
            folder_section_item = QTreeWidgetItem([f"📁 {folder_name} ({total_files} files)"])
            folder_section_item.setData(0, Qt.UserRole + 1, folder_path)  # Store folder path
            folder_section_item.setData(0, Qt.UserRole + 2, "folder_section")  # Mark as folder section

            # Insert at the top (index 0)
            self.tree_data.insertTopLevelItem(0, folder_section_item)

            # Load each supported file under the folder section
            loaded_count = 0
            for file_path in h5_files:
                try:
                    self.load_single_h5_file_under_section(file_path, folder_section_item)
                    loaded_count += 1
                except Exception as e:
                    self.update_status(f"Failed to load {file_path}: {e}")
                    continue
            for file_path in img_files:
                try:
                    self.add_image_file_under_section(file_path, folder_section_item)
                    loaded_count += 1
                except Exception as e:
                    self.update_status(f"Failed to add image {file_path}: {e}")
                    continue

            # Expand the folder section to show the files
            folder_section_item.setExpanded(True)

            # Select the folder section
            self.tree_data.setCurrentItem(folder_section_item)

            # Clear any existing visualizations
            self.clear_2d_plot()

            # Update workspace displays
            if hasattr(self, 'file_status_label'):
                self.file_status_label.setText(f"Folder loaded: {folder_name} ({loaded_count} files)")
            if hasattr(self, 'dataset_info_text'):
                self.dataset_info_text.setPlainText("Select a dataset from the tree to view detailed information.")

            self.update_status(f"Loaded {loaded_count} files from folder: {folder_name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load folder: {str(e)}")
            self.update_status("Failed to load folder")

    # === Load utilities ===
    def _populate_tree_recursive(self, h5_group, parent_item):
        """
        Recursively populate the tree widget with HDF5 structure.

        Args:
            h5_group: HDF5 group or file object
            parent_item: QTreeWidgetItem to add children to
        """
        for key in h5_group.keys():
            item = h5_group[key]

            # Create tree item
            tree_item = QTreeWidgetItem([key])
            parent_item.addChild(tree_item)

            # Store the full path as item data
            full_path = item.name
            tree_item.setData(0, 32, full_path)  # Qt.UserRole = 32

            if isinstance(item, h5py.Group):
                # It's a group, add group indicator and recurse
                tree_item.setText(0, f"{key} (Group)")
                self._populate_tree_recursive(item, tree_item)
            elif isinstance(item, h5py.Dataset):
                # It's a dataset, show shape and dtype info
                shape_str = f"{item.shape}" if item.shape else "scalar"
                dtype_str = str(item.dtype)
                tree_item.setText(0, f"{key} (Dataset: {shape_str}, {dtype_str})")
                # Color renderable datasets blue (similar to speckle_thing visual hint)
                if self.is_dataset_renderable(item):
                    tree_item.setForeground(0, QBrush(QColor('blue')))
                    tree_item.setData(0, Qt.UserRole + 3, True)

    def load_single_h5_file(self, file_path):
        """
        Load a single HDF5 file and add it to the tree.

        Args:
            file_path (str): Path to the HDF5 file
        """
        with h5py.File(file_path, 'r') as h5file:
            # Create root item for this file
            root_item = QTreeWidgetItem([os.path.basename(file_path)])
            root_item.setData(0, Qt.UserRole + 1, file_path)  # Store file path for removal
            root_item.setData(0, Qt.UserRole + 2, "file_root")  # Mark as file root
            self.tree_data.addTopLevelItem(root_item)

            # Recursively populate tree
            self._populate_tree_recursive(h5file, root_item)

            # Keep the root item collapsed by default
            root_item.setExpanded(False)

    def load_single_h5_file_under_section(self, file_path, parent_section):
        """
        Load a single HDF5 file and add it under a folder section.

        Args:
            file_path (str): Path to the HDF5 file
            parent_section (QTreeWidgetItem): Parent folder section item
        """
        with h5py.File(file_path, 'r') as h5file:
            # Create root item for this file under the section
            root_item = QTreeWidgetItem([os.path.basename(file_path)])
            root_item.setData(0, Qt.UserRole + 1, file_path)  # Store file path
            root_item.setData(0, Qt.UserRole + 2, "file_root")  # Mark as file root
            parent_section.addChild(root_item)

            # Recursively populate tree
            self._populate_tree_recursive(h5file, root_item)

            # Keep the root item collapsed by default
            root_item.setExpanded(False)

    def add_image_file_under_section(self, file_path, parent_section):
        """Add a single image file under a folder section in the data structure tree."""
        try:
            name = os.path.basename(file_path)
            root_item = QTreeWidgetItem([name])
            root_item.setData(0, Qt.UserRole + 1, file_path)
            root_item.setData(0, Qt.UserRole + 2, "file_root")
            # Optional child with basic info
            try:
                data = self._load_image_file(file_path)
            except Exception:
                data = None
            try:
                shape_str = ""
                if data is not None and hasattr(data, 'shape'):
                    shp = tuple(map(int, data.shape))
                    shape_str = f" (Shape: {shp})"
                child = QTreeWidgetItem([f"image{shape_str}"])
                child.setData(0, 32, "__image__")
                child.setData(0, Qt.UserRole + 3, True)
                root_item.addChild(child)
            except Exception:
                pass
            parent_section.addChild(root_item)
            root_item.setExpanded(False)
        except Exception:
            pass

    def is_dataset_renderable(self, dset):
        """Return True if dataset is numeric and can be rendered (2D/3D, or 1D perfect square)."""
        try:
            dtype = dset.dtype
            ndim = len(dset.shape)
            if np.issubdtype(dtype, np.number):
                if ndim >= 2:
                    return True
                if ndim == 1:
                    size = dset.size
                    if size >= 100:
                        side = int(np.sqrt(size))
                        return side * side == size
            return False
        except Exception:
            return False

    # Async dataset loader to prevent UI freeze
    class DatasetLoader(QObject):
        loaded = pyqtSignal(object)  # numpy array
        failed = pyqtSignal(str)

        def __init__(self, file_path, dataset_path, max_frames=100):
            super().__init__()
            self.file_path = file_path
            self.dataset_path = dataset_path
            self.max_frames = max_frames

        @pyqtSlot()
        def run(self):
            try:
                import h5py, numpy as np
                with h5py.File(self.file_path, 'r') as h5file:
                    if self.dataset_path not in h5file:
                        self.failed.emit("Dataset not found")
                        return
                    dset = h5file[self.dataset_path]
                    if not isinstance(dset, h5py.Dataset):
                        self.failed.emit("Selected item is not a dataset")
                        return

                    # Efficient loading to avoid blocking on huge datasets
                    if len(dset.shape) == 3:
                        max_frames = min(self.max_frames, dset.shape[0])
                        data = dset[:max_frames]
                    else:
                        # Guard against extremely large 2D datasets by center cropping
                        try:
                            estimated_size = dset.size * dset.dtype.itemsize
                        except Exception:
                            estimated_size = 0
                        if len(dset.shape) == 2 and estimated_size > 512 * 1024 * 1024:  # >512MB
                            h, w = dset.shape
                            ch = min(h, 2048)
                            cw = min(w, 2048)
                            y0 = max(0, (h - ch) // 2)
                            x0 = max(0, (w - cw) // 2)
                            data = dset[y0:y0+ch, x0:x0+cw]
                        else:
                            data = dset[...]

                    data = np.asarray(data, dtype=np.float32)
                    # Clean high values
                    high_mask = data > 5e6
                    if np.any(high_mask):
                        data[high_mask] = 0

                    # 1D handling: emit raw 1D data for dedicated 1D view
                    if data.ndim == 1:
                        # keep as 1D; no failure
                        pass

                    self.loaded.emit(data)
            except Exception as e:
                self.failed.emit(f"Error loading dataset: {e}")

    def load_dataset_robustly(self, dataset):
        """
        Load dataset with robust error handling and data cleaning like speckle_thing.py

        Args:
            dataset: h5py.Dataset object

        Returns:
            numpy.ndarray: Cleaned and processed data, or None if loading failed
        """
        try:
            self.update_status("Loading dataset...")

            # Load the data
            if len(dataset.shape) == 3:
                # For 3D datasets, load a reasonable number of frames (limit to 100 for memory)
                max_frames = min(100, dataset.shape[0])
                if max_frames < dataset.shape[0]:
                    self.update_status(f"Loading first {max_frames} frames of {dataset.shape[0]} total frames")
                data = dataset[:max_frames]
            else:
                # For 2D datasets, load all data
                data = dataset[...]

            # Convert to float32 for consistent processing
            data = np.asarray(data, dtype=np.float32)

            # Clean up data - set all values above 5e6 to zero (like speckle_thing.py)
            self.update_status("Cleaning data (setting values > 5e6 to zero)...")
            high_values_mask = data > 5e6
            if np.any(high_values_mask):
                num_cleaned = np.count_nonzero(high_values_mask)
                data[high_values_mask] = 0
                print(f"Cleaned {num_cleaned} pixels with values > 5e6")

            # Check for valid data
            if data.size == 0:
                self.update_status("Error: Dataset is empty")
                return None

            # Check for all-zero data
            if np.all(data == 0):
                self.update_status("Warning: All data values are zero")

            # 1D data: if perfect square and reasonably large, reshape to 2D; otherwise keep as 1D for 1D view
            if data.ndim == 1:
                side_length = int(np.sqrt(data.size))
                if data.size >= 100 and side_length * side_length == data.size:
                    data = data.reshape(side_length, side_length)
                    self.update_status(f"Reshaped 1D data to {side_length}x{side_length}")
                else:
                    self.update_status(f"Loaded 1D data of length {data.size}")
                    return data

            self.update_status(f"Successfully loaded data with shape {data.shape}")
            return data

        except Exception as e:
            error_msg = f"Error loading dataset robustly: {str(e)}"
            self.update_status(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None

    def visualize_selected_dataset(self):
        """Load and plot the selected dataset; clear/render ROIs depending on dataset type (image vs ROI)."""
        if not self.current_file_path or not self.selected_dataset_path:
            print("[DEBUG] visualize_selected_dataset: no current_file_path or selected_dataset_path")
            return
        try:
            print(f"[DEBUG] visualize_selected_dataset: selected_dataset_path={self.selected_dataset_path}")
            sel_path = str(self.selected_dataset_path)
            is_image_data = sel_path.endswith("/entry/data/data") or sel_path == "/entry/data/data" or sel_path.endswith("entry/data/data")
            is_roi_data = sel_path.startswith("/entry/data/rois") or "/entry/data/rois/" in sel_path

            # Always clear existing ROI graphics before switching
            try:
                self.roi_manager.clear_all_rois()
            except Exception:
                pass

            if is_image_data:
                # Load original image dataset via HDF5Loader (preferred)
                use_h5loader = True
                valid = self.h5loader.validate_file(self.current_file_path)
                print(f"[DEBUG] HDF5Loader.validate_file -> {valid}")
                if not valid:
                    self.update_status(f"HDF5 validation failed: {self.h5loader.get_last_error()}")
                    return
                volume, vol_shape = self.h5loader.load_h5_volume_3d(self.current_file_path)
                print(f"[DEBUG] HDF5Loader.load_h5_volume_3d shape={getattr(volume,'shape',None)}")
                if volume is None or volume.size == 0:
                    self.update_status("No data in /entry/data/data")
                    return
                data = volume
                # Display image data
                self.display_2d_data(data)
                if hasattr(self, 'tabWidget_analysis'):
                    self.tabWidget_analysis.setCurrentIndex(0)
                # Render ROIs associated with this dataset
                try:
                    self.roi_manager.render_rois_for_dataset(self.current_file_path, '/entry/data/data')
                except Exception:
                    pass
            elif is_roi_data:
                # When clicking on an ROI dataset, clear existing ROI boxes and render the ROI dataset itself as the image
                with h5py.File(self.current_file_path, 'r') as h5f:
                    exists = self.selected_dataset_path in h5f
                    print(f"[DEBUG] ROI dataset exists in file? {exists}")
                    if not exists:
                        self.update_status("ROI dataset not found")
                        return
                    dset = h5f[self.selected_dataset_path]
                    if not isinstance(dset, h5py.Dataset):
                        self.update_status("Selected ROI item is not a dataset")
                        return
                    data = np.asarray(dset[...], dtype=np.float32)
                # Display ROI-only data (2D or 3D with frames)
                if data.ndim >= 2 and np.issubdtype(data.dtype, np.number):
                    self.display_2d_data(data)
                    if hasattr(self, 'tabWidget_analysis'):
                        self.tabWidget_analysis.setCurrentIndex(0)
                    # No ROI overlays when viewing ROI-only dataset
                    self.update_status(f"Loaded ROI dataset: {self.selected_dataset_path}")
                else:
                    # Non-visualizable
                    self.clear_2d_plot()
                    self.update_status("ROI dataset loaded but not visualizable")
            else:
                # Fallback: open generic dataset directly
                with h5py.File(self.current_file_path, 'r') as h5file:
                    exists = self.selected_dataset_path in h5file
                    print(f"[DEBUG] Dataset exists in file? {exists}")
                    if not exists:
                        self.update_status("Dataset not found")
                        return
                    dataset = h5file[self.selected_dataset_path]
                    print(f"[DEBUG] Dataset type={type(dataset)} shape={getattr(dataset,'shape',None)}")
                    if not isinstance(dataset, h5py.Dataset):
                        self.update_status("Selected item is not a dataset")
                        return
                    data = self.load_dataset_robustly(dataset)
                    print(f"[DEBUG] load_dataset_robustly returned shape={getattr(data,'shape',None)}")
                    if data is None:
                        return
                if data.ndim >= 2 and np.issubdtype(data.dtype, np.number):
                    self.display_2d_data(data)
                    if hasattr(self, 'tabWidget_analysis'):
                        self.tabWidget_analysis.setCurrentIndex(0)
                    # Render ROIs for this dataset if any
                    try:
                        self.roi_manager.render_rois_for_dataset(self.current_file_path, self.selected_dataset_path)
                    except Exception:
                        pass
                elif data.ndim == 1:
                    self.display_1d_data(data)
                    self.update_status("Loaded 1D dataset")
                else:
                    self.clear_2d_plot()
                    self.update_status("Dataset loaded but not visualizable")
        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}"
            if hasattr(self, 'dataset_info_text'):
                self.dataset_info_text.setPlainText(error_msg)
            self.update_status(error_msg)

    def start_dataset_load(self):
        """Create a worker thread to load dataset without blocking the UI."""
        try:
            self.update_status(f"Loading dataset: {self.selected_dataset_path}")
            self._dataset_thread = QThread()
            self._dataset_worker = self.DatasetLoader(self.current_file_path, self.selected_dataset_path)
            self._dataset_worker.moveToThread(self._dataset_thread)
            self._dataset_thread.started.connect(self._dataset_worker.run)
            self._dataset_worker.loaded.connect(self.on_dataset_loaded)
            self._dataset_worker.failed.connect(self.on_dataset_failed)
            # Ensure thread quits after work
            self._dataset_worker.loaded.connect(self._dataset_thread.quit)
            self._dataset_worker.failed.connect(self._dataset_thread.quit)
            self._dataset_thread.start()
        except Exception as e:
            self.update_status(f"Error starting dataset load: {e}")

    @pyqtSlot(object)
    def on_dataset_loaded(self, data):
        """Handle dataset loaded event on main thread."""
        try:
            # Visualize data
            if data is None:
                self.update_status("Loaded empty dataset")
                return
            if data.ndim >= 2 and np.issubdtype(data.dtype, np.number):
                self.display_2d_data(data)
                if hasattr(self, 'tabWidget_analysis'):
                    self.tabWidget_analysis.setCurrentIndex(0)
                # Build info
                info_lines = []
                info_lines.append(f"Dataset: {self.selected_dataset_path}")
                # Read original shape/dtype quickly
                try:
                    with h5py.File(self.current_file_path, 'r') as h5file:
                        dset = h5file[self.selected_dataset_path]
                        info_lines.append(f"Original Shape: {dset.shape}")
                        info_lines.append(f"Original Type: {dset.dtype}")
                except Exception:
                    pass
                info_lines.append(f"Loaded Shape: {data.shape}")
                info_lines.append(f"Data Type: {data.dtype}")
                info_lines.append(f"Size: {data.size:,} elements")
                info_lines.append("\nData Statistics:")
                info_lines.append(f"Min: {np.min(data):.6f}")
                info_lines.append(f"Max: {np.max(data):.6f}")
                info_lines.append(f"Mean: {np.mean(data):.6f}")
                info_lines.append(f"Std: {np.std(data):.6f}")
                # Memory usage
                mem_size = data.size * data.dtype.itemsize
                if mem_size < 1024:
                    mem_str = f"{mem_size} bytes"
                elif mem_size < 1024 * 1024:
                    mem_str = f"{mem_size / 1024:.1f} KB"
                elif mem_size < 1024 * 1024 * 1024:
                    mem_str = f"{mem_size / (1024 * 1024):.1f} MB"
                else:
                    mem_str = f"{mem_size / (1024 * 1024 * 1024):.1f} GB"
                info_lines.append(f"\nMemory Usage: {mem_str}")
                info_text = "\n".join(info_lines)
                if hasattr(self, 'dataset_info_text'):
                    self.dataset_info_text.setPlainText(info_text)
                if hasattr(self, 'file_status_label'):
                    self.file_status_label.setText(f"Loaded: {os.path.basename(self.selected_dataset_path)}")
                self.update_status(f"Loaded dataset: {self.selected_dataset_path}")
                try:
                    if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                        self.info_2d_dock.refresh()
                except Exception:
                    pass
            else:
                if data.ndim == 1:
                    self.display_1d_data(data)
                    self.update_status("Loaded 1D dataset")
                    try:
                        if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                            self.info_2d_dock.refresh()
                    except Exception:
                        pass
                else:
                    self.clear_2d_plot()
                    self.update_status("Dataset loaded but not visualizable")
        except Exception as e:
            self.update_status(f"Error handling loaded dataset: {e}")

    @pyqtSlot(str)
    def on_dataset_failed(self, message):
        """Handle dataset load failure."""
        try:
            if hasattr(self, 'dataset_info_text'):
                self.dataset_info_text.setPlainText(message)
            self.update_status(message)
            try:
                if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                    self.info_2d_dock.refresh()
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error updating failure status: {e}")

    def load_3d_data(self):
        """Delegate 3D data loading to Tab3D."""
        try:
            if hasattr(self, 'tab_3d') and self.tab_3d is not None:
                self.tab_3d.load_data()
        except Exception as e:
            self.update_status(f"Error loading 3D data: {e}")

    # === 2D Helpers ===
    def set_2d_axes(self, x_axis, y_axis):
        try:
            self.axis_2d_x = str(x_axis) if x_axis else None
            self.axis_2d_y = str(y_axis) if y_axis else None
            if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                try:
                    self.info_2d_dock.refresh()
                except Exception:
                    pass
        except Exception:
            pass
        
    def clear_2d_plot(self):
        """Clear the 2D plot and show placeholder."""
        try:
            if hasattr(self, 'image_view'):
                # Create a small placeholder image
                placeholder = np.zeros((100, 100), dtype=np.float32)
                self.image_view.setImage(placeholder, autoLevels=False, autoRange=True)

                # Remove any existing ROIs
                if hasattr(self, 'rois') and isinstance(self.rois, list):
                    for roi in self.rois:
                        try:
                            self.image_view.removeItem(roi)
                        except Exception:
                            pass
                    self.rois.clear()
                self.current_roi = None
                # Clear docked ROI list
                if hasattr(self, 'roi_list') and self.roi_list is not None:
                    try:
                        self.roi_list.clear()
                        self.roi_by_item = {}
                        self.item_by_roi_id = {}
                    except Exception:
                        pass
                # Clear ROI stats dock
                if hasattr(self, 'roi_stats_table') and self.roi_stats_table is not None:
                    try:
                        self.roi_stats_table.setRowCount(0)
                        self.stats_row_by_roi_id = {}
                    except Exception:
                        pass

                # Set default axis labels
                self.plot_item.setLabel('bottom', 'X')
                self.plot_item.setLabel('left', 'Y')
                try:
                    self.set_2d_axes("Columns", "Row")
                except Exception:
                    pass
                try:
                    if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                        self.info_2d_dock.refresh()
                except Exception:
                    pass

                # Update above-image info label with placeholder dimensions
                if hasattr(self, 'image_info_label'):
                    try:
                        self.image_info_label.setText("Image Dimensions: 100x100 pixels")
                    except Exception:
                        pass

                # Remove hover overlays and clear HKL caches
                try:
                    view = self.image_view.getView() if hasattr(self.image_view, 'getView') else None
                    if view is not None:
                        if hasattr(self, '_hover_hline') and self._hover_hline is not None:
                            try:
                                view.removeItem(self._hover_hline)
                            except Exception:
                                pass
                            self._hover_hline = None
                        if hasattr(self, '_hover_vline') and self._hover_vline is not None:
                            try:
                                view.removeItem(self._hover_vline)
                            except Exception:
                                pass
                            self._hover_vline = None
                        if hasattr(self, '_hover_text') and self._hover_text is not None:
                            try:
                                view.removeItem(self._hover_text)
                            except Exception:
                                pass
                            self._hover_text = None
                    self._mouse_proxy = None
                    self._qx_grid = None
                    self._qy_grid = None
                    self._qz_grid = None
                except Exception:
                    pass

        except Exception as e:
            self.update_status(f"Error clearing 2D plot: {e}")

    def update_overlay_text(self, width, height, frame_info=None):
        """Update the label above the image with dimensions and optional frame info.
        Augmented to append current motor position (if available) for the selected frame.
        """
        try:
            text = f"Image Dimensions: {width}x{height} pixels"
            info = frame_info or ""
            # Try to append motor position for current frame if 3D data
            try:
                if hasattr(self, 'current_2d_data') and self.current_2d_data is not None and self.current_2d_data.ndim == 3:
                    num_frames = int(self.current_2d_data.shape[0])
                    idx = 0
                    if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                        try:
                            idx = int(self.frame_spinbox.value())
                        except Exception:
                            idx = 0
                    motor_val = None
                    fp = getattr(self, 'current_file_path', None)
                    if fp and os.path.exists(fp):
                        try:
                            with h5py.File(fp, 'r') as h5f:
                                arr = self._find_motor_positions(h5f, num_frames)
                                if arr is not None and 0 <= idx < arr.size:
                                    motor_val = float(arr[idx])
                        except Exception:
                            motor_val = None
                    if motor_val is not None:
                        if info:
                            info = f"{info} | Motor {motor_val:.6f}"
                        else:
                            info = f"Motor {motor_val:.6f}"
            except Exception:
                pass
            if info:
                text = f"{text} ({info})"
            if hasattr(self, 'image_info_label'):
                self.image_info_label.setText(text)
        except Exception as e:
            self.update_status(f"Error updating image info label: {e}")

    def display_2d_data(self, data):
        """Display 2D or 3D numeric data in the PyQtGraph ImageView."""
        try:
            if not hasattr(self, 'image_view'):
                print("Warning: ImageView not initialized")
                return

            # Store the original data for frame navigation
            self.current_2d_data = data
            try:
                print(f"[DISPLAY] data ndim={getattr(data,'ndim',None)}, shape={getattr(data,'shape',None)}")
            except Exception:
                pass

            # Handle different data dimensions
            if data.ndim == 2:
                # 2D data - display directly
                image_data = np.asarray(data, dtype=np.float32)

                # Update frame controls for 2D data
                self.update_frame_controls_for_2d_data()

                height, width = image_data.shape
                if hasattr(self, 'frame_info_label'):
                    self.frame_info_label.setText(f"Image Dimensions: {width}x{height} pixels")
                # Update overlay text
                self.update_overlay_text(width, height, None)

            elif data.ndim == 3:
                # 3D data - display first frame and set up navigation
                image_data = np.asarray(data[0], dtype=np.float32)

                # Update frame controls for 3D data
                num_frames = data.shape[0]
                self.update_frame_controls_for_3d_data(num_frames)

                height, width = image_data.shape
                if hasattr(self, 'frame_info_label'):
                    self.frame_info_label.setText(f"Image Dimensions: {width}x{height} pixels (frame 0 of {num_frames})")
                # Update overlay text
                self.update_overlay_text(width, height, f"Frame 0 of {num_frames}")

            else:
                print(f"Unsupported data dimensions: {data.ndim}")
                return

            # Set the image data
            auto_levels = hasattr(self, 'cbAutoLevels') and self.cbAutoLevels.isChecked()
            self.image_view.setImage(
                image_data,
                autoLevels=auto_levels,
                autoRange=True,
                autoHistogramRange=auto_levels
            )
            # Ensure hover overlays exist after any prior clear
            try:
                self._setup_2d_hover()
            except Exception:
                pass

            # Update axis labels based on data shape
            height, width = image_data.shape
            self.plot_item.setLabel('bottom', f'Columns [pixels] (0 to {width-1})')
            self.plot_item.setLabel('left', f'Row [pixels] (0 to {height-1})')
            try:
                self.set_2d_axes("Columns", "Row")
            except Exception:
                pass

            # Apply current colormap
            if hasattr(self, 'cbColorMapSelect_2d'):
                current_colormap = self.cbColorMapSelect_2d.currentText()
                self.apply_colormap(current_colormap)

            # Update speckle analysis controls programmatically
            self.update_speckle_controls_for_data(data)

            # Update vmin/vmax controls based on data
            self.update_vmin_vmax_controls_for_data(image_data)

            # Refresh ROI stats for current frame/data
            try:
                self.roi_manager.update_all_roi_stats()
            except Exception:
                pass
            try:
                if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                    self.info_2d_dock.refresh()
            except Exception:
                pass
            # Refresh any open ROI Plot docks to reflect dataset change (axes and series)
            try:
                docks = []
                if hasattr(self, '_roi_plot_dock_widgets') and self._roi_plot_dock_widgets:
                    docks.extend(list(self._roi_plot_dock_widgets))
                if hasattr(self, 'roi_plot_docks_by_roi_id') and self.roi_plot_docks_by_roi_id:
                    for lst in self.roi_plot_docks_by_roi_id.values():
                        docks.extend(list(lst))
                for d in docks:
                    try:
                        if hasattr(d, 'refresh_for_dataset_change'):
                            d.refresh_for_dataset_change()
                    except Exception:
                        continue
            except Exception:
                pass

        except Exception as e:
            self.update_status(f"Error displaying 2D data: {e}")

    def _show_image_context_menu(self, pos):
        """Show right-click menu for the 2D image with hover toggle and HKL plotting."""
        try:
            menu = QMenu(self)
            # Enable/Disable Hover
            action_hover = QAction("Enable Hover", self)
            action_hover.setCheckable(True)
            action_hover.setChecked(bool(getattr(self, '_hover_enabled', True)))
            action_hover.toggled.connect(self._toggle_hover_enabled)
            menu.addAction(action_hover)
            # Show current hover state explicitly (do not remove original options)
            try:
                state_text = "Hover: ON" if bool(getattr(self, '_hover_enabled', True)) else "Hover: OFF"
            except Exception:
                state_text = "Hover: ON"
            action_state = QAction(state_text, self)
            action_state.setEnabled(False)
            menu.addAction(action_state)
            # Show last HKL value if available (disabled info item)
            hkl_label = "HKL: N/A"
            try:
                xy = getattr(self, '_last_hover_xy', None)
                qxg = getattr(self, '_qx_grid', None)
                qyg = getattr(self, '_qy_grid', None)
                qzg = getattr(self, '_qz_grid', None)
                if xy and qxg is not None and qyg is not None and qzg is not None:
                    x, y = int(xy[0]), int(xy[1])
                    if qxg.ndim == 3 and qyg.ndim == 3 and qzg.ndim == 3:
                        idx = 0
                        try:
                            idx = int(self.frame_spinbox.value()) if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled() else 0
                        except Exception:
                            idx = 0
                        if 0 <= idx < qxg.shape[0]:
                            H = float(qxg[idx, y, x]); K = float(qyg[idx, y, x]); L = float(qzg[idx, y, x])
                            hkl_label = f"HKL: H={H:.6f}, K={K:.6f}, L={L:.6f}"
                    elif qxg.ndim == 2 and qyg.ndim == 2 and qzg.ndim == 2:
                        H = float(qxg[y, x]); K = float(qyg[y, x]); L = float(qzg[y, x])
                        hkl_label = f"HKL: H={H:.6f}, K={K:.6f}, L={L:.6f}"
            except Exception:
                pass
            action_hkl_info = QAction(hkl_label, self)
            action_hkl_info.setEnabled(False)
            menu.addAction(action_hkl_info)
            # Plot HKL in 3D
            action_plot_hkl = QAction("Plot HKL (3D)", self)
            action_plot_hkl.setToolTip("Plot current frame intensity at HKL (qx,qy,qz) points")
            action_plot_hkl.triggered.connect(self._plot_current_hkl_points)
            menu.addAction(action_plot_hkl)
            # Show menu at global position
            try:
                gpos = self.image_view.mapToGlobal(pos)
            except Exception:
                gpos = QCursor.pos()
            menu.exec_(gpos)
        except Exception as e:
            self.update_status(f"Error showing image context menu: {e}")

    def _toggle_hover_enabled(self, enabled: bool):
        try:
            self._hover_enabled = bool(enabled)
            self._update_hover_visibility()
        except Exception:
            pass

    def _update_hover_visibility(self):
        try:
            visible = bool(getattr(self, '_hover_enabled', True))
            for item_name in ['_hover_hline', '_hover_vline', '_hover_text']:
                it = getattr(self, item_name, None)
                try:
                    if it is not None:
                        it.setVisible(visible)
                except Exception:
                    pass
        except Exception:
            pass

    def _update_hover_text_at(self, x: int, y: int):
        """Update hover crosshair and tooltip for given pixel coordinates on current frame."""
        try:
            frame = self.get_current_frame_data()
            if frame is None or frame.ndim != 2:
                return
            height, width = frame.shape
            if x < 0 or y < 0 or x >= width or y >= height:
                return
            # Update crosshair positions
            try:
                if hasattr(self, '_hover_hline') and self._hover_hline is not None:
                    self._hover_hline.setPos(float(y))
                if hasattr(self, '_hover_vline') and self._hover_vline is not None:
                    self._hover_vline.setPos(float(x))
            except Exception:
                pass
            # Intensity
            try:
                intensity = float(frame[x, y])
            except Exception:
                intensity = float('nan')
            # HKL text
            hkl_str = ""
            try:
                qxg = getattr(self, '_qx_grid', None)
                qyg = getattr(self, '_qy_grid', None)
                qzg = getattr(self, '_qz_grid', None)
                if qxg is not None and qyg is not None and qzg is not None:
                    if qxg.ndim == 3 and qyg.ndim == 3 and qzg.ndim == 3:
                        idx = 0
                        try:
                            idx = int(self.frame_spinbox.value()) if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled() else 0
                        except Exception:
                            idx = 0
                        if 0 <= idx < qxg.shape[0]:
                            H = float(qxg[idx, y, x]); K = float(qyg[idx, y, x]); L = float(qzg[idx, y, x])
                            hkl_str = f" | H={H:.6f}, K={K:.6f}, L={L:.6f}"
                    elif qxg.ndim == 2 and qyg.ndim == 2 and qzg.ndim == 2:
                        H = float(qxg[y, x]); K = float(qyg[y, x]); L = float(qzg[y, x])
                        hkl_str = f" | H={H:.6f}, K={K:.6f}, L={L:.6f}"
            except Exception:
                hkl_str = ""
            # Tooltip text removed; keep crosshair only
            try:
                if hasattr(self, '_hover_text') and self._hover_text is not None:
                    self._hover_text.setVisible(False)
            except Exception:
                pass
            # Update 2D Info dock Mouse section even during playback
            try:
                if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                    H_val = K_val = L_val = None
                    try:
                        qxg = getattr(self, '_qx_grid', None)
                        qyg = getattr(self, '_qy_grid', None)
                        qzg = getattr(self, '_qz_grid', None)
                        if qxg is not None and qyg is not None and qzg is not None:
                            if qxg.ndim == 3 and qyg.ndim == 3 and qzg.ndim == 3:
                                idx = int(self.frame_spinbox.value()) if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled() else 0
                                if 0 <= idx < qxg.shape[0]:
                                    H_val = float(qxg[idx, y, x]); K_val = float(qyg[idx, y, x]); L_val = float(qzg[idx, y, x])
                            elif qxg.ndim == 2 and qyg.ndim == 2 and qzg.ndim == 2:
                                H_val = float(qxg[y, x]); K_val = float(qyg[y, x]); L_val = float(qzg[y, x])
                    except Exception:
                        H_val = K_val = L_val = None
                    self.info_2d_dock.set_mouse_info((x, y), intensity, H_val, K_val, L_val)
            except Exception:
                pass
        except Exception:
            pass

    def _plot_current_hkl_points(self):
        """Plot current frame intensities at HKL positions in an HKL 3D Plot Dock."""
        try:
            # Ensure q-grids are available
            if getattr(self, '_qx_grid', None) is None or getattr(self, '_qy_grid', None) is None or getattr(self, '_qz_grid', None) is None:
                try:
                    self._try_load_hkl_grids()
                except Exception:
                    pass
            qxg = getattr(self, '_qx_grid', None)
            qyg = getattr(self, '_qy_grid', None)
            qzg = getattr(self, '_qz_grid', None)
            frame = self.get_current_frame_data()
            if qxg is None or qyg is None or qzg is None or frame is None:
                self.update_status("HKL grids or frame not available for plotting")
                return
            # Select frame index if 3D
            idx = 0
            try:
                idx = int(self.frame_spinbox.value()) if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled() else 0
            except Exception:
                idx = 0
            # Extract H,K,L arrays matching frame
            try:
                if qxg.ndim == 3:
                    qx = qxg[idx]; qy = qyg[idx]; qz = qzg[idx]
                else:
                    qx = qxg; qy = qyg; qz = qzg
            except Exception:
                self.update_status("Error extracting HKL arrays for current frame")
                return
            # Build points and intensities
            try:
                H = np.asarray(qx, dtype=np.float32).ravel()
                K = np.asarray(qy, dtype=np.float32).ravel()
                L = np.asarray(qz, dtype=np.float32).ravel()
                points = np.column_stack([H, K, L])
                intens = np.asarray(frame, dtype=np.float32).ravel()
            except Exception:
                self.update_status("Error building HKL points")
                return
            # Create or reuse HKL 3D Plot Dock
            try:
                from viewer.workbench.hkl_3d_plot_dock import HKL3DPlotDock
            except Exception:
                HKL3DPlotDock = None
            if HKL3DPlotDock is None:
                self.update_status("HKL3DPlotDock not available")
                return
            if not hasattr(self, '_hkl3d_plot_dock') or self._hkl3d_plot_dock is None:
                dock_title = "HKL 3D Plot"
                dock = HKL3DPlotDock(self, dock_title, self)
                self.addDockWidget(Qt.RightDockWidgetArea, dock)
                try:
                    self.add_dock_toggle_action(dock, dock_title, segment_name="2d")
                except Exception:
                    pass
                dock.show()
                self._hkl3d_plot_dock = dock
            # Plot points
            try:
                self._hkl3d_plot_dock._plot_points(points, intens)
                self.update_status("Plotted HKL points for current frame")
            except Exception as e:
                self.update_status(f"Error plotting HKL points: {e}")
        except Exception as e:
            self.update_status(f"Error in HKL plot: {e}")

    def _update_hkl3d_plot_for_current_frame(self):
        """If HKL 3D plot dock is open, update it to current frame."""
        try:
            if hasattr(self, '_hkl3d_plot_dock') and self._hkl3d_plot_dock is not None:
                self._plot_current_hkl_points()
        except Exception:
            pass

    def _setup_2d_hover(self):
        """Create crosshair and tooltip overlays, and connect mouse move events via SignalProxy."""
        try:
            if not hasattr(self, 'image_view') or self.image_view is None:
                return
            view = self.image_view.getView() if hasattr(self.image_view, 'getView') else None
            if view is None:
                return
            # Create overlays only once
            if not hasattr(self, '_hover_hline') or self._hover_hline is None:
                try:
                    self._hover_hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=(255, 255, 0, 150), width=1))
                    self.plot_item.addItem(self._hover_hline)
                    try:
                        self._hover_hline.setZValue(1000)
                    except Exception:
                        pass
                except Exception:
                    self._hover_hline = None
            if not hasattr(self, '_hover_vline') or self._hover_vline is None:
                try:
                    self._hover_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(255, 255, 0, 150), width=1))
                    self.plot_item.addItem(self._hover_vline)
                    try:
                        self._hover_vline.setZValue(1000)
                    except Exception:
                        pass
                except Exception:
                    self._hover_vline = None
            if not hasattr(self, '_hover_text') or self._hover_text is None:
                try:
                    self._hover_text = pg.TextItem("", color=(255, 255, 255))
                    try:
                        self._hover_text.setAnchor((0, 1))
                    except Exception:
                        pass
                    self.plot_item.addItem(self._hover_text)
                    try:
                        self._hover_text.setZValue(1000)
                    except Exception:
                        pass
                except Exception:
                    self._hover_text = None
            # Connect mouse move via SignalProxy to throttle updates
            try:
                vb = getattr(self.plot_item, 'vb', None)
                scene = vb.scene() if vb is not None else self.plot_item.scene()
                self._mouse_proxy = pg.SignalProxy(scene.sigMouseMoved, rateLimit=60, slot=self._on_2d_mouse_moved)
            except Exception:
                self._mouse_proxy = None
        except Exception as e:
            try:
                self.update_status(f"Error setting up 2D hover: {e}")
            except Exception:
                pass

    def _on_2d_mouse_moved(self, evt):
        """Map scene coordinates to pixel indices; update crosshair and tooltip with intensity and HKL if available."""
        try:
            # evt may be (QPointF,) from SignalProxy
            pos = evt[0] if isinstance(evt, (tuple, list)) and len(evt) > 0 else evt
            if not hasattr(self, 'image_view') or self.image_view is None:
                return
            view = self.image_view.getView() if hasattr(self.image_view, 'getView') else None
            image_item = getattr(self.image_view, 'imageItem', None)
            if view is None or image_item is None:
                return
            # Map to data coordinates
            try:
                vb = getattr(self.plot_item, 'vb', None)
                if vb is not None:
                    mouse_point = vb.mapSceneToView(pos)
                else:
                    mouse_point = view.mapSceneToView(pos)
            except Exception:
                return
            # Respect hover enabled flag
            if not bool(getattr(self, '_hover_enabled', True)):
                return
            x = int(round(float(mouse_point.x())))
            y = int(round(float(mouse_point.y())))
            frame = self.get_current_frame_data()
            if frame is None or frame.ndim != 2:
                return
            height, width = frame.shape
            # Move crosshairs regardless, using float positions
            try:
                if hasattr(self, '_hover_hline') and self._hover_hline is not None:
                    self._hover_hline.setPos(mouse_point.y())
                if hasattr(self, '_hover_vline') and self._hover_vline is not None:
                    self._hover_vline.setPos(mouse_point.x())
            except Exception:
                pass
            if x < 0 or y < 0 or x >= width or y >= height:
                return
            # Remember last valid hover position
            try:
                self._last_hover_xy = (x, y)
            except Exception:
                pass
            # Intensity at pixel
            try:
                intensity = float(frame[x, y])
            except Exception:
                intensity = float('nan')
            # HKL from cached q-grids if present
            hkl_str = ""
            try:
                qxg = getattr(self, '_qx_grid', None)
                qyg = getattr(self, '_qy_grid', None)
                qzg = getattr(self, '_qz_grid', None)
                if qxg is not None and qyg is not None and qzg is not None:
                    if qxg.ndim == 3 and qyg.ndim == 3 and qzg.ndim == 3:
                        idx = 0
                        try:
                            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                                idx = int(self.frame_spinbox.value())
                        except Exception:
                            idx = 0
                        if 0 <= idx < qxg.shape[0]:
                            H = float(qxg[idx, y, x])
                            K = float(qyg[idx, y, x])
                            L = float(qzg[idx, y, x])
                            hkl_str = f" | H={H:.6f}, K={K:.6f}, L={L:.6f}"
                    elif qxg.ndim == 2 and qyg.ndim == 2 and qzg.ndim == 2:
                        H = float(qxg[y, x])
                        K = float(qyg[y, x])
                        L = float(qzg[y, x])
                        hkl_str = f" | H={H:.6f}, K={K:.6f}, L={L:.6f}"
            except Exception:
                hkl_str = ""
            # Update tooltip text near cursor
            try:
                if hasattr(self, '_hover_text') and self._hover_text is not None:
                    # Hide hover text; keep crosshair only
                    self._hover_text.setVisible(False)
                    # Update 2D Info dock Mouse section
                    try:
                        if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                            # Derive H,K,L values again here for precision
                            H_val = K_val = L_val = None
                            try:
                                qxg = getattr(self, '_qx_grid', None)
                                qyg = getattr(self, '_qy_grid', None)
                                qzg = getattr(self, '_qz_grid', None)
                                if qxg is not None and qyg is not None and qzg is not None:
                                    if qxg.ndim == 3 and qyg.ndim == 3 and qzg.ndim == 3:
                                        idx = int(self.frame_spinbox.value()) if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled() else 0
                                        if 0 <= idx < qxg.shape[0]:
                                            H_val = float(qxg[idx, y, x]); K_val = float(qyg[idx, y, x]); L_val = float(qzg[idx, y, x])
                                    elif qxg.ndim == 2 and qyg.ndim == 2 and qzg.ndim == 2:
                                        H_val = float(qxg[y, x]); K_val = float(qyg[y, x]); L_val = float(qzg[y, x])
                            except Exception:
                                H_val = K_val = L_val = None
                            self.info_2d_dock.set_mouse_info((x, y), intensity, H_val, K_val, L_val)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def _try_load_hkl_grids(self):
        """Load and cache qx/qy/qz grids (supports 2D HxW and 3D FxHxW). Called after display_2d_data."""
        try:
            # Reset caches by default
            self._qx_grid = None
            self._qy_grid = None
            self._qz_grid = None
            if not getattr(self, 'current_file_path', None) or not getattr(self, 'selected_dataset_path', None):
                return
            with h5py.File(self.current_file_path, 'r') as h5f:
                sel_path = str(self.selected_dataset_path)
                parent_path = sel_path.rsplit('/', 1)[0] if '/' in sel_path else '/'
                candidates = []
                try:
                    if parent_path in h5f:
                        candidates.append(h5f[parent_path])
                except Exception:
                    pass
                try:
                    if '/entry/data' in h5f:
                        candidates.append(h5f['/entry/data'])
                except Exception:
                    pass
                qx = qy = qz = None
                def find_in_group(g, name):
                    for key in g.keys():
                        try:
                            if isinstance(g[key], h5py.Dataset) and key.lower() == name:
                                return g[key]
                        except Exception:
                            pass
                    return None
                # Try strict names first
                for g in candidates:
                    if g is None:
                        continue
                    try:
                        qx = find_in_group(g, 'qx')
                        qy = find_in_group(g, 'qy')
                        qz = find_in_group(g, 'qz')
                    except Exception:
                        qx = qy = qz = None
                    if qx is not None and qy is not None and qz is not None:
                        break
                # Fallback: case-insensitive suffix match within parent group
                if (qx is None or qy is None or qz is None) and parent_path in h5f:
                    g = h5f[parent_path]
                    for key in g.keys():
                        try:
                            if not isinstance(g[key], h5py.Dataset):
                                continue
                        except Exception:
                            continue
                        lk = key.lower()
                        # Support additional naming conventions for HKL/q grids
                        if lk.endswith('qx') or lk == 'qx' or lk in ('q_x', 'qx_grid', 'qgrid_x', 'h', 'QX'.lower()):
                            qx = g[key]
                        elif lk.endswith('qy') or lk == 'qy' or lk in ('q_y', 'qy_grid', 'qgrid_y', 'k', 'QY'.lower()):
                            qy = g[key]
                        elif lk.endswith('qz') or lk == 'qz' or lk in ('q_z', 'qz_grid', 'qgrid_z', 'l', 'QZ'.lower()):
                            qz = g[key]
                # Last resort: search entire file for datasets named like qx/qy/qz
                if qx is None or qy is None or qz is None:
                    for group in [h5f]:
                        for key in group.keys():
                            try:
                                item = group[key]
                                if not isinstance(item, h5py.Dataset):
                                    continue
                                lk = key.lower()
                                if qx is None and (lk.endswith('qx') or lk == 'qx' or lk in ('q_x', 'h')):
                                    qx = item
                                elif qy is None and (lk.endswith('qy') or lk == 'qy' or lk in ('q_y', 'k')):
                                    qy = item
                                elif qz is None and (lk.endswith('qz') or lk == 'qz' or lk in ('q_z', 'l')):
                                    qz = item
                            except Exception:
                                continue
                if qx is None or qy is None or qz is None:
                    return
                # Read arrays
                try:
                    qx_arr = np.asarray(qx[...], dtype=np.float32)
                    qy_arr = np.asarray(qy[...], dtype=np.float32)
                    qz_arr = np.asarray(qz[...], dtype=np.float32)
                except Exception:
                    return
                frame = self.get_current_frame_data()
                if frame is None or frame.ndim != 2:
                    return
                h, w = frame.shape
                # Normalize shapes: transpose 2D grids if (w,h)
                if qx_arr.ndim == 2 and qy_arr.ndim == 2 and qz_arr.ndim == 2:
                    if qx_arr.shape == (w, h) and qy_arr.shape == (w, h) and qz_arr.shape == (w, h):
                        try:
                            qx_arr = qx_arr.T; qy_arr = qy_arr.T; qz_arr = qz_arr.T
                        except Exception:
                            pass
                    if qx_arr.shape == (h, w) and qy_arr.shape == (h, w) and qz_arr.shape == (h, w):
                        self._qx_grid = qx_arr
                        self._qy_grid = qy_arr
                        self._qz_grid = qz_arr
                    else:
                        return
                elif qx_arr.ndim == 3 and qy_arr.ndim == 3 and qz_arr.ndim == 3:
                    # Expect (F, H, W), but reorder axes if needed
                    def reorder_to_fhw(arr, h, w):
                        try:
                            shp = arr.shape
                            if len(shp) != 3:
                                return None
                            # Identify axes matching h and w
                            idx_h = None; idx_w = None
                            for i, d in enumerate(shp):
                                if d == h and idx_h is None:
                                    idx_h = i
                            for i, d in enumerate(shp):
                                if d == w and i != idx_h and idx_w is None:
                                    idx_w = i
                            if idx_h is None or idx_w is None:
                                return None
                            idx_f = [0, 1, 2]
                            idx_f.remove(idx_h); idx_f.remove(idx_w)
                            idx_f = idx_f[0]
                            order = [idx_f, idx_h, idx_w]
                            return np.transpose(arr, axes=order)
                        except Exception:
                            return None
                    if not (qx_arr.shape[1:] == (h, w) and qy_arr.shape[1:] == (h, w) and qz_arr.shape[1:] == (h, w)):
                        rqx = reorder_to_fhw(qx_arr, h, w)
                        rqy = reorder_to_fhw(qy_arr, h, w)
                        rqz = reorder_to_fhw(qz_arr, h, w)
                        if rqx is not None and rqy is not None and rqz is not None:
                            qx_arr, qy_arr, qz_arr = rqx, rqy, rqz
                    if qx_arr.shape[1:] == (h, w) and qy_arr.shape[1:] == (h, w) and qz_arr.shape[1:] == (h, w):
                        self._qx_grid = qx_arr
                        self._qy_grid = qy_arr
                        self._qz_grid = qz_arr
                    else:
                        return
                else:
                    return
                try:
                    self.update_status("HKL q-grids loaded for hover")
                except Exception:
                    pass
                try:
                    self.set_2d_axes("h", "k")
                except Exception:
                    pass
                try:
                    if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                        self.info_2d_dock.refresh()
                except Exception:
                    pass
        except Exception as e:
            try:
                self.update_status(f"HKL q-grids load failed: {e}")
            except Exception:
                pass

    def clear_1d_plot(self):
        """Clear the 1D plot."""
        try:
            if hasattr(self, 'plot_item_1d'):
                self.plot_item_1d.clear()
        except Exception as e:
            self.update_status(f"Error clearing 1D plot: {e}")

    def display_1d_data(self, data):
        """Display 1D numeric data in the 1D View."""
        try:
            if not hasattr(self, 'plot_item_1d'):
                print("Warning: 1D plot not initialized")
                return
            y = np.asarray(data, dtype=np.float32).ravel()
            x = np.arange(len(y))
            self.plot_item_1d.clear()
            self.plot_item_1d.plot(x, y, pen='y')
            # Switch to 1D view tab
            if hasattr(self, 'tabWidget_analysis'):
                for i in range(self.tabWidget_analysis.count()):
                    if self.tabWidget_analysis.tabText(i) == "1D View":
                        self.tabWidget_analysis.setCurrentIndex(i)
                        break
        except Exception as e:
            self.update_status(f"Error displaying 1D data: {e}")

    def on_colormap_changed(self, colormap_name):
        """Handle colormap changes."""
        try:
            self.apply_colormap(colormap_name)
        except Exception as e:
            self.update_status(f"Error changing colormap: {e}")

    def on_auto_levels_toggled(self, enabled):
        """Handle auto levels toggle."""
        try:
            if hasattr(self, 'image_view') and hasattr(self.image_view, 'imageItem'):
                if enabled:
                    # Enable auto levels
                    self.image_view.autoLevels()
                # If disabled, keep current levels
        except Exception as e:
            self.update_status(f"Error toggling auto levels: {e}")

    def apply_colormap(self, colormap_name):
        """Apply a colormap to the image view."""
        try:
            if not hasattr(self, 'image_view'):
                return

            lut = None
            # Try pyqtgraph ColorMap first
            try:
                if hasattr(pg, "colormap") and hasattr(pg.colormap, "get"):
                    try:
                        cmap = pg.colormap.get(colormap_name)
                    except Exception:
                        cmap = None
                    if cmap is not None:
                        lut = cmap.getLookupTable(nPts=256)
            except Exception:
                lut = None

            # Fallback to matplotlib if needed
            if lut is None:
                try:
                    import matplotlib.pyplot as plt
                    mpl_cmap = plt.get_cmap(colormap_name)
                    # Build LUT as uint8 Nx3
                    xs = np.linspace(0.0, 1.0, 256, dtype=float)
                    colors = mpl_cmap(xs, bytes=True)  # returns Nx4 uint8
                    lut = colors[:, :3]
                except Exception:
                    # Last resort: grayscale
                    xs = (np.linspace(0, 255, 256)).astype(np.uint8)
                    lut = np.column_stack([xs, xs, xs])

            # Apply the lookup table
            try:
                self.image_view.imageItem.setLookupTable(lut)
            except Exception:
                pass

        except Exception as e:
            self.update_status(f"Error applying colormap: {e}")

    # === 3D Helpers ===
    def _set_3d_overlay(self, text: str):
        pass

    def _debug_3d_state(self, tag: str = ""):
        pass
    def clear_3d_plot(self):
        """Delegate 3D plot clearing to Tab3D."""
        try:
            if hasattr(self, 'tab_3d') and self.tab_3d is not None:
                self.tab_3d.clear_plot()
        except Exception as e:
            self.update_status(f"Error clearing 3D plot: {e}")

    def create_3d_from_2d(self, data):
        pass

    def create_3d_from_3d(self, data):
        pass



    def update_intensity_controls(self, data):
        """Update the intensity control ranges based on data."""
        try:
            min_val = int(np.min(data))
            max_val = int(np.max(data))

            if hasattr(self, 'sb_min_intensity_3d'):
                self.sb_min_intensity_3d.setRange(min_val - 1000, max_val + 1000)
                self.sb_min_intensity_3d.setValue(min_val)

            if hasattr(self, 'sb_max_intensity_3d'):
                self.sb_max_intensity_3d.setRange(min_val - 1000, max_val + 1000)
                self.sb_max_intensity_3d.setValue(max_val)

        except Exception as e:
            self.update_status(f"Error updating intensity controls: {e}")

    def apply_3d_visibility_settings(self):
        pass

    def on_3d_colormap_changed(self, colormap_name):
        """Delegate 3D colormap change to Tab3D."""
        try:
            if hasattr(self, 'tab_3d') and self.tab_3d is not None:
                self.tab_3d.on_colormap_changed(colormap_name)
        except Exception as e:
            self.update_status(f"Error changing 3D colormap: {e}")

    def toggle_3d_volume(self, checked):
        """Delegate 3D volume visibility toggle to Tab3D."""
        try:
            if hasattr(self, 'tab_3d') and self.tab_3d is not None:
                self.tab_3d.toggle_volume(bool(checked))
        except Exception as e:
            self.update_status(f"Error toggling 3D volume: {e}")

    def toggle_3d_slice(self, checked):
        """Delegate 3D slice visibility toggle to Tab3D."""
        try:
            if hasattr(self, 'tab_3d') and self.tab_3d is not None:
                self.tab_3d.toggle_slice(bool(checked))
        except Exception as e:
            self.update_status(f"Error toggling 3D slice: {e}")

    def toggle_3d_pointer(self, checked):
        """Delegate 3D pointer visibility toggle to Tab3D."""
        try:
            if hasattr(self, 'tab_3d') and self.tab_3d is not None:
                self.tab_3d.toggle_pointer(bool(checked))
        except Exception as e:
            self.update_status(f"Error toggling 3D pointer: {e}")

    def update_3d_intensity(self):
        """Delegate 3D intensity update to Tab3D."""
        try:
            if hasattr(self, 'tab_3d') and self.tab_3d is not None:
                self.tab_3d.update_intensity()
        except Exception as e:
            self.update_status(f"Error updating 3D intensity: {e}")

    def change_slice_orientation(self, orientation):
        """Delegate slice orientation change to Tab3D."""
        try:
            if hasattr(self, 'tab_3d') and self.tab_3d is not None:
                self.tab_3d.change_slice_orientation(orientation)
        except Exception as e:
            self.update_status(f"Error changing slice orientation: {e}")

    def reset_3d_slice(self):
        """Delegate resetting 3D slice to Tab3D."""
        try:
            if hasattr(self, 'tab_3d') and self.tab_3d is not None:
                self.tab_3d.reset_slice()
        except Exception as e:
            self.update_status(f"Error resetting 3D slice: {e}")

    # === Tree & Context Menu Events ===
    def _ensure_current_file_from_item(self, item):
        """Ensure current_file_path is set by walking up the tree to the file root."""
        try:
            cur = item
            while cur is not None:
                item_type = cur.data(0, Qt.UserRole + 2)
                if item_type == "file_root":
                    file_path = cur.data(0, Qt.UserRole + 1)
                    if file_path:
                        self.current_file_path = file_path
                    break
                cur = cur.parent()
        except Exception as e:
            self.update_status(f"Error resolving current file: {e}")

    def on_tree_item_clicked(self, item, column):
        """
        Handle tree item single-click events - only show selection info.

        Args:
            item: QTreeWidgetItem that was clicked
            column: Column index (not used)
        """
        # Ensure we know which file this item belongs to
        self._ensure_current_file_from_item(item)
        # Get the full path stored in the item
        full_path = item.data(0, 32)  # Qt.UserRole = 32

        if full_path:
            # Update status to show selected item
            self.update_status(f"Selected: {full_path} (Double-click to load)")

            # Store selected dataset path
            self.selected_dataset_path = full_path

            # Update file info display with dataset details
            self.update_file_info_with_dataset(full_path)

            # Show dataset info in the workspace without loading
            self.show_dataset_info(item)
        else:
            # Root item or group without data
            item_text = item.text(0)
            self.update_status(f"Selected: {item_text}")

            # Show selection info for non-dataset items
            if hasattr(self, 'file_status_label'):
                self.file_status_label.setText(f"Selected: {item_text}")
            if hasattr(self, 'dataset_info_text'):
                self.dataset_info_text.setPlainText("Double-click on a dataset to load it into the workspace.")

    def on_tree_item_double_clicked(self, item, column):
        """
        Handle tree item double-click events - load dataset into workspace.

        Args:
            item: QTreeWidgetItem that was double-clicked
            column: Column index (not used)
        """
        # Get the full path stored in the item
        full_path = item.data(0, 32)  # Qt.UserRole = 32
        print(f"[DEBUG] Double-clicked item text: {item.text(0)}")
        print(f"[DEBUG] Double-clicked item full_path (Qt.UserRole=32): {full_path}")
        # Detect file root items to auto-open default dataset
        try:
            item_type = item.data(0, Qt.UserRole + 2)
        except Exception:
            item_type = None

        # If a dataset/group path exists, load it as before
        if full_path:
            # Update status to show loading
            self.update_status(f"Loading dataset: {full_path}")

            # Ensure current_file_path points to the owning file
            self._ensure_current_file_from_item(item)
            # Store selected dataset path
            self.selected_dataset_path = full_path
            print(f"[DEBUG] selected_dataset_path set: {self.selected_dataset_path}")

            # Load and visualize the dataset
            try:
                self.start_dataset_load()
                print("[DEBUG] visualize_selected_dataset call completed")
            except Exception as e:
                self.update_status(f"Error in double-click load: {e}")
                print(f"[DEBUG] Exception in double-click load: {e}")
        else:
            # If this is a file root, branch by file type
            if item_type == "file_root":
                # Ensure current_file_path points to this file
                self._ensure_current_file_from_item(item)
                fp = getattr(self, 'current_file_path', None)
                if fp and self._is_hdf5_path(fp):
                    print("[DEBUG] Double-clicked HDF5 file root; attempting to load default dataset '/entry/data/data'")
                    # Set the default dataset path
                    self.selected_dataset_path = '/entry/data/data'
                    # Verify the dataset exists; if not, show message
                    try:
                        with h5py.File(fp, 'r') as h5f:
                            exists = self.selected_dataset_path in h5f
                        print(f"[DEBUG] Default dataset exists? {exists}")
                        if not exists:
                            self.update_status("Default dataset '/entry/data/data' not found in file")
                            return
                    except Exception as e:
                        self.update_status(f"Error verifying default dataset: {e}")
                        return
                    # Visualize using existing logic (will use HDF5Loader for image data)
                    try:
                        self.visualize_selected_dataset()
                        print("[DEBUG] Default dataset visualization completed")
                    except Exception as e:
                        self.update_status(f"Error loading default dataset: {e}")
                elif fp and self._is_image_path(fp):
                    print("[DEBUG] Double-clicked image file root; loading image")
                    data = self._load_image_file(fp)
                    if data is None or int(getattr(data, 'size', 0)) == 0:
                        self.update_status("Failed to load image", level='error')
                        return
                    self.display_2d_data(data)
                    if hasattr(self, 'tabWidget_analysis'):
                        try:
                            self.tabWidget_analysis.setCurrentIndex(0)
                        except Exception:
                            pass
                    # Update info text
                    try:
                        if hasattr(self, 'dataset_info_text'):
                            shp = tuple(map(int, getattr(data, 'shape', ())))
                            info = [f"Image file: {os.path.basename(fp)}",
                                    f"Shape: {shp}",
                                    f"Data Type: {getattr(data, 'dtype', '')}"]
                            self.dataset_info_text.setPlainText("\n".join(info))
                        if hasattr(self, 'file_status_label'):
                            self.file_status_label.setText(f"Image loaded: {os.path.basename(fp)}")
                    except Exception:
                        pass
                else:
                    # Unknown type: toggle expand/collapse
                    print("[DEBUG] Double-clicked unknown file root. Toggling expand/collapse.")
                    if item.isExpanded():
                        item.setExpanded(False)
                    else:
                        item.setExpanded(True)
            else:
                # Non-file root (group or child): toggle expand/collapse
                print("[DEBUG] Double-clicked a non-dataset (root/group/child). Toggling expand/collapse.")
                if item.isExpanded():
                    item.setExpanded(False)
                else:
                    item.setExpanded(True)

    def show_context_menu(self, position):
        """
        Show context menu for tree items.

        Args:
            position: Position where the context menu was requested
        """
        item = self.tree_data.itemAt(position)
        if not item:
            return

        # Check the item type
        item_type = item.data(0, Qt.UserRole + 2)

        if item_type == "file_root":
            # Create context menu for file root
            menu = QMenu(self)

            # Add collapse/expand options
            if item.isExpanded():
                collapse_action = QAction("Collapse", self)
                collapse_action.triggered.connect(lambda: self.collapse_item(item))
                menu.addAction(collapse_action)
            else:
                expand_action = QAction("Expand", self)
                expand_action.triggered.connect(lambda: self.expand_item(item))
                menu.addAction(expand_action)

            menu.addSeparator()

            remove_action = QAction("Remove File", self)
            remove_action.triggered.connect(lambda: self.remove_file(item))
            menu.addAction(remove_action)

            # Show menu at the requested position
            menu.exec_(self.tree_data.mapToGlobal(position))

        elif item_type == "folder_section":
            # Create context menu for folder section
            menu = QMenu(self)

            # Add collapse/expand options
            if item.isExpanded():
                collapse_action = QAction("Collapse Folder", self)
                collapse_action.triggered.connect(lambda: self.collapse_item(item))
                menu.addAction(collapse_action)
            else:
                expand_action = QAction("Expand Folder", self)
                expand_action.triggered.connect(lambda: self.expand_item(item))
                menu.addAction(expand_action)

            menu.addSeparator()

            # New: play images in this folder as a stacked sequence
            play_stack_action = QAction("Play images as stack", self)
            play_stack_action.setToolTip("Load all images in this folder as a frame stack and enable playback")
            play_stack_action.triggered.connect(lambda: self._play_folder_section_stack(item))
            menu.addAction(play_stack_action)

            menu.addSeparator()

            # Add option to collapse/expand all files in folder
            collapse_all_files_action = QAction("Collapse All Files", self)
            collapse_all_files_action.triggered.connect(lambda: self.collapse_all_files_in_folder(item))
            menu.addAction(collapse_all_files_action)

            expand_all_files_action = QAction("Expand All Files", self)
            expand_all_files_action.triggered.connect(lambda: self.expand_all_files_in_folder(item))
            menu.addAction(expand_all_files_action)

            menu.addSeparator()

            remove_folder_action = QAction("Remove Folder", self)
            remove_folder_action.triggered.connect(lambda: self.remove_folder_section(item))
            menu.addAction(remove_folder_action)

            # Show menu at the requested position
            menu.exec_(self.tree_data.mapToGlobal(position))

    def _play_folder_section_stack(self, item):
        """Stack all supported images within the folder section and play them in the 2D viewer.

        Supported formats: .tif, .tiff, .png, .jpg, .jpeg, .bmp
        Search is non-recursive (immediate files only) for the first iteration.
        """
        try:
            # Resolve folder path and name
            folder_path = item.data(0, Qt.UserRole + 1)
            if not folder_path or not os.path.isdir(folder_path):
                QMessageBox.information(self, "Play Images", "Selected folder path is invalid.")
                return
            folder_name = os.path.basename(folder_path)

            # Import on demand to avoid unnecessary startup overhead
            try:
                from viewer.tools.file_convert import list_images, stack_images
            except Exception:
                QMessageBox.critical(self, "Play Images", "Required image utilities are unavailable.")
                return

            # Collect image files (non-recursive, lowercase patterns)
            patterns = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
            files = list_images(Path(folder_path), patterns, recursive=False)
            if not files:
                QMessageBox.information(self, "No Images", "No supported images found in the selected folder.")
                try:
                    self.update_status("No images found in folder", level='warning')
                except Exception:
                    pass
                return

            # Stack images; skip size mismatches internally
            vol, shape_hw, used = stack_images(files, log=lambda msg: self.update_status(str(msg)))
            if vol.size == 0:
                QMessageBox.information(self, "No Usable Images", "No images with a consistent shape could be loaded.")
                try:
                    self.update_status("No usable images (size mismatches)", level='warning')
                except Exception:
                    pass
                return

            # Safety: warn if very large memory footprint (>1 GB); proceed (no cancel) for initial iteration
            try:
                if getattr(vol, 'nbytes', 0) > 1_000_000_000:
                    sz_gb = vol.nbytes / (1024 * 1024 * 1024)
                    QMessageBox.warning(self, "Large Stack", f"Stack size is large: {sz_gb:.2f} GB. Playback may impact performance.")
                    self.update_status("Warning: Large stack loaded into memory")
            except Exception:
                pass

            # Display data in 2D viewer; enables frame controls and shows "Frame 0 of N"
            self.display_2d_data(vol)
            try:
                if hasattr(self, 'tabWidget_analysis') and self.tabWidget_analysis is not None:
                    self.tabWidget_analysis.setCurrentIndex(0)
            except Exception:
                pass

            # Update UI labels
            try:
                if hasattr(self, 'file_status_label') and self.file_status_label is not None:
                    self.file_status_label.setText(f"Playing folder: {folder_name} ({int(vol.shape[0])} frames)")
            except Exception:
                pass
            try:
                if hasattr(self, 'dataset_info_text') and self.dataset_info_text is not None:
                    h, w = (int(shape_hw[0]), int(shape_hw[1])) if isinstance(shape_hw, tuple) and len(shape_hw) == 2 else (int(vol.shape[1]), int(vol.shape[2]))
                    info_lines = [
                        f"Folder path: {folder_path}",
                        f"Frame count: {int(vol.shape[0])}",
                        f"Frame shape (H,W): ({h}, {w})",
                        f"Dtype: {vol.dtype}",
                        f"Total elements: {int(vol.size):,}",
                    ]
                    self.dataset_info_text.setPlainText("\n".join(info_lines))
            except Exception:
                pass

            # Auto-start playback when multiple frames are available (default behavior)
            try:
                if int(vol.shape[0]) > 1:
                    self.start_playback()
            except Exception:
                pass

            # Status update
            try:
                self.update_status(f"Playing stack from folder: {folder_name} ({int(vol.shape[0])} frames)")
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Play Images", f"Failed to stack images from folder: {e}")
            try:
                self.update_status(f"Error stacking images: {e}", level='error')
            except Exception:
                pass

    def remove_file(self, item):
        """
        Remove a file from the tree.

        Args:
            item: QTreeWidgetItem representing the file root to remove
        """
        file_path = item.data(0, Qt.UserRole + 1)
        file_name = item.text(0)

        # Confirm removal
        reply = QMessageBox.question(
            self,
            "Remove File",
            f"Are you sure you want to remove '{file_name}' from the tree?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Remove the item from the tree
            root = self.tree_data.invisibleRootItem()
            root.removeChild(item)

            self.update_status(f"Removed file: {file_name}")

            # Update analysis placeholder if no files remain
            if self.tree_data.topLevelItemCount() == 0:
                if hasattr(self, 'label_analysis_placeholder'):
                    self.label_analysis_placeholder.setText("Load HDF5 data to begin analysis")

    def collapse_item(self, item):
        """
        Collapse a specific tree item.

        Args:
            item: QTreeWidgetItem to collapse
        """
        item.setExpanded(False)
        self.update_status(f"Collapsed: {item.text(0)}")

    def expand_item(self, item):
        """
        Expand a specific tree item.

        Args:
            item: QTreeWidgetItem to expand
        """
        item.setExpanded(True)
        self.update_status(f"Expanded: {item.text(0)}")

    def collapse_all(self):
        """Collapse all items in the tree."""
        if hasattr(self, 'tree_data'):
            self.tree_data.collapseAll()
            self.update_status("Collapsed all tree items")

    def expand_all(self):
        """Expand all items in the tree."""
        if hasattr(self, 'tree_data'):
            self.tree_data.expandAll()
            self.update_status("Expanded all tree items")

    def collapse_all_files_in_folder(self, folder_item):
        """
        Collapse all files within a folder section.

        Args:
            folder_item: QTreeWidgetItem representing the folder section
        """
        for i in range(folder_item.childCount()):
            child_item = folder_item.child(i)
            child_item.setExpanded(False)

        folder_name = folder_item.text(0)
        self.update_status(f"Collapsed all files in folder: {folder_name}")

    def expand_all_files_in_folder(self, folder_item):
        """
        Expand all files within a folder section.

        Args:
            folder_item: QTreeWidgetItem representing the folder section
        """
        for i in range(folder_item.childCount()):
            child_item = folder_item.child(i)
            child_item.setExpanded(True)

        folder_name = folder_item.text(0)
        self.update_status(f"Expanded all files in folder: {folder_name}")

    def remove_folder_section(self, folder_item):
        """
        Remove an entire folder section from the tree.

        Args:
            folder_item: QTreeWidgetItem representing the folder section to remove
        """
        folder_path = folder_item.data(0, Qt.UserRole + 1)
        folder_name = folder_item.text(0)

        # Confirm removal
        reply = QMessageBox.question(
            self,
            "Remove Folder",
            f"Are you sure you want to remove the entire folder section '{folder_name}' from the tree?\n\nThis will remove all files in this folder from the tree.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Remove the folder section from the tree
            root = self.tree_data.invisibleRootItem()
            root.removeChild(folder_item)

            self.update_status(f"Removed folder section: {folder_name}")

            # Update analysis placeholder if no files remain
            if self.tree_data.topLevelItemCount() == 0:
                if hasattr(self, 'file_status_label'):
                    self.file_status_label.setText("No HDF5 file loaded")
                if hasattr(self, 'dataset_info_text'):
                    self.dataset_info_text.setPlainText("Load HDF5 data to begin analysis")

    def show_dataset_info(self, item):
        """
        Show information about the selected dataset without loading it.

        Args:
            item: QTreeWidgetItem representing the dataset
        """
        try:
            full_path = item.data(0, 32)
            if not full_path or not self.current_file_path:
                return

            # Get dataset information
            with h5py.File(self.current_file_path, 'r') as h5file:
                if full_path in h5file:
                    dataset = h5file[full_path]

                    if isinstance(dataset, h5py.Dataset):
                        # Show dataset information
                        shape_str = f"{dataset.shape}" if dataset.shape else "scalar"
                        dtype_str = str(dataset.dtype)
                        size_str = f"{dataset.size:,}" if dataset.size > 0 else "0"

                        info_text = (f"Dataset: {full_path}\n"
                                   f"Shape: {shape_str}\n"
                                   f"Data Type: {dtype_str}\n"
                                   f"Size: {size_str} elements\n\n"
                                   f"Double-click to load into workspace")

                        # Update 2D display
                        if hasattr(self, 'dataset_info_text'):
                            self.dataset_info_text.setPlainText(info_text)
                    else:
                        # It's a group
                        group_info = f"Group: {full_path}\n\nContains {len(dataset)} items\n\nDouble-click to expand/collapse"
                        if hasattr(self, 'dataset_info_text'):
                            self.dataset_info_text.setPlainText(group_info)

        except Exception as e:
            error_msg = f"Error reading dataset info: {str(e)}"
            if hasattr(self, 'dataset_info_text'):
                self.dataset_info_text.setPlainText(error_msg)

    # === Frame Navigation ===
    @pyqtSlot()
    def previous_frame(self):
        """Navigate to the previous frame."""
        try:
            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                current_frame = self.frame_spinbox.value()
                if current_frame > 0:
                    self.frame_spinbox.setValue(current_frame - 1)
        except Exception as e:
            self.update_status(f"Error navigating to previous frame: {e}")

    @pyqtSlot()
    def next_frame(self):
        """Navigate to the next frame."""
        try:
            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                current_frame = self.frame_spinbox.value()
                max_frame = self.frame_spinbox.maximum()
                if current_frame < max_frame:
                    self.frame_spinbox.setValue(current_frame + 1)
        except Exception as e:
            self.update_status(f"Error navigating to next frame: {e}")

    @pyqtSlot(int)
    def on_frame_spinbox_changed(self, frame_index):
        """Handle frame spinbox changes for 3D data navigation."""
        try:
            if not hasattr(self, 'current_2d_data') or self.current_2d_data is None:
                return

            if self.current_2d_data.ndim != 3:
                return

            # Get the selected frame
            if frame_index < 0 or frame_index >= self.current_2d_data.shape[0]:
                frame_index = 0

            frame_data = np.asarray(self.current_2d_data[frame_index], dtype=np.float32)

            # Update the image view with the new frame
            auto_levels = hasattr(self, 'cbAutoLevels') and self.cbAutoLevels.isChecked()
            self.image_view.setImage(
                frame_data,
                autoLevels=auto_levels,
                autoRange=False,  # Don't auto-range when changing frames
                autoHistogramRange=auto_levels
            )

            # Update frame info label and button states
            num_frames = self.current_2d_data.shape[0]
            print(f"[FRAME] on_frame_spinbox_changed: frame_index={frame_index}, num_frames={num_frames}")
            height, width = frame_data.shape
            if hasattr(self, 'frame_info_label'):
                self.frame_info_label.setText(f"Image Dimensions: {width}x{height} pixels (frame {frame_index} of {num_frames})")
            # Update overlay text
            self.update_overlay_text(width, height, f"Frame {frame_index} of {num_frames}")

            # Update hover tooltip/crosshair at last position during playback
            try:
                xy = getattr(self, '_last_hover_xy', None)
                if xy and bool(getattr(self, '_hover_enabled', True)):
                    self._update_hover_text_at(int(xy[0]), int(xy[1]))
            except Exception:
                pass

            # Update HKL 3D plot if open
            try:
                self._update_hkl3d_plot_for_current_frame()
            except Exception:
                pass

            # Update button states
            if hasattr(self, 'btn_prev_frame'):
                self.btn_prev_frame.setEnabled(frame_index > 0)
            if hasattr(self, 'btn_next_frame'):
                self.btn_next_frame.setEnabled(frame_index < num_frames - 1)

            # Refresh ROI stats when frame changes
            try:
                self.roi_manager.update_all_roi_stats()
            except Exception:
                pass
            try:
                if hasattr(self, 'info_2d_dock') and self.info_2d_dock is not None:
                    self.info_2d_dock.refresh()
            except Exception:
                pass

        except Exception as e:
            self.update_status(f"Error changing frame: {e}")

    def start_playback(self):
        """Start frame playback if a 3D stack is loaded and controls are enabled."""
        try:
            if not hasattr(self, 'current_2d_data') or self.current_2d_data is None:
                return
            if self.current_2d_data.ndim != 3:
                return
            # Only play if more than 1 frame
            num_frames = self.current_2d_data.shape[0]
            if num_frames <= 1:
                print(f"[PLAYBACK] start_playback: num_frames={num_frames} -> not enough frames to play")
                return
            # Set timer interval from FPS
            fps = 2
            try:
                if hasattr(self, 'sb_fps'):
                    fps = max(1, int(self.sb_fps.value()))
            except Exception:
                fps = 2
            interval_ms = int(1000 / max(1, fps))
            print(f"[PLAYBACK] start_playback: num_frames={num_frames}, fps={fps}, interval_ms={interval_ms}")
            # Reset frame index to 0 at playback start to avoid stale index from previous data
            try:
                if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                    self.frame_spinbox.setValue(0)
            except Exception:
                pass
            if hasattr(self, 'play_timer') and self.play_timer is not None:
                try:
                    self.play_timer.setInterval(interval_ms)
                    self.play_timer.start()
                    try:
                        print(f"[PLAYBACK] timer state: {'active' if self.play_timer.isActive() else 'inactive'}")
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[PLAYBACK] ERROR starting timer: {e}")
            # Update control states
            try:
                self.btn_play.setEnabled(False)
                self.btn_pause.setEnabled(True)
            except Exception:
                pass
            self.update_status("Playback started")
        except Exception as e:
            self.update_status(f"Error starting playback: {e}")

    def pause_playback(self):
        """Pause frame playback."""
        try:
            if hasattr(self, 'play_timer') and self.play_timer is not None:
                try:
                    self.play_timer.stop()
                except Exception:
                    pass
            try:
                self.btn_play.setEnabled(True)
                self.btn_pause.setEnabled(False)
            except Exception:
                pass
            self.update_status("Playback paused")
        except Exception as e:
            self.update_status(f"Error pausing playback: {e}")

    def on_fps_changed(self, value):
        """Update timer interval when FPS changes."""
        try:
            fps = max(1, int(value))
            interval_ms = int(1000 / fps)
            if hasattr(self, 'play_timer') and self.play_timer is not None:
                try:
                    self.play_timer.setInterval(interval_ms)
                except Exception:
                    pass
        except Exception as e:
            self.update_status(f"Error updating FPS: {e}")

    def _advance_frame_playback(self):
        """Advance one frame; handle auto replay at end."""
        try:
            if not hasattr(self, 'current_2d_data') or self.current_2d_data is None:
                return
            if self.current_2d_data.ndim != 3:
                return
            num_frames = self.current_2d_data.shape[0]
            if num_frames <= 1:
                print("[PLAYBACK] tick: num_frames<=1 -> pausing")
                self.pause_playback()
                return
            idx = 0
            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                try:
                    idx = int(self.frame_spinbox.value())
                except Exception:
                    idx = 0
            # Clamp idx to valid range for current data
            if idx < 0 or idx >= num_frames:
                idx = 0
            next_idx = idx + 1
            if next_idx >= num_frames:
                # Auto replay from beginning if checked
                auto = False
                try:
                    auto = bool(self.cb_auto_replay.isChecked()) if hasattr(self, 'cb_auto_replay') else False
                except Exception:
                    auto = False
                print(f"[PLAYBACK] tick: idx={idx}, next_idx={next_idx} reached end, auto_replay={auto}")
                if auto:
                    next_idx = 0
                else:
                    self.pause_playback()
                    return
            print(f"[PLAYBACK] tick: advancing to next_idx={next_idx} of num_frames={num_frames}")
            # Set via spinbox to reuse existing update logic
            if hasattr(self, 'frame_spinbox'):
                try:
                    self.frame_spinbox.setValue(next_idx)
                except Exception as e:
                    print(f"[PLAYBACK] ERROR setting frame_spinbox: {e}")
        except Exception:
            pass

    # === File Info Display ===
    def update_file_info_display(self, file_path, additional_info=None):
        """
        Update the file information display with details about the current file.

        Args:
            file_path (str): Path to the current file or status message
            additional_info (dict): Additional information to display
        """
        if not hasattr(self, 'file_info_text'):
            return

        try:
            if file_path == "No file loaded" or not os.path.exists(file_path):
                # Show default message
                self.file_info_text.setPlainText("Load an HDF5 file to view file information.")
                return

            # Get file information
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            file_modified = os.path.getmtime(file_path)

            # Format file size
            if file_size < 1024:
                size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            elif file_size < 1024 * 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{file_size / (1024 * 1024 * 1024):.1f} GB"

            # Format modification time
            import datetime
            mod_time = datetime.datetime.fromtimestamp(file_modified).strftime("%Y-%m-%d %H:%M:%S")

            # Get HDF5 file information
            info_lines = []
            info_lines.append(f"File: {os.path.basename(file_path)}")
            info_lines.append(f"Path: {os.path.dirname(file_path)}")
            info_lines.append(f"Size: {size_str}")
            info_lines.append(f"Modified: {mod_time}")

            # Show type-specific info
            try:
                if self._is_hdf5_path(file_path):
                    with h5py.File(file_path, 'r') as h5file:
                        # Count groups and datasets
                        def count_items(group, counts):
                            for key in group.keys():
                                item = group[key]
                                if isinstance(item, h5py.Group):
                                    counts['groups'] += 1
                                    count_items(item, counts)
                                elif isinstance(item, h5py.Dataset):
                                    counts['datasets'] += 1

                        counts = {'groups': 0, 'datasets': 0}
                        count_items(h5file, counts)

                        info_lines.append("")
                        info_lines.append("HDF5 Structure:")
                        info_lines.append(f"Groups: {counts['groups']}")
                        info_lines.append(f"Datasets: {counts['datasets']}")

                        # Get HDF5 attributes if any
                        if h5file.attrs:
                            info_lines.append(f"File Attributes: {len(h5file.attrs)}")
                elif self._is_image_path(file_path):
                    # Try to read image dimensions without forcing display
                    try:
                        import tifffile as tiff
                        data = tiff.imread(file_path)
                    except Exception:
                        data = None
                    if data is None:
                        try:
                            from PIL import Image
                            img = Image.open(file_path)
                            data = np.asarray(img)
                        except Exception:
                            data = None
                    if data is None:
                        try:
                            import imageio.v2 as iio
                            data = iio.imread(file_path)
                        except Exception:
                            data = None
                    if data is not None:
                        arr = np.asarray(data)
                        shp = tuple(map(int, getattr(arr, 'shape', ())))
                        info_lines.append("")
                        info_lines.append("Image Info:")
                        info_lines.append(f"Shape: {shp}")
                        info_lines.append(f"Data Type: {getattr(arr, 'dtype', '')}")
            except Exception:
                pass

            # Add additional info if provided
            if additional_info:
                info_lines.append("")
                info_lines.append("Selection Details:")
                for key, value in additional_info.items():
                    info_lines.append(f"{key}: {value}")

            # Update the file info tab
            self.file_info_text.setPlainText("\n".join(info_lines))

        except Exception as e:
            error_text = f"Error reading file information: {str(e)}"
            self.file_info_text.setPlainText(error_text)

    def update_file_info_with_dataset(self, dataset_path):
        """
        Update the file information display with details about the selected dataset.

        Args:
            dataset_path (str): Path to the selected dataset within the HDF5 file
        """
        if not self.current_file_path or not dataset_path:
            return

        try:
            with h5py.File(self.current_file_path, 'r') as h5file:
                if dataset_path in h5file:
                    item = h5file[dataset_path]

                    additional_info = {}

                    if isinstance(item, h5py.Dataset):
                        # Dataset information
                        additional_info['Selected Dataset'] = dataset_path
                        additional_info['Shape'] = str(item.shape) if item.shape else "scalar"
                        additional_info['Data Type'] = str(item.dtype)
                        additional_info['Size'] = f"{item.size:,} elements" if item.size > 0 else "0 elements"

                        # Memory size estimation
                        if item.size > 0:
                            mem_size = item.size * item.dtype.itemsize
                            if mem_size < 1024:
                                mem_str = f"{mem_size} bytes"
                            elif mem_size < 1024 * 1024:
                                mem_str = f"{mem_size / 1024:.1f} KB"
                            elif mem_size < 1024 * 1024 * 1024:
                                mem_str = f"{mem_size / (1024 * 1024):.1f} MB"
                            else:
                                mem_str = f"{mem_size / (1024 * 1024 * 1024):.1f} GB"
                            additional_info['Memory Size'] = mem_str

                        # Dataset attributes
                        if item.attrs:
                            additional_info['Dataset Attributes'] = f"{len(item.attrs)} attributes"

                        # Compression info
                        if item.compression:
                            additional_info['Compression'] = item.compression
                            if item.compression_opts:
                                additional_info['Compression Level'] = str(item.compression_opts)

                    elif isinstance(item, h5py.Group):
                        # Group information
                        additional_info['Selected Group'] = dataset_path
                        additional_info['Contains'] = f"{len(item)} items"

                        # Count subgroups and datasets
                        subgroups = sum(1 for key in item.keys() if isinstance(item[key], h5py.Group))
                        subdatasets = sum(1 for key in item.keys() if isinstance(item[key], h5py.Dataset))

                        if subgroups > 0:
                            additional_info['Subgroups'] = str(subgroups)
                        if subdatasets > 0:
                            additional_info['Subdatasets'] = str(subdatasets)

                        # Group attributes
                        if item.attrs:
                            additional_info['Group Attributes'] = f"{len(item.attrs)} attributes"

                    # Update the display with additional dataset/group info
                    self.update_file_info_display(self.current_file_path, additional_info)

        except Exception as e:
            # Fall back to basic file info if dataset reading fails
            self.update_file_info_display(self.current_file_path,
                                        {'Error': f"Could not read dataset info: {str(e)}"})

    # === Speckle Analysis & ROI ===
    def on_log_scale_toggled(self, checked):
        """Handle log scale checkbox toggle."""
        try:
            if hasattr(self, 'image_view') and hasattr(self, 'current_2d_data') and self.current_2d_data is not None:
                # Get current frame index
                current_frame = 0
                if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                    current_frame = self.frame_spinbox.value()

                # Get the current frame data
                if self.current_2d_data.ndim == 3:
                    frame_data = self.current_2d_data[current_frame]
                else:
                    frame_data = self.current_2d_data

                # Apply or remove log scale
                if checked:
                    # Apply log scale (log1p to handle zeros)
                    display_data = np.log1p(np.maximum(frame_data, 0))
                else:
                    # Use original data
                    display_data = frame_data

                # Update the image view
                auto_levels = hasattr(self, 'cbAutoLevels') and self.cbAutoLevels.isChecked()
                self.image_view.setImage(
                    display_data,
                    autoLevels=auto_levels,
                    autoRange=False,
                    autoHistogramRange=auto_levels
                )

                # Apply current colormap
                if hasattr(self, 'cbColorMapSelect_2d'):
                    current_colormap = self.cbColorMapSelect_2d.currentText()
                    self.apply_colormap(current_colormap)

                # Update vmin/vmax controls for log scale
                self.update_vmin_vmax_for_log_scale(frame_data, checked)

                # Refresh ROI stats to reflect displayed image
                try:
                    self.roi_manager.update_all_roi_stats()
                except Exception:
                    pass

                self.update_status(f"Log scale {'enabled' if checked else 'disabled'}")
            else:
                print("No image data available for log scale")
        except Exception as e:
            self.update_status(f"Error toggling log scale: {e}")

    def update_vmin_vmax_for_log_scale(self, data, log_scale_enabled):
        """Update vmin/vmax controls based on log scale state."""
        try:
            if log_scale_enabled:
                # For log scale, set reasonable ranges
                min_val = max(1, int(np.min(data[data > 0]))) if np.any(data > 0) else 1
                max_val = int(np.max(data))

                if hasattr(self, 'sbVmin'):
                    self.sbVmin.setRange(1, max_val)
                    self.sbVmin.setValue(min_val)

                if hasattr(self, 'sbVmax'):
                    self.sbVmax.setRange(min_val + 1, max_val * 2)
                    self.sbVmax.setValue(max_val)
            else:
                # For linear scale, use full data range
                min_val = int(np.min(data))
                max_val = int(np.max(data))

                if hasattr(self, 'sbVmin'):
                    self.sbVmin.setRange(min_val, max_val)
                    self.sbVmin.setValue(min_val)

                if hasattr(self, 'sbVmax'):
                    self.sbVmax.setRange(min_val + 1, max_val * 2)
                    self.sbVmax.setValue(max_val)
        except Exception as e:
            self.update_status(f"Error updating vmin/vmax controls: {e}")

    def on_vmin_changed(self, value):
        """Handle vmin spinbox value change."""
        try:
            if hasattr(self, 'image_view') and hasattr(self, 'current_2d_data') and self.current_2d_data is not None:
                # Get current vmax
                vmax = self.sbVmax.value() if hasattr(self, 'sbVmax') else 100

                # Ensure vmin < vmax
                if value >= vmax:
                    return

                # Apply log scale if enabled
                if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                    vmin_display = np.log1p(value)
                    vmax_display = np.log1p(vmax)
                else:
                    vmin_display = value
                    vmax_display = vmax

                # Update image levels
                self.image_view.setLevels(min=vmin_display, max=vmax_display)
                # Refresh ROI stats (based on displayed image)
                try:
                    self.roi_manager.update_all_roi_stats()
                except Exception:
                    pass
                self.update_status(f"Vmin set to: {value}")
        except Exception as e:
            self.update_status(f"Error changing vmin: {e}")

    def on_vmax_changed(self, value):
        """Handle vmax spinbox value change."""
        try:
            if hasattr(self, 'image_view') and hasattr(self, 'current_2d_data') and self.current_2d_data is not None:
                # Get current vmin
                vmin = self.sbVmin.value() if hasattr(self, 'sbVmin') else 0

                # Ensure vmax > vmin
                if value <= vmin:
                    return

                # Apply log scale if enabled
                if hasattr(self, 'cbLogScale') and self.cbLogScale.isChecked():
                    vmin_display = np.log1p(vmin)
                    vmax_display = np.log1p(value)
                else:
                    vmin_display = vmin
                    vmax_display = value

                # Update image levels
                self.image_view.setLevels(min=vmin_display, max=vmax_display)
                # Refresh ROI stats (based on displayed image)
                try:
                    self.roi_manager.update_all_roi_stats()
                except Exception:
                    pass
                self.update_status(f"Vmax set to: {value}")
        except Exception as e:
            self.update_status(f"Error changing vmax: {e}")

    def on_draw_roi_clicked(self):
        """Handle Draw ROI button click (delegated to ROIManager)."""
        try:
            self.roi_manager.create_and_add_roi()
        except Exception as e:
            self.update_status(f"Error drawing ROI: {e}")

    def on_ref_frame_changed(self, value):
        """Handle reference frame spinbox value change."""
        try:
            # Update the current frame to match reference frame
            if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                self.frame_spinbox.setValue(value)
            self.update_status(f"Reference frame set to: {value}")
        except Exception as e:
            self.update_status(f"Error changing reference frame: {e}")

    def on_other_frame_changed(self, value):
        """Handle other frame spinbox value change."""
        try:
            self.update_status(f"Other frame set to: {value}")
        except Exception as e:
            self.update_status(f"Error changing other frame: {e}")







    # === Control Updates ===
    def update_frame_controls_for_2d_data(self):
        """Update frame controls for 2D data (disable frame navigation)."""
        try:
            if hasattr(self, 'frame_spinbox'):
                self.frame_spinbox.setEnabled(False)
                self.frame_spinbox.setValue(0)
                self.frame_spinbox.setMaximum(0)

            if hasattr(self, 'btn_prev_frame'):
                self.btn_prev_frame.setEnabled(False)
            if hasattr(self, 'btn_next_frame'):
                self.btn_next_frame.setEnabled(False)

            # Stop playback timer and disable controls
            try:
                if hasattr(self, 'play_timer') and self.play_timer is not None:
                    self.play_timer.stop()
                if hasattr(self, 'btn_play'):
                    self.btn_play.setEnabled(False)
                if hasattr(self, 'btn_pause'):
                    self.btn_pause.setEnabled(False)
                if hasattr(self, 'sb_fps'):
                    self.sb_fps.setEnabled(False)
                if hasattr(self, 'cb_auto_replay'):
                    self.cb_auto_replay.setEnabled(False)
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error updating frame controls for 2D data: {e}")

    def update_frame_controls_for_3d_data(self, num_frames):
        """Update frame controls for 3D data (enable frame navigation)."""
        try:
            if hasattr(self, 'frame_spinbox'):
                self.frame_spinbox.setEnabled(True)
                self.frame_spinbox.setMaximum(num_frames - 1)
                self.frame_spinbox.setValue(0)

            if hasattr(self, 'btn_prev_frame'):
                self.btn_prev_frame.setEnabled(False)  # Disabled for frame 0
            if hasattr(self, 'btn_next_frame'):
                self.btn_next_frame.setEnabled(num_frames > 1)

            # Enable or disable playback controls based on frame count
            try:
                enable_playback = num_frames > 1
                if hasattr(self, 'btn_play'):
                    self.btn_play.setEnabled(enable_playback)
                if hasattr(self, 'btn_pause'):
                    self.btn_pause.setEnabled(False)  # initially paused
                if hasattr(self, 'sb_fps'):
                    self.sb_fps.setEnabled(enable_playback)
                if hasattr(self, 'cb_auto_replay'):
                    self.cb_auto_replay.setEnabled(enable_playback)
                    try:
                        # Select auto replay when playback becomes available
                        self.cb_auto_replay.setChecked(True)
                    except Exception:
                        pass
                # Stop timer on reconfigure
                if hasattr(self, 'play_timer') and self.play_timer is not None:
                    self.play_timer.stop()
            except Exception:
                pass
        except Exception as e:
            self.update_status(f"Error updating frame controls for 3D data: {e}")

    def update_speckle_controls_for_data(self, data):
        """Update speckle analysis controls based on loaded data."""
        try:
            if data.ndim == 3:
                # 3D data - enable frame selection
                max_frame = data.shape[0] - 1

                if hasattr(self, 'sbRefFrame'):
                    self.sbRefFrame.setMaximum(max_frame)
                    self.sbRefFrame.setValue(0)
                    self.sbRefFrame.setEnabled(True)

                if hasattr(self, 'sbOtherFrame'):
                    self.sbOtherFrame.setMaximum(max_frame)
                    self.sbOtherFrame.setValue(min(1, max_frame))
                    self.sbOtherFrame.setEnabled(True)
            else:
                # 2D data - disable frame selection
                if hasattr(self, 'sbRefFrame'):
                    self.sbRefFrame.setValue(0)
                    self.sbRefFrame.setMaximum(0)
                    self.sbRefFrame.setEnabled(False)

                if hasattr(self, 'sbOtherFrame'):
                    self.sbOtherFrame.setValue(0)
                    self.sbOtherFrame.setMaximum(0)
                    self.sbOtherFrame.setEnabled(False)

        except Exception as e:
            self.update_status(f"Error updating speckle controls: {e}")

    def update_vmin_vmax_controls_for_data(self, data):
        """Update vmin/vmax controls based on data range."""
        try:
            min_val = int(np.min(data))
            max_val = int(np.max(data))

            if hasattr(self, 'sbVmin'):
                self.sbVmin.setRange(min_val, max_val)
                self.sbVmin.setValue(min_val)

            if hasattr(self, 'sbVmax'):
                self.sbVmax.setRange(min_val + 1, max_val * 2)
                self.sbVmax.setValue(max_val)

        except Exception as e:
            self.update_status(f"Error updating vmin/vmax controls: {e}")

    # === ROI Stats & Context Menu Actions ===
    def get_current_frame_data(self):
        try:
            if not hasattr(self, 'current_2d_data') or self.current_2d_data is None:
                return None
            if self.current_2d_data.ndim == 3:
                frame_index = 0
                if hasattr(self, 'frame_spinbox') and self.frame_spinbox.isEnabled():
                    frame_index = self.frame_spinbox.value()
                if frame_index < 0 or frame_index >= self.current_2d_data.shape[0]:
                    frame_index = 0
                return np.asarray(self.current_2d_data[frame_index], dtype=np.float32)
            else:
                return np.asarray(self.current_2d_data, dtype=np.float32)
        except Exception:
            return None

    def compute_roi_stats(self, frame_data, roi):
        try:
            if frame_data is None or roi is None:
                return None
            height, width = frame_data.shape
            pos = roi.pos(); size = roi.size()
            x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
            w = max(1, int(size.x())); h = max(1, int(size.y()))
            x1 = min(width, x0 + w); y1 = min(height, y0 + h)
            if x0 >= x1 or y0 >= y1:
                return None
            sub = frame_data[y0:y1, x0:x1]
            stats = {
                'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0,
                'sum': float(np.sum(sub)),
                'min': float(np.min(sub)),
                'max': float(np.max(sub)),
                'mean': float(np.mean(sub)),
                'std': float(np.std(sub)),
                'count': int(sub.size),
            }
            return stats
        except Exception:
            return None

    def show_roi_stats_for_roi(self, roi):
        """Delegate to ROIManager."""
        try:
            self.roi_manager.show_roi_stats_for_roi(roi)
        except Exception as e:
            self.update_status(f"Error showing ROI stats: {e}")

    def set_active_roi(self, roi):
        """Delegate to ROIManager."""
        try:
            self.roi_manager.set_active_roi(roi)
        except Exception as e:
            self.update_status(f"Error setting active ROI: {e}")

    def delete_roi(self, roi):
        """Delegate to ROIManager."""
        try:
            self.roi_manager.delete_roi(roi)
        except Exception as e:
            self.update_status(f"Error deleting ROI: {e}")

    # === File & Dataset Info Helpers ===

# === Entrypoint ===
def main():
    """Main entry point for the Workbench application."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Workbench")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("DashPVA")

    # Global excepthook to log unhandled errors to error_output.txt
    def _log_excepthook(exctype, value, tb):
        try:
            import datetime, traceback
            error_file = project_root / "error_output.txt"
            with open(error_file, "a") as f:
                f.write(f"[{datetime.datetime.now().isoformat()}] Unhandled exception: {exctype.__name__}: {value}\n")
                traceback.print_tb(tb, file=f)
        except Exception:
            pass
    sys.excepthook = _log_excepthook

    # Create and show the main window
    window = WorkbenchWindow()
    window.show()

    # Start the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
