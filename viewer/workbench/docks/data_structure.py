from viewer.workbench.docks.base_dock import BaseDock
from viewer.base_window import BaseWindow
from viewer.workbench.workers import DatasetLoader
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QAction, QVBoxLayout, QHBoxLayout, QTreeWidget, QGroupBox, QPushButton, QMessageBox
from utils.hdf5_loader import HDF5Loader
import os

class DataStructureDock(BaseDock):
    def __init__(self, title="Data Structure", main_window:BaseWindow=None, segment_name="other", dock_area=Qt.LeftDockWidgetArea):
        super().__init__(title, main_window, segment_name=segment_name, dock_area=dock_area)
        self.title = title
        self.main_window = main_window
        # Parent BaseDock.__init__ already performs setup; no need to call again
        self.build_dock()

    def setup(self):
        try:
            # Delegate core docking and Windows menu registration to BaseDock.setup
            super().setup()
        except Exception as e:
            print(e)
            pass

    def build_dock(self):
        """Build the UI Dock"""
        # Create a group box to mirror the Workbench's "Data Structure" panel
        self.gb_data_structure = QGroupBox()
        self.gb_data_structure.setObjectName("groupBox_dataTree")

        # Create the tree widget that will display the hierarchical dataset
        self.tree_data = QTreeWidget()
        self.tree_data.setObjectName("tree_data")
        self.tree_data.setHeaderHidden(True)

        # Layout the tree inside the group box with a simple Refresh button
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        header = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        try:
            self.btn_refresh.setToolTip("Refresh Data Structure")
            self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        except Exception:
            pass
        header.addWidget(self.btn_refresh)
        header.addStretch(1)
        layout.addLayout(header)
        layout.addWidget(self.tree_data)
        self.gb_data_structure.setLayout(layout)

        # Set the group box as the dock's main widget
        self.setWidget(self.gb_data_structure)

    def _on_refresh_clicked(self):
        try:
            mw = self.main_window
            if hasattr(mw, 'refresh_data_structure'):
                mw.refresh_data_structure()
            else:
                # Fallback: reload current file if available
                fp = getattr(mw, 'current_file_path', None)
                if fp and os.path.exists(fp) and hasattr(mw, 'load_file_content'):
                    mw.load_file_content(fp)
                else:
                    QMessageBox.information(self, "Refresh", "No file loaded to refresh.")
        except Exception as e:
            QMessageBox.critical(self, "Refresh Error", f"Failed to refresh: {e}")

    def connect(self):
        pass

    def refresh_data_structure_display(self, file_path):
        pass

    def _load_data_structure(self):
        try: 
            folder_name = os.path.basename(self.current_file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load folder: {str(e)}")
            self.update_status("Failed to load folder")

    def _populate_tree_recursive(self):
        pass

    def _start_dataset_load(self):
        """Create a worker thread to load dataset without blocking the UI."""
        try:
            self.update_status(f"Loading dataset: {self.selected_dataset_path}")
            self._dataset_thread = QThread()
            self._dataset_worker = DatasetLoader(self.current_file_path, self.selected_dataset_path)
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
