from viewer.workbench.docks.base_dock import BaseDock
from viewer.base_window import BaseWindow
from viewer.workbench.workers import DatasetLoader
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QAction, QVBoxLayout, QHBoxLayout, QTreeWidget, QGroupBox, QPushButton, QMessageBox
from utils.hdf5_loader import HDF5Loader
import os

class DataStructureDock(BaseDock):
    def __init__(self, title="Data Structure", main_window:BaseWindow=None, segment_name="other", dock_area=Qt.LeftDockWidgetArea, show: bool = True):
        super().__init__(title, main_window, segment_name=segment_name, dock_area=dock_area, show=show)
        self.title = title
        self.main_window = main_window
        # Parent BaseDock.__init__ already performs setup; no need to call again
        self.build_dock()

    def setup(self):
        try:
            # Delegate core docking and Windows menu registration to BaseDock.setup
            super().setup()
            try:
                self.connect()
            except Exception:
                pass
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
        # Add Load Dataset button next to Refresh
        self.btn_load = QPushButton("Load Dataset")
        try:
            self.btn_load.setToolTip("Load default dataset for selected file or the selected dataset")
            self.btn_load.clicked.connect(self._on_load_clicked)
        except Exception:
            pass
        header.addWidget(self.btn_load)
        header.addStretch(1)
        layout.addLayout(header)
        layout.addWidget(self.tree_data)
        self.gb_data_structure.setLayout(layout)

        # Set the group box as the dock's main widget
        self.setWidget(self.gb_data_structure)

    def _on_refresh_clicked(self):
        try:
            # Perform refresh within the dock, using the same functions that populated the tree originally
            self.refresh_data_structure_display()
        except Exception as e:
            QMessageBox.critical(self, "Refresh Error", f"Failed to refresh: {e}")

    def _on_load_clicked(self):
        try:
            tree = getattr(self, 'tree_data', None)
            mw = getattr(self, 'main_window', None)
            if tree is None or mw is None:
                QMessageBox.information(self, "Load Dataset", "Tree or main window not available.")
                return
            item = tree.currentItem()
            if item is None:
                # If no selection, try current file
                fp = getattr(mw, 'current_file_path', None)
                if fp and os.path.exists(fp):
                    self._load_main_data_for_path(fp)
                else:
                    QMessageBox.information(self, "Load Dataset", "No selection or current file.")
                return
            item_type = item.data(0, Qt.UserRole + 2)
            if item_type == "file_root":
                path = item.data(0, Qt.UserRole + 1)
                if path and os.path.exists(path):
                    self._load_main_data_for_path(path)
                else:
                    QMessageBox.information(self, "Load Dataset", "Selected file is not available.")
                return
            # If a dataset/group is selected, try to load that selection; otherwise fall back to default
            full_path = item.data(0, 32)  # Qt.UserRole = 32
            if full_path:
                self._ensure_current_file_from_item(item)
                try:
                    mw.selected_dataset_path = full_path
                except Exception:
                    pass
                mw.start_dataset_load()
            else:
                fp = getattr(mw, 'current_file_path', None)
                if fp and os.path.exists(fp):
                    self._load_main_data_for_path(fp)
                else:
                    QMessageBox.information(self, "Load Dataset", "No dataset path found.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load dataset: {e}")

    def _ensure_current_file_from_item(self, item):
        try:
            mw = getattr(self, 'main_window', None)
            cur = item
            while cur is not None:
                t = cur.data(0, Qt.UserRole + 2)
                if t == "file_root":
                    fp = cur.data(0, Qt.UserRole + 1)
                    if fp and mw is not None:
                        mw.current_file_path = fp
                    break
                cur = cur.parent()
        except Exception:
            pass

    def connect(self):
        try:
            if hasattr(self, 'tree_data') and self.tree_data is not None:
                try:
                    self.tree_data.itemClicked.connect(self._on_tree_item_clicked)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_tree_item_clicked(self, item, column):
        try:
            # If a file root is clicked, load its default main dataset into the active workspace
            item_type = item.data(0, Qt.UserRole + 2)
            mw = getattr(self, 'main_window', None)
            if item_type == "file_root":
                path = item.data(0, Qt.UserRole + 1)
                if path and os.path.exists(path):
                    self._load_main_data_for_path(path)
            else:
                # If a dataset node is clicked, load that dataset into the active workspace
                full_path = item.data(0, 32)
                if full_path and mw is not None:
                    # Ensure the main window knows the current file
                    cur = item
                    while cur is not None:
                        t = cur.data(0, Qt.UserRole + 2)
                        if t == "file_root":
                            fp = cur.data(0, Qt.UserRole + 1)
                            if fp:
                                mw.current_file_path = fp
                            break
                        cur = cur.parent()
                    try:
                        mw.selected_dataset_path = full_path
                    except Exception:
                        pass
                    mw.start_dataset_load()
        except Exception:
            pass

    def refresh_data_structure_display(self, file_path=None):
        """Refresh the data tree by reloading currently listed top-level items (files and folder sections).
        This uses the main window's existing load functions to ensure identical population behavior.
        If a specific file_path is provided, it will attempt to refresh only that entry; otherwise, refreshes all."""
        try:
            tree = getattr(self, 'tree_data', None)
            if tree is None:
                QMessageBox.information(self, "Refresh", "Data tree is not available.")
                return
            mw = getattr(self, 'main_window', None)
            if mw is None:
                QMessageBox.information(self, "Refresh", "Main window is not available.")
                return

            # Snapshot existing top-level items and their paths/types
            snapshot = []
            try:
                if file_path:
                    # If a specific path was requested, try to locate it among top-level items
                    for i in range(tree.topLevelItemCount()):
                        item = tree.topLevelItem(i)
                        path = item.data(0, Qt.UserRole + 1)
                        if path == file_path:
                            item_type = item.data(0, Qt.UserRole + 2)
                            snapshot.append((item_type, path))
                            break
                else:
                    for i in range(tree.topLevelItemCount()):
                        item = tree.topLevelItem(i)
                        item_type = item.data(0, Qt.UserRole + 2)
                        path = item.data(0, Qt.UserRole + 1)
                        snapshot.append((item_type, path))
            except Exception:
                snapshot = []

            # Clear the tree before repopulating
            try:
                tree.clear()
            except Exception:
                pass

            # Rebuild using the same loading functions used initially
            rebuilt_any = False
            for item_type, path in snapshot:
                if not path:
                    continue
                try:
                    if item_type == "file_root" and os.path.exists(path) and hasattr(mw, 'load_single_h5_file'):
                        mw.load_single_h5_file(path)
                        rebuilt_any = True
                    elif item_type == "folder_section" and os.path.isdir(path) and hasattr(mw, 'load_folder_content'):
                        mw.load_folder_content(path)
                        rebuilt_any = True
                except Exception as e:
                    print(f"[DataStructureDock] Refresh failed for {path}: {e}")

            # Fallback: if nothing was rebuilt, try the current file
            if not rebuilt_any:
                fp = getattr(mw, 'current_file_path', None)
                try:
                    if fp and os.path.exists(fp) and hasattr(mw, 'load_single_h5_file'):
                        mw.load_single_h5_file(fp)
                        rebuilt_any = True
                except Exception:
                    pass

            if rebuilt_any:
                # Suppress popup per user request - update status silently if available
                if hasattr(self, 'update_status'):
                    try:
                        self.update_status("Data structure refreshed.")
                    except Exception:
                        pass
                # If a dataset is selected, start background load to keep UI responsive
                try:
                    if mw is not None and getattr(mw, 'selected_dataset_path', None):
                        mw.start_dataset_load()
                except Exception:
                    pass
            else:
                # Suppress popup per user request - update status silently if available
                if hasattr(self, 'update_status'):
                    try:
                        self.update_status("Nothing to refresh.")
                    except Exception:
                        pass
                # If a dataset is selected, start background load to keep UI responsive
                try:
                    if mw is not None and getattr(mw, 'selected_dataset_path', None):
                        mw.start_dataset_load()
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, "Refresh Error", f"Failed to refresh: {e}")

    def _load_data_structure(self):
        try: 
            folder_name = os.path.basename(self.current_file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load folder: {str(e)}")
            self.update_status("Failed to load folder")

    def _populate_tree_recursive(self):
        raise NotImplementedError

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
