#!/usr/bin/env python3
"""
Base Window Class
A base class for all main window interfaces in the DashPVA application.
Provides common functionality and consistent behavior across all windows.
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import uic
from database import DatabaseInterface
from viewer.documentation.dialog import DocumentationDialog

di = DatabaseInterface()

# Add the project root to the Python path
project_root = Path(__file__).parent.parent

class BaseWindow(QMainWindow):
    """
    Base class for all main windows in the DashPVA application.
    Provides common functionality like file operations, UI loading, and standard menu actions.
    """
    
    # Signals
    file_opened = pyqtSignal(str)  # Emitted when a file is opened
    file_saved = pyqtSignal(str)   # Emitted when a file is saved
    
    def __init__(self, ui_file_name=None, viewer_name=None):
        """
        Initialize the base window.
        
        Args:
            ui_file_name (str): Name of the UI file to load (without path)
            viewer_name (str, optional): Human-friendly name of the viewer for status messages
        """
        super().__init__()
        self.ui_file_name = ui_file_name
        self.current_file_path = None
        self.viewer_name = viewer_name
        
        if ui_file_name:
            self.load_ui()
        
        self.setup_base_connections()
        
    def load_ui(self):
        """Load the UI file for this window."""
        if not self.ui_file_name:
            return
            
        ui_file = project_root / "gui" / self.ui_file_name
        if ui_file.exists():
            uic.loadUi(str(ui_file), self)
        else:
            QMessageBox.critical(self, "Error", f"UI file not found: {ui_file}")
            sys.exit(1)
            
    def setup_base_connections(self):
        """Set up connections for standard menu actions."""
        # Only connect if the actions exist (they should from base_mainwindow.ui)
        if hasattr(self, 'actionOpen'):
            self.actionOpen.triggered.connect(self.open_file)
        if hasattr(self, 'actionOpenFolder'):
            self.actionOpenFolder.triggered.connect(self.open_folder)
        if hasattr(self, 'actionSave'):
            self.actionSave.triggered.connect(self.save_file)
        if hasattr(self, 'actionExit'):
            self.actionExit.triggered.connect(self.close)
        # Documentation menu (wired here; content discovery handled by DocumentationDialog)
        if hasattr(self, 'actionOpenDocumentation'):
            self.actionOpenDocumentation.triggered.connect(self.open_documentation)
            
    def open_file(self):
        """
        Handle File -> Open action.
        Shows file dialog and calls load_file_content if a file is selected.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            self.get_file_filters()
        )
        if file_path:
            self.current_file_path = file_path
            self.load_file_content(file_path)
            self.file_opened.emit(file_path)
            
    def save_file(self):
        """
        Handle File -> Save action.
        Shows file dialog and calls save_file_content if a path is selected.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            self.get_file_filters()
        )
        if file_path:
            self.current_file_path = file_path
            self.save_file_content(file_path)
            self.file_saved.emit(file_path)
            
    def get_file_filters(self):
        """
        Get file filters for open/save dialogs.
        Override in subclasses to provide specific file types.
        
        Returns:
            str: File filter string for QFileDialog
        """
        return "All Files (*)"
        
    def load_file_content(self, file_path):
        """
        Load content from the specified file.
        Override in subclasses to implement specific loading logic.
        
        Args:
            file_path (str): Path to the file to load
        """
        # Base implementation - override in subclasses
        self.update_status(f"Loaded: {os.path.basename(file_path)}")
        
    def open_folder(self):
        """
        Handle File -> Open Folder action.
        Shows folder dialog and calls load_folder_content if a folder is selected.
        Override in subclasses to implement specific folder loading logic.
        """
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            ""
        )
        if folder_path:
            self.load_folder_content(folder_path)
    
    def load_folder_content(self, folder_path):
        """
        Load content from the specified folder.
        Override in subclasses to implement specific folder loading logic.
        
        Args:
            folder_path (str): Path to the folder to load
        """
        # Base implementation - override in subclasses
        self.update_status(f"Opened folder: {os.path.basename(folder_path)}")
        
    def save_file_content(self, file_path):
        """
        Save content to the specified file.
        Override in subclasses to implement specific saving logic.
        
        Args:
            file_path (str): Path to save the file to
        """
        # Base implementation - override in subclasses
        self.update_status(f"Saved: {os.path.basename(file_path)}")

    def open_documentation(self):
        """Open the documentation dialog without disabling the main window (modeless)."""
        try:
            # Keep a reference so it isn't garbage collected
            self._documentation_dialog = DocumentationDialog(self)
            dlg = self._documentation_dialog
            # Delete dialog on close to avoid leaks
            try:
                dlg.setAttribute(Qt.WA_DeleteOnClose, True)
            except Exception:
                pass
            # Auto-discovery is handled by the dialog; pass this viewer instance
            dlg.open_for_viewer(self)
            dlg.setWindowTitle("Documentation")
            dlg.resize(900, 700)
            # Show modeless, do not block/disable main window
            dlg.show()
        except Exception as e:
            self.update_status(f"Failed to open documentation: {e}", level='error')
        
    def update_status(self, message, level: str = 'info', source: str = None):
        """
        Update the status label with a message and include the viewer name.
        Also logs error-like messages to error_output.txt with viewer context.
        
        Args:
            message (str): Status message to display
            level (str): Optional level ('info', 'warning', 'error')
            source (str): Optional override for viewer/source name; defaults to self.viewer_name
        """
        # Compose prefixed message
        src = source or getattr(self, 'viewer_name', None) or self.__class__.__name__
        prefix = f"[{src}] "
        full_msg = f"{prefix}{message}" if isinstance(message, str) else message
        
        # Update status label if it exists
        if hasattr(self, 'label_status'):
            try:
                self.label_status.setText(full_msg)
            except Exception:
                # Fallback to plain message if label does not accept complex types
                self.label_status.setText(str(message))
        
        # Log to error_output.txt if message indicates an error or failure
        try:
            import datetime
            error_file = project_root / "error_output.txt"
            should_log = False
            if isinstance(message, str):
                msg_lower = message.lower()
                should_log = (level in ('error', 'warning')) or ('error' in msg_lower) or ('failed' in msg_lower)
            if should_log:
                with open(error_file, "a") as f:
                    f.write(f"[{datetime.datetime.now().isoformat()}] {prefix}{message}\n")
        except Exception:
            # Avoid raising during logging
            pass
            
    def setup_window_properties(self, title, width=800, height=600):
        """
        Set up basic window properties.
        
        Args:
            title (str): Window title
            width (int): Window width
            height (int): Window height
        """
        self.setWindowTitle(title)
        # If no explicit viewer_name was set, derive a friendly name from the title
        if not getattr(self, 'viewer_name', None) and isinstance(title, str):
            # Use the segment before the first ' - ' as the viewer name
            self.viewer_name = title.split(' - ')[0].strip() or self.__class__.__name__
        self.resize(width, height)
        
    def closeEvent(self, event):
        """
        Handle window close event.
        Override in subclasses to add custom close behavior.
        """
        # Base implementation - just accept the close event
        event.accept()
        
    def set_viewer_name(self, name: str) -> None:
        """Set or update the viewer name used in status messages."""
        try:
            self.viewer_name = str(name) if name else None
        except Exception:
            self.viewer_name = None
