from PyQt5.QtWidgets import QWidget
from PyQt5 import uic

class BaseTab(QWidget):
    """
    Base class for all tabs in the Workbench.
    Provides common functionality and a consistent interface.
    """
    def __init__(self, ui_file, parent=None, main_window=None, title=""):
        super().__init__(parent)
        self.main_window = main_window
        self.title = title
        uic.loadUi(ui_file, self)
        self.setObjectName(self.__class__.__name__) # Set object name for easier identification
        self.setup()

    def setup(self):
        self.main_window.tabWidget_analysis.addTab(self, self.title)

    def on_tab_selected(self):
        """
        Called when this tab is selected.
        Can be overridden by subclasses to perform tab-specific actions.
        """
        pass

    def on_tab_deselected(self):
        """
        Called when this tab is deselected.
        Can be overridden by subclasses to perform tab-specific actions.
        """
        pass

    def update_data(self, data_path: str):
        """
        Called when new data is loaded or the selected dataset changes.
        Subclasses should implement this to update their display.
        """
        pass

    def clear_data(self):
        """
        Called when the HDF5 file is closed or cleared.
        Subclasses should implement this to clear their display.
        """
        pass

    def get_tab_name(self) -> str:
        """
        Returns the display name for the tab.
        """
        return self.__class__.__name__.replace('Tab', '') # Default to class name without 'Tab'
