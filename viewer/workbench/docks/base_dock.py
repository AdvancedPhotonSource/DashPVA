from PyQt5.QtWidgets import QDockWidget, QAction
from PyQt5.QtCore import Qt

class BaseDock(QDockWidget):
    def __init__(self, title="", main_window=None, segment_name=None, dock_area=Qt.LeftDockWidgetArea):
        super().__init__(title, main_window)
        self.title = title
        self.main_window = main_window
        self.segment_name = (segment_name or "").strip().lower() if segment_name is not None else None
        self.dock_area = dock_area
        self.setup()

    def setup(self):
        # Dock
        self.setWindowTitle(self.title)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.main_window.addDockWidget(self.dock_area, self)

        # Register dock toggle under segmented Windows menu via BaseWindow helper
        self.action_window_dock = self.main_window.add_dock_toggle_action(
            self, self.title, segment_name=self.segment_name
        )
        self.visibilityChanged.connect(lambda visible: self.action_window_dock.setChecked(bool(visible)))
