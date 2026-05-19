from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QDockWidget, QMenu


class BaseDock(QDockWidget):
    # Subclasses that hold expandable widgets (trees, lists) should set this
    # to False so the dock can grow vertically inside a splitter.
    _lock_height_to_widget = True

    def __init__(self, title="", main_window=None, segment_name=None, dock_area=Qt.LeftDockWidgetArea, show: bool = True):
        super().__init__(title, main_window)
        self.title = title
        self.main_window = main_window
        self.segment_name = (segment_name or "").strip().lower() if segment_name is not None else None
        self.dock_area = dock_area
        # Store initial visibility preference
        self._initial_show = bool(show)
        self.setup()

    def setup(self):
        # Dock
        self.setWindowTitle(self.title)
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.main_window.addDockWidget(self.dock_area, self)
        # Apply initial visibility before registering Windows menu action,
        # so the QAction's checked state reflects the dock's current visibility.
        if not getattr(self, "_initial_show", True):
            self.hide()
        else:
            self.show()

        # Register dock toggle under segmented Windows menu via BaseWindow helper
        self.action_window_dock = self.main_window.add_dock_toggle_action(
            self, self.title, segment_name=self.segment_name
        )

    def setWidget(self, widget):
        super().setWidget(widget)
        if self._lock_height_to_widget:
            # After the widget's layout settles, lock the dock's max height to the
            # smallest size that still fits its contents.
            QTimer.singleShot(0, self._lock_to_min_height)

    def _lock_to_min_height(self):
        w = self.widget()
        if w is None:
            return
        hint = w.minimumSizeHint().height()
        if hint <= 0:
            hint = w.sizeHint().height()
        if hint > 0:
            self.setMaximumHeight(hint + 40)  # padding for title bar + chrome

    def _current_host(self):
        """Return the QMainWindow currently hosting this dock."""
        if self in self.main_window.findChildren(QDockWidget):
            return self.main_window
        for win in getattr(self.main_window, '_dock_windows', []):
            if self in win.findChildren(QDockWidget):
                return win
        return None

    def _send_to(self, target_window, area):
        """Move this dock from its current host to target_window."""
        host = self._current_host()
        if host is not None:
            host.removeDockWidget(self)
        target_window.addDockWidget(area, self)
        self.show()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        dock_windows = getattr(self.main_window, '_dock_windows', [])
        host = self._current_host()

        if host is self.main_window:
            for i, win in enumerate(dock_windows, 1):
                action = menu.addAction(f"Send to Dock Window {i}")
                action.triggered.connect(lambda checked, w=win: self._send_to(w, Qt.LeftDockWidgetArea))
        elif host is not None:
            action = menu.addAction("Send to Workbench")
            action.triggered.connect(lambda: self._send_to(self.main_window, self.dock_area))

        if not menu.isEmpty():
            menu.exec_(event.globalPos())
        else:
            super().contextMenuEvent(event)
