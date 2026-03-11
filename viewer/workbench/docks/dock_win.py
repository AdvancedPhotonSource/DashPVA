#!/usr/bin/env python3
"""
DockWinDock
A small control surface dock that lets users create up to two additional
modeless DockWindow instances. It mirrors the remaining capacity in its
title, a push button, and a segmented Windows menu action.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QMessageBox, QAction
from PyQt5.QtCore import Qt, QTimer

from viewer.workbench.docks.base_dock import BaseDock


class DockWinDock(BaseDock):
    """
    Dock for creating additional windows with a max cap of 2.

    Title shows dynamic counter: "Add Window (n/2)" where n is the number of
    open DockWindow instances tracked by the Workbench main window.
    """

    MAX_WINDOWS = 2

    def __init__(self, title: str = "Add Window", main_window=None, segment_name: str = "other", dock_area=Qt.RightDockWidgetArea, show: bool = False):
        # Store title separately so BaseDock.setup can wire the dock and toggle
        super().__init__(title=title, main_window=main_window, segment_name=segment_name, dock_area=dock_area, show=show)

        # UI
        self._container = QWidget(self)
        self._layout = QVBoxLayout(self._container)
        self.btn_add_window = QPushButton(title, self._container)
        self.btn_add_window.clicked.connect(self._on_add_window)
        self._layout.addWidget(self.btn_add_window)
        self.setWidget(self._container)

        # Separate action under Windows segmented menu mirroring the same label/state
        self.action_add_window_menu = None
        try:
            self.action_add_window_menu = self._create_menu_action()
        except Exception:
            self.action_add_window_menu = None

        # Fallback timer to refresh counts in case signals are missed
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(1000)
        self._refresh_timer.timeout.connect(self.refresh_counts)
        self._refresh_timer.start()

        # Initial refresh to sync texts and states
        try:
            self.refresh_counts()
        except Exception:
            pass

    # --- Internals ---
    def _get_count(self) -> int:
        """Safely return the number of currently open DockWindow instances."""
        try:
            wins = getattr(self.main_window, "_dock_windows", None)
            if isinstance(wins, (list, tuple)):
                return len(wins)
        except Exception:
            pass
        return 0

    def refresh_counts(self) -> None:
        """Recompute count, update dock title, button/action text, and enabled state."""
        try:
            n = max(0, int(self._get_count()))
        except Exception:
            n = 0
        capped = min(n, self.MAX_WINDOWS)
        label = f"Add Window ({capped}/{self.MAX_WINDOWS})"

        # Update dock title
        try:
            self.setWindowTitle(label)
        except Exception:
            pass
        # Update the BaseDock-created visibility toggle text if available
        try:
            if hasattr(self, 'action_window_dock') and self.action_window_dock is not None:
                self.action_window_dock.setText(label)
        except Exception:
            pass
        # Update button
        try:
            if self.btn_add_window is not None:
                self.btn_add_window.setText(label)
                self.btn_add_window.setEnabled(n < self.MAX_WINDOWS)
        except Exception:
            pass
        # Update separate menu action
        try:
            if self.action_add_window_menu is not None:
                self.action_add_window_menu.setText(label)
                self.action_add_window_menu.setEnabled(n < self.MAX_WINDOWS)
        except Exception:
            pass

    def _on_add_window(self) -> None:
        """Create a new DockWindow via main_window if below the cap; otherwise show info."""
        n = self._get_count()
        if n >= self.MAX_WINDOWS:
            try:
                QMessageBox.information(self, "Add Window", f"Maximum of {self.MAX_WINDOWS} additional windows is reached.")
            except Exception:
                pass
            # Keep disabled state; ensure labels remain correct
            self.refresh_counts()
            return
        try:
            # Delegate creation to main window; it enforces its own cap as well
            if hasattr(self.main_window, 'create_dock_window_and_show'):
                self.main_window.create_dock_window_and_show()
        finally:
            # Recompute counts regardless of outcome
            try:
                self.refresh_counts()
            except Exception:
                pass

    def _create_menu_action(self):
        """Create a separate QAction under Windows->segment that mirrors the button.
        Uses BaseWindow.add_windows_menu_action for segmented placement.
        """
        act = QAction(self.windowTitle(), self)
        act.setToolTip("Open an additional window for dockable tools")
        act.triggered.connect(self._on_add_window)
        try:
            if hasattr(self.main_window, 'add_windows_menu_action'):
                self.main_window.add_windows_menu_action(act, segment_name=self.segment_name)
        except Exception:
            pass
        return act
