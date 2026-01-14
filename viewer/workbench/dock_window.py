#!/usr/bin/env python3
"""
Dock Window
A lightweight secondary QMainWindow intended to host dockable tools.
Modeless (does not disable the main Workbench window), can be filled
with QDockWidget-based panels like ROI Plot and ROI Math.
"""

from PyQt5.QtWidgets import QMainWindow, QWidget, QDockWidget, QLabel
from PyQt5.QtCore import Qt


class DockWindow(QMainWindow):
    """
    Secondary window for hosting dockable tools.

    Attributes:
        main (QMainWindow): Reference to the primary Workbench window for callbacks.
    """

    def __init__(self, main_window, title: str = None, width: int = 1000, height: int = 700):
        super().__init__(parent=None)  # top-level, modeless
        self.main = main_window
        self.setWindowTitle(title or "Dock Window")
        self.resize(width, height)

        # Central placeholder; docks will live around this
        central = QWidget(self)
        self.setCentralWidget(central)

        # Create an initial empty dock so users can dock panels into this window
        try:
            self.empty_dock = QDockWidget("Dock", self)
            self.empty_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            placeholder = QLabel("Empty Dock — you can add dockable panels here", self.empty_dock)
            placeholder.setAlignment(Qt.AlignCenter)
            self.empty_dock.setWidget(placeholder)
            self.addDockWidget(Qt.RightDockWidgetArea, self.empty_dock)
            self.empty_dock.show()
        except Exception:
            # Even if dock creation fails, the window remains usable
            pass

        # Ensure deletion on close; Workbench keeps reference list to avoid GC while open
        try:
            self.setAttribute(Qt.WA_DeleteOnClose, True)
        except Exception:
            pass

    def show_and_focus(self) -> None:
        """Show the window modeless and bring it to the foreground."""
        try:
            self.show()
            self.raise_()
            self.activateWindow()
        except Exception:
            # Best-effort foregrounding only
            self.show()
