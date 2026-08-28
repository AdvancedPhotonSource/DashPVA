#!/usr/bin/env python3
# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

"""
Dock Window
A lightweight secondary QMainWindow intended to host dockable tools.
Modeless (does not disable the main Workbench window), can be filled
with QDockWidget-based panels like ROI Plot and ROI Math.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMenu,
    QPushButton,
    QToolBar,
    QToolButton,
    QWidget,
)


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

        # Toolbar with action buttons pinned above the dock areas
        toolbar = QToolBar("Actions", self)
        toolbar.setMovable(False)
        self.btn_get_2d_docks = QPushButton("Get 2D Docks")
        self.btn_get_3d_docks = QPushButton("Get 3D Docks")
        self.btn_get_2d_docks.setProperty("role", "info")
        self.btn_get_3d_docks.setProperty("role", "success")
        self.btn_get_2d_docks.clicked.connect(lambda: self._move_docks_by_segment("2d"))
        self.btn_get_3d_docks.clicked.connect(lambda: self._move_docks_by_segment("3d"))
        toolbar.addWidget(self.btn_get_2d_docks)
        toolbar.addWidget(self.btn_get_3d_docks)
        toolbar.addSeparator()
        self.btn_send_2d_docks = QPushButton("Send 2D to Workbench")
        self.btn_send_3d_docks = QPushButton("Send 3D to Workbench")
        self.btn_send_2d_docks.setProperty("role", "warning")
        self.btn_send_3d_docks.setProperty("role", "error")
        self.btn_send_2d_docks.clicked.connect(lambda: self._return_docks_by_segment("2d"))
        self.btn_send_3d_docks.clicked.connect(lambda: self._return_docks_by_segment("3d"))
        toolbar.addWidget(self.btn_send_2d_docks)
        toolbar.addWidget(self.btn_send_3d_docks)
        toolbar.addSeparator()
        self._view_menu = QMenu(self)
        self._view_menu.aboutToShow.connect(self._populate_view_menu)
        btn_view = QToolButton()
        btn_view.setText("View")
        btn_view.setPopupMode(QToolButton.InstantPopup)
        btn_view.setMenu(self._view_menu)
        toolbar.addWidget(btn_view)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Allow docks to be nested side by side within this window
        self.setDockNestingEnabled(True)

        # Central placeholder; docks will live around this
        self.setCentralWidget(QWidget(self))

        # Ensure deletion on close; Workbench keeps reference list to avoid GC while open
        try:
            self.setAttribute(Qt.WA_DeleteOnClose, True)
        except Exception:
            pass

    def _populate_view_menu(self):
        """Rebuild the View menu with a toggle entry for every dock in this window."""
        self._view_menu.clear()
        docks = [d for d in self.findChildren(QDockWidget)
                 if getattr(d, 'segment_name', None) is not None]
        if not docks:
            self._view_menu.addAction("No docks").setEnabled(False)
            return
        for dock in docks:
            action = self._view_menu.addAction(dock.windowTitle())
            action.setCheckable(True)
            action.setChecked(dock.isVisible())
            action.triggered.connect(lambda checked, d=dock: d.setVisible(checked))

    DOCKS_PER_COLUMN = 3

    def _move_docks_by_segment(self, segment: str) -> None:
        """Transfer docks from the main window, arranged in columns of DOCKS_PER_COLUMN rows."""
        try:
            docks = [d for d in self.main.findChildren(QDockWidget)
                     if getattr(d, 'segment_name', None) == segment]
            if not docks:
                return
            for dock in docks:
                self.main.removeDockWidget(dock)

            N = self.DOCKS_PER_COLUMN
            columns = [docks[i:i + N] for i in range(0, len(docks), N)]
            # Native areas for the first two columns; split for any beyond that
            native_areas = [Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea]
            col_tops = []
            for col_idx, col_docks in enumerate(columns):
                top = col_docks[0]
                if col_idx < len(native_areas):
                    self.addDockWidget(native_areas[col_idx], top)
                else:
                    self.splitDockWidget(col_tops[col_idx - 1], top, Qt.Horizontal)
                top.show()
                col_tops.append(top)
                # Fill rows — show each before splitting the next
                for row_idx in range(1, len(col_docks)):
                    self.splitDockWidget(col_docks[row_idx - 1], col_docks[row_idx], Qt.Vertical)
                    col_docks[row_idx].show()
        except Exception:
            pass

    def _return_docks_by_segment(self, segment: str) -> None:
        """Send docks of a given segment from this window back to the main window."""
        try:
            docks = [d for d in self.findChildren(QDockWidget)
                     if getattr(d, 'segment_name', None) == segment]
            for dock in docks:
                try:
                    self.removeDockWidget(dock)
                    area = getattr(dock, 'dock_area', Qt.RightDockWidgetArea)
                    self.main.addDockWidget(area, dock)
                    dock.show()
                except Exception:
                    pass
        except Exception:
            pass

    def _return_docks_to_main(self) -> None:
        """Return all segment docks in this window back to the main window."""
        try:
            docks = [d for d in self.findChildren(QDockWidget)
                     if getattr(d, 'segment_name', None) is not None]
            for dock in docks:
                try:
                    self.removeDockWidget(dock)
                    area = getattr(dock, 'dock_area', Qt.RightDockWidgetArea)
                    self.main.addDockWidget(area, dock)
                    dock.show()
                except Exception:
                    pass
        except Exception:
            pass

    def closeEvent(self, event):
        # Only return docks if the main window is still open
        if self.main is not None and self.main.isVisible():
            self._return_docks_to_main()
        super().closeEvent(event)

    def show_and_focus(self) -> None:
        """Show the window modeless and bring it to the foreground."""
        try:
            self.show()
            self.raise_()
            self.activateWindow()
        except Exception:
            # Best-effort foregrounding only
            self.show()
