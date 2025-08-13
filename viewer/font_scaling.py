import sys
from PyQt5.QtCore import QObject, QEvent
from PyQt5.QtGui   import QFont
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QMainWindow
)

class GlobalFontScaling(QObject):
    def __init__(self, app: QApplication):
        super().__init__(app)
        self.app = app

        self.base_width:      int | None = None
        self.base_font_size:  float | None = None

        self._in_update = False           # re-entrance guard
        app.installEventFilter(self)


    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Resize
                and isinstance(watched, QWidget)
                and watched.isWindow()):
            # ignore tiny jitters and re-entrant calls -----------------
            if self._in_update or event.size().width() <= 0:
                return False

            self._in_update = True
            try:
                self._rescale(watched, event.size().width())
            finally:
                self._in_update = False
        return False        # let Qt continue normal processing


    # ---------------------------------------------------------------- #
    # Do the actual scaling                                             #
    # ---------------------------------------------------------------- #
    def _rescale(self, window: QWidget, new_width: int):
        # 1) establish reference values on first invocation ------------
        if self.base_width is None:
            self.base_width = new_width

        if self.base_font_size is None:
            f = window.font()
            self.base_font_size = (
                f.pointSizeF() if f.pointSizeF() > 0 else f.pixelSize()
            ) or 12                     # fallback: 12 pt

        # 2) compute scale factor --------------------------------------
        factor   = new_width / self.base_width
        font_px  = max(1, int(self.base_font_size * factor))

        # 3) application font -----------------------------------------
        new_font = QFont(window.font().family(), font_px)
        self.app.setFont(new_font)

        # stylesheet so even widgets with their own font-family inherit the size
        self.app.setStyleSheet(f"* {{ font-size: {font_px}px; }}")

        # 4) adjust widgets that use a *fixed* size --------------------
        for w in self.app.allWidgets():
            if w.isWindow():
                continue
            if w.minimumWidth() == w.maximumWidth():
                w.setFixedWidth(int(w.minimumWidth() * factor))
            if w.minimumHeight() == w.maximumHeight():
                w.setFixedHeight(int(w.minimumHeight() * factor))