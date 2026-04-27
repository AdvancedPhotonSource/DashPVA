from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal, Qt, QRect, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen


class RangeSlider(QWidget):
    """Horizontal slider with two handles for a min/max range (integer 0-1000)."""
    range_changed = pyqtSignal(int, int)  # low, high

    _HANDLE_R = 7
    _TRACK_H = 5
    _MARGIN = 12

    def __init__(self, minimum=0, maximum=1000, parent=None):
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self._low = minimum
        self._high = maximum
        self._dragging = None  # 'low' | 'high'
        self.setMinimumHeight(28)
        self.setCursor(Qt.ArrowCursor)

    # ── public API ──────────────────────────────────────────────────────────
    def low(self):
        return self._low

    def high(self):
        return self._high

    def setLow(self, val, emit=True):
        self._low = max(self._minimum, min(int(val), self._high))
        self.update()
        if emit:
            self.range_changed.emit(self._low, self._high)

    def setHigh(self, val, emit=True):
        self._high = max(self._low, min(int(val), self._maximum))
        self.update()
        if emit:
            self.range_changed.emit(self._low, self._high)

    def setFullRange(self):
        self._low = self._minimum
        self._high = self._maximum
        self.update()
        self.range_changed.emit(self._low, self._high)

    # ── geometry helpers ────────────────────────────────────────────────────
    def _track_rect(self):
        cy = self.height() // 2
        m = self._MARGIN
        return QRect(m, cy - self._TRACK_H // 2,
                     self.width() - 2 * m, self._TRACK_H)

    def _x_for(self, val):
        tr = self._track_rect()
        span = max(1, self._maximum - self._minimum)
        return tr.left() + int((val - self._minimum) / span * tr.width())

    def _val_for_x(self, x):
        tr = self._track_rect()
        frac = (x - tr.left()) / max(1, tr.width())
        frac = max(0.0, min(1.0, frac))
        return int(round(self._minimum + frac * (self._maximum - self._minimum)))

    def _nearest_handle(self, x):
        if abs(x - self._x_for(self._low)) <= abs(x - self._x_for(self._high)):
            return 'low'
        return 'high'

    # ── painting ────────────────────────────────────────────────────────────
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tr = self._track_rect()
        cy = self.height() // 2

        # groove
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(70, 70, 70))
        p.drawRoundedRect(tr, 3, 3)

        # selected range
        xl = self._x_for(self._low)
        xh = self._x_for(self._high)
        sel = QRect(xl, tr.top(), xh - xl, tr.height())
        p.setBrush(QColor(60, 120, 200))
        p.drawRoundedRect(sel, 2, 2)

        # handles
        r = self._HANDLE_R
        for x in (xl, xh):
            p.setBrush(QColor(210, 210, 210))
            p.setPen(QPen(QColor(90, 90, 90), 1))
            p.drawEllipse(QPoint(x, cy), r, r)

        p.end()

    # ── interaction ─────────────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._dragging = self._nearest_handle(ev.x())

    def mouseMoveEvent(self, ev):
        if not self._dragging:
            return
        val = self._val_for_x(ev.x())
        if self._dragging == 'low':
            new = max(self._minimum, min(val, self._high))
            if new != self._low:
                self._low = new
                self.update()
                self.range_changed.emit(self._low, self._high)
        else:
            new = max(self._low, min(val, self._maximum))
            if new != self._high:
                self._high = new
                self.update()
                self.range_changed.emit(self._low, self._high)

    def mouseReleaseEvent(self, _ev):
        self._dragging = None
