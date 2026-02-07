import sys
import os
import math
import time
from PyQt5.QtWidgets import (QWidget, QApplication, QVBoxLayout, QLabel, 
                             QMainWindow, QFrame, QSizePolicy, QHBoxLayout)
from PyQt5.QtCore import (Qt, QTimer, pyqtSignal, QRectF, QSize, QObject, QPointF)
from PyQt5.QtGui import (QPainter, QColor, QBrush, QPen, QFont, QLinearGradient)

# =============================================================================
#  CONFIGURATION & PV DEFINITIONS
# =============================================================================
# Define your two separate status channels here.
# If these PVs are not reachable, the UI will default to "SYSTEM UNREADY".

PV_TRAINING_STATUS = "vit:status:training" 
# Expected Values:
# 0: Not Tuned (Grey)
# 1: Unusable / Poor Loss (Red)
# 2: Usable / Fair (Yellow)
# 3: Great / High Precision (Green)

PV_TRANSFER_STATUS = "vit:status:transfer"
# Expected Values:
# 0: Idle / Pending
# 1: Transferring (Blue Scan Animation)
# 2: Transfer Complete (Static Blue or return to Training Color)

# EPICS Connection Settings (For Sector 26)
os.environ['EPICS_CA_ADDR_LIST'] = '10.54.116.22'
os.environ['EPICS_CA_AUTO_ADDR_LIST'] = 'NO'

# =============================================================================
#  EPICS HANDLER
# =============================================================================
try:
    import epics
    EPICS_AVAILABLE = True
except ImportError:
    EPICS_AVAILABLE = False
    print("[Analysis] Warning: pyepics not found. Running in SIMULATION mode.")

class EpicsListener(QObject):
    """
    Handles threaded EPICS callbacks and emits Qt signals to the GUI.
    This prevents the GUI from freezing during network lag.
    """
    status_updated = pyqtSignal(str, int)  # pv_type ('train' or 'xfer'), value
    connection_changed = pyqtSignal(bool)   # connected?

    def __init__(self):
        super().__init__()
        self.pv_train = None
        self.pv_xfer = None
        self.connected_train = False
        self.connected_xfer = False

    def start(self):
        if not EPICS_AVAILABLE:
            return

        # Initialize PV objects with connection and value callbacks
        self.pv_train = epics.PV(PV_TRAINING_STATUS, 
                                 callback=self._on_train_change, 
                                 connection_callback=self._on_conn_change)
        
        self.pv_xfer = epics.PV(PV_TRANSFER_STATUS, 
                                callback=self._on_xfer_change,
                                connection_callback=self._on_conn_change)

    def _on_train_change(self, value, **kwargs):
        self.status_updated.emit('train', int(value) if value is not None else 0)

    def _on_xfer_change(self, value, **kwargs):
        self.status_updated.emit('xfer', int(value) if value is not None else 0)

    def _on_conn_change(self, pvname, connected, **kwargs):
        # We consider the system "Ready" only if both PVs are connected
        if pvname == PV_TRAINING_STATUS:
            self.connected_train = connected
        elif pvname == PV_TRANSFER_STATUS:
            self.connected_xfer = connected
        
        # Emit combined connection state
        self.connection_changed.emit(self.connected_train and self.connected_xfer)

# =============================================================================
#  CUSTOM WIDGET: ANIMATED SERVER RACK
# =============================================================================
class ServerRackWidget(QWidget):
    """
    Visualizes the HPC status using a stylized server rack.
    Priority Logic:
    1. If Disconnected -> Dark Grey / "Offline"
    2. If Transferring -> Blue "Scanning" Animation overlay
    3. Else -> Show Training Quality (Red/Yellow/Green)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # State
        self.is_connected = False if EPICS_AVAILABLE else True
        self.train_code = 0
        self.xfer_code = 0
        self.anim_phase = 0.0

        # Animation Timer (always running for smooth effects)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(50) # 20 FPS

    def set_connection_status(self, connected):
        self.is_connected = connected
        self.update()

    def set_training_status(self, code):
        self.train_code = code
        self.update()

    def set_transfer_status(self, code):
        self.xfer_code = code
        self.update()

    def _animate(self):
        # Increment phase for ripple/scan effects
        self.anim_phase += 0.2
        if self.anim_phase > 20.0:
            self.anim_phase = 0.0
        
        # Only trigger repaint if we are in an animated state
        # (Transferring=1, or Training Great=3 for ripple)
        if self.xfer_code == 1 or self.train_code == 3:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # --- 1. Draw Casing ---
        margin = 15
        rack_rect = QRectF(margin, margin, w - 2*margin, h - 2*margin)
        
        # Case Color: Darker if disconnected
        case_color = QColor("#2d3436") if self.is_connected else QColor("#1e272e")
        painter.setBrush(QBrush(case_color))
        
        # Border Color: Red glow if disconnected
        if self.is_connected:
            border_color = QColor("#636e72")
        else:
            border_color = QColor(255, 0, 0, 100)
        painter.setPen(QPen(border_color, 4))
        
        painter.drawRoundedRect(rack_rect, 6, 6)
        
        # --- 2. Determine Display Mode ---
        # Logic: Transferring (1) overrides Training status visual
        is_transferring = (self.xfer_code == 1)
        
        if not self.is_connected:
            base_color = QColor("#2f3640") # Offline Grey
        elif is_transferring:
            base_color = QColor("#0984e3") # Transfer Blue
        else:
            # Map Training Code to Color
            if self.train_code == 1:   base_color = QColor("#d63031") # Red (Bad)
            elif self.train_code == 2: base_color = QColor("#fdcb6e") # Yellow (Fair)
            elif self.train_code == 3: base_color = QColor("#00b894") # Green (Great)
            else:                      base_color = QColor("#636e72") # Grey (Not Tuned)

        # --- 3. Draw Rack Units (Lights) ---
        unit_height = (rack_rect.height() - 20) / 10
        unit_width = rack_rect.width() - 20
        start_x = rack_rect.x() + 10

        for i in range(10):
            y_pos = rack_rect.bottom() - 10 - ((i + 1) * unit_height)
            slot_rect = QRectF(start_x, y_pos, unit_width, unit_height - 2)
            
            # -- Lighting Logic --
            lit_intensity = 0
            
            if not self.is_connected:
                lit_intensity = 0 # All dark
            elif is_transferring:
                # Scanning Effect (Knight Rider style moving up)
                scan_pos = (self.anim_phase * 1.5) % 14 
                dist = abs(scan_pos - i)
                if dist < 2.5:
                    lit_intensity = max(0, 255 - (dist * 100))
                else:
                    lit_intensity = 20 # Dim background
            else:
                # Training Status Logic
                if self.train_code == 0: lit_intensity = 0
                elif self.train_code == 1: lit_intensity = 255 if i < 3 else 20
                elif self.train_code == 2: lit_intensity = 255 if i < 7 else 20
                elif self.train_code == 3: 
                    # Great: Full green with slight ripple
                    ripple = math.sin(self.anim_phase + i)
                    lit_intensity = 180 + (ripple * 50)
            
            # -- Draw the Slot --
            c = QColor(base_color)
            if lit_intensity > 0:
                # Apply intensity to alpha/brightness
                painter.setBrush(c)
                if lit_intensity > 200:
                    painter.setBrush(c.lighter(130)) # Glow
            else:
                painter.setBrush(QColor("#353b48")) # Dark slot background
                
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(slot_rect, 2, 2)

        # --- 4. Transfer Icon / Text Overlay ---
        if is_transferring:
            painter.setPen(QColor("white"))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(rack_rect, Qt.AlignCenter, "SENDING...")

        # --- 5. Glass Reflection ---
        reflection_rect = QRectF(rack_rect.x(), rack_rect.y(), rack_rect.width()/2.5, rack_rect.height())
        grad = QLinearGradient(reflection_rect.topLeft(), reflection_rect.bottomRight())
        grad.setColorAt(0, QColor(255, 255, 255, 20))
        grad.setColorAt(1, QColor(255, 255, 255, 5))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(reflection_rect, 6, 6)


# =============================================================================
#  SCROLLING TEXT BANNER
# =============================================================================
class ScrollingTextWidget(QWidget):
    """
    Animated scrolling text banner - perfect for status updates
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setMaximumHeight(48)

        self.text = "ALCF POLARIS SUPERCOMPUTER • READY FOR TRANSFER"
        self.offset = 0
        self.speed = 2

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(30)

    def set_text(self, text):
        self.text = text.upper()
        self.offset = 0

    def _animate(self):
        self.offset += self.speed
        if self.offset > 800:
            self.offset = -200
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0, QColor("#1a1a2e"))
        grad.setColorAt(0.5, QColor("#16213e"))
        grad.setColorAt(1, QColor("#1a1a2e"))
        painter.fillRect(self.rect(), grad)

        painter.setPen(QColor("#0abde3"))
        # Scale font with widget width (base width 340)
        base_w = 340
        scale = min(2.0, max(0.7, self.width() / base_w))
        pt = max(10, min(28, round(13 * scale)))
        painter.setFont(QFont("Courier New", pt, QFont.Bold))
        painter.drawText(int(self.offset), 30, self.text)


# =============================================================================
#  STATUS INDICATORS (PULSE)
# =============================================================================
class StatusIndicator(QWidget):
    """
    Animated status dot with label (like LED indicators)
    """
    def __init__(self, label="STATUS", parent=None):
        super().__init__(parent)
        self.setMinimumSize(140, 36)
        self.label = label
        self.color = QColor("#636e72")
        self.pulse_value = 0
        self.is_active = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._pulse)
        self.timer.start(50)

    def set_status(self, color, active=False):
        self.color = color
        self.is_active = active

    def _pulse(self):
        if self.is_active:
            self.pulse_value = (self.pulse_value + 0.1) % (2 * math.pi)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Scale font and layout with widget width (base width 140)
        base_w = 140
        scale = min(2.0, max(0.7, self.width() / base_w))
        pt = max(8, min(20, round(10 * scale)))
        painter.setPen(QColor("#bdc3c7"))
        painter.setFont(QFont("Arial", pt))
        text_x = int(28 * scale)
        text_y = int(22 * scale)
        painter.drawText(text_x, text_y, self.label)

        dot_size = max(8, min(18, int(12 * scale)))
        cx = 8 + (dot_size // 2)
        cy = int(11 * scale) + (dot_size // 2)
        if self.is_active:
            glow_size = dot_size + int(4 * math.sin(self.pulse_value))
            glow_color = QColor(self.color)
            glow_color.setAlpha(80)
            painter.setBrush(glow_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(cx - glow_size // 2, cy - glow_size // 2, glow_size, glow_size)

        painter.setBrush(self.color)
        painter.drawEllipse(cx - dot_size // 2, cy - dot_size // 2, dot_size, dot_size)


# =============================================================================
#  DATA FLOW WIDGET
# =============================================================================
class DataFlowWidget(QWidget):
    """
    Animated flowing particles/arrows showing data movement
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.particles = []
        self.is_transferring = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_particles)
        self.timer.start(40)

    def start_flow(self):
        self.is_transferring = True

    def stop_flow(self):
        self.is_transferring = False
        self.particles.clear()

    def _update_particles(self):
        if self.is_transferring and len(self.particles) < 8:
            self.particles.append({
                'x': 0, 'y': self.height() / 2,
                'speed': 3 + (len(self.particles) % 3)
            })

        for p in self.particles[:]:
            p['x'] += p['speed']
            if p['x'] > self.width():
                self.particles.remove(p)

        if self.particles or self.is_transferring:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor("#0f0f0f"))

        painter.setPen(QPen(QColor("#34495e"), 2, Qt.DashLine))
        painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)

        for p in self.particles:
            grad = QLinearGradient(p['x'] - 15, p['y'], p['x'], p['y'])
            grad.setColorAt(0, QColor(0, 168, 255, 0))
            grad.setColorAt(1, QColor(0, 168, 255, 255))

            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)

            painter.drawPolygon([
                QPointF(p['x'], p['y']),
                QPointF(p['x'] - 10, p['y'] - 5),
                QPointF(p['x'] - 10, p['y'] + 5)
            ])

        # Scale font with widget size (base 340x60)
        scale_w = min(2.0, max(0.7, self.width() / 340))
        scale_h = min(2.0, max(0.7, self.height() / 60))
        scale = min(scale_w, scale_h)
        pt = max(8, min(20, round(9 * scale)))
        painter.setPen(QColor("#7f8c8d"))
        painter.setFont(QFont("Arial", pt))
        label_y = int(15 * scale_h)
        painter.drawText(10, label_y, "BEAMLINE")
        painter.drawText(self.width() - int(60 * scale_w), label_y, "ALCF")


# =============================================================================
#  METRICS PANEL
# =============================================================================
class MetricsPanel(QWidget):
    """
    Display live metrics with smooth number transitions
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)

        self.metrics = {
            'loss': {'current': 0.0, 'target': 0.0, 'label': 'Training Loss'},
            'throughput': {'current': 0, 'target': 0, 'label': 'MB/s'},
            'model_size': {'current': 0, 'target': 0, 'label': 'Model Size (MB)'}
        }

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._smooth_update)
        self.timer.start(50)

    def set_metric(self, key, value):
        if key in self.metrics:
            self.metrics[key]['target'] = value

    def _smooth_update(self):
        changed = False
        for metric in self.metrics.values():
            diff = metric['target'] - metric['current']
            if abs(diff) > 0.01:
                metric['current'] += diff * 0.15
                changed = True

        if changed:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor("#1e272e"))

        # Scale fonts and layout with widget size (base 340x90)
        base_w, base_h = 340, 90
        scale_w = min(2.0, max(0.7, self.width() / base_w))
        scale_h = min(2.0, max(0.7, self.height() / base_h))
        scale = min(scale_w, scale_h)
        label_pt = max(8, min(18, round(9 * scale)))
        value_pt = max(12, min(28, round(16 * scale)))
        x_pos = int(10 * scale_w)
        y_label = int(18 * scale_h)
        y_value = int(48 * scale_h)
        x_step = int(110 * scale_w)

        for key, data in self.metrics.items():
            painter.setPen(QColor("#95a5a6"))
            painter.setFont(QFont("Arial", label_pt))
            painter.drawText(x_pos, y_label, data['label'])

            painter.setPen(QColor("#00d2d3"))
            painter.setFont(QFont("Consolas", value_pt, QFont.Bold))

            if key == 'loss':
                text = f"{data['current']:.4f}"
            else:
                text = f"{int(data['current'])}"

            painter.drawText(x_pos, y_value, text)

            x_pos += x_step


# =============================================================================
#  MAIN WINDOW
# =============================================================================
class AnalysisWindow(QMainWindow):
    """
    Main container window called by area_det_viewer.py.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Status")
        self.resize(340, 680)
        self.setStyleSheet("QMainWindow { background-color: #121212; } QLabel { color: #ecf0f1; }")

        # Setup Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # 1. Header (font scales on resize)
        self.lbl_title = QLabel("MODEL TRAINING NODE")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self._base_width = 340
        self.lbl_title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        layout.addWidget(self.lbl_title)

        # 2. Scrolling Status Banner
        self.status_banner = ScrollingTextWidget()
        layout.addWidget(self.status_banner)

        # 3. Rack Widget
        self.rack = ServerRackWidget()
        layout.addWidget(self.rack, stretch=1)

        # 4. Status Indicators Row
        indicators_layout = QHBoxLayout()
        self.ind_training = StatusIndicator("TRAINING")
        self.ind_transfer = StatusIndicator("TRANSFER")
        self.ind_alcf = StatusIndicator("ALCF LINK")
        indicators_layout.addWidget(self.ind_training)
        indicators_layout.addWidget(self.ind_transfer)
        indicators_layout.addWidget(self.ind_alcf)
        layout.addLayout(indicators_layout)

        # 5. Data Flow Visualization
        self.data_flow = DataFlowWidget()
        layout.addWidget(self.data_flow)

        # 6. Info Panel (larger text)
        self.info_frame = QFrame()
        self.info_frame.setStyleSheet("background-color: #2d3436; border-radius: 8px; padding: 12px;")
        info_layout = QVBoxLayout(self.info_frame)

        self.lbl_train_status = QLabel("Training: INITIALIZING")
        self.lbl_xfer_status = QLabel("Transfer: WAITING")
        self.lbl_train_status.setFont(QFont("Consolas", 12))
        self.lbl_xfer_status.setFont(QFont("Consolas", 12))

        info_layout.addWidget(self.lbl_train_status)
        info_layout.addWidget(self.lbl_xfer_status)
        layout.addWidget(self.info_frame)

        # Scale all label fonts to current window size
        self._update_label_fonts()

        # 7. Metrics Panel
        self.metrics = MetricsPanel()
        layout.addWidget(self.metrics)

        # Setup EPICS Logic
        self.epics_listener = EpicsListener()
        self.epics_listener.status_updated.connect(self.handle_pv_update)
        self.epics_listener.connection_changed.connect(self.handle_connection)
        
        # Start Listening
        self.epics_listener.start()
        
        # Simulation Mode Trigger (if no EPICS)
        if not EPICS_AVAILABLE:
            self.lbl_train_status.setText("MODE: SIMULATION")
            self.sim_timer = QTimer(self)
            self.sim_timer.timeout.connect(self.simulate_data)
            self.sim_timer.start(2000)
            self.sim_step = 0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_label_fonts()

    def _update_label_fonts(self):
        """Scale header and info-panel fonts with window width."""
        w = self.width()
        scale = min(2.0, max(0.7, w / self._base_width))
        title_pt = max(14, min(32, round(18 * scale)))
        info_pt = max(10, min(22, round(12 * scale)))
        self.lbl_title.setFont(QFont("Segoe UI", title_pt, QFont.Bold))
        self.lbl_train_status.setFont(QFont("Consolas", info_pt))
        self.lbl_xfer_status.setFont(QFont("Consolas", info_pt))

    def handle_connection(self, connected):
        self.rack.set_connection_status(connected)
        if not connected:
            self.lbl_train_status.setText("Status: CA DISCONNECTED")
            self.lbl_train_status.setStyleSheet("color: #ff7675") # Red text
            self.lbl_xfer_status.setText("Link: DOWN")
        else:
            self.lbl_train_status.setStyleSheet("color: #ecf0f1") # White text

    def handle_pv_update(self, pv_type, value):
        if pv_type == 'train':
            self.rack.set_training_status(value)
            text_map = {0: "NOT TUNED", 1: "POOR (Unusable)", 2: "FAIR (Usable)", 3: "EXCELLENT"}
            self.lbl_train_status.setText(f"Model: {text_map.get(value, 'UNKNOWN')}")

            color_map = {
                0: QColor("#636e72"),
                1: QColor("#d63031"),
                2: QColor("#fdcb6e"),
                3: QColor("#00b894")
            }
            self.ind_training.set_status(color_map.get(value, QColor("#636e72")), value == 3)

            loss_map = {0: 1.0, 1: 0.85, 2: 0.42, 3: 0.012}
            self.metrics.set_metric('loss', loss_map.get(value, 1.0))
            self.metrics.set_metric('model_size', 847)

        elif pv_type == 'xfer':
            self.rack.set_transfer_status(value)
            text_map = {0: "IDLE", 1: ">> UPLOADING TO ALCF >>", 2: "TRANSFER COMPLETE"}
            self.lbl_xfer_status.setText(f"ALCF: {text_map.get(value, 'UNKNOWN')}")

            if value == 1:
                self.lbl_xfer_status.setStyleSheet("color: #74b9ff; font-weight: bold;")
                self.ind_transfer.set_status(QColor("#0984e3"), True)
                self.ind_alcf.set_status(QColor("#0984e3"), True)
                self.data_flow.start_flow()
                self.status_banner.set_text("TRANSFERRING TO ALCF POLARIS • PHASE 1 → PHASE 2")
                self.metrics.set_metric('throughput', 1250)
            else:
                self.lbl_xfer_status.setStyleSheet("color: #ecf0f1;")
                self.ind_transfer.set_status(QColor("#636e72"), False)
                self.ind_alcf.set_status(
                    QColor("#00b894") if value == 2 else QColor("#636e72"), False
                )
                self.data_flow.stop_flow()
                self.status_banner.set_text("ALCF POLARIS SUPERCOMPUTER • READY FOR TRANSFER")
                self.metrics.set_metric('throughput', 0)

    def simulate_data(self):
        # Only runs if EPICS is missing
        self.sim_step += 1
        
        # Simulate Training States (0->1->2->3)
        train_val = (self.sim_step // 2) % 4
        self.handle_pv_update('train', train_val)
        
        # Simulate Transfer periodically
        xfer_val = 1 if (self.sim_step % 6 == 0) else 0
        self.handle_pv_update('xfer', xfer_val)

# =============================================================================
#  STANDALONE EXECUTION
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalysisWindow()
    window.show()
    sys.exit(app.exec_())