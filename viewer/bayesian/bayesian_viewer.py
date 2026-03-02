"""
bayesian_viewer.py
==================
PyQt5 DashPVA-compatible GUI for the 2-D Bayesian Optimization scan at
APS 6-ID.

Architecture
------------
- **Main thread** : Qt event loop + all UI rendering
- **Scan thread** : A ``QThread`` subclass that owns the Bluesky
  ``RunEngine`` and calls ``RE(bayesian2d(...))`` in a blocking loop.
  Cross-thread communication uses only Qt signals (thread-safe).

The scan thread emits a ``scan_update`` signal after each ``tell()``
call; the main thread's slot refreshes the Matplotlib heatmap.

Panel layout::

    ┌────────────────────────── BayesianViewer ───────────────────────────┐
    │  ┌── Control Panel ─────────────────────┐  ┌── Live Plot ─────────┐ │
    │  │  X Motor PV     [___________]        │  │                      │ │
    │  │  Y Motor PV     [___________]        │  │   2-D GP Mean Map    │ │
    │  │  Detector PV    [___________]        │  │   (Matplotlib axes)  │ │
    │  │  Scalar Key     [___________]        │  │                      │ │
    │  │  x_lo / x_hi   [____] / [____]      │  │  • scanned X markers │ │
    │  │  y_lo / y_hi   [____] / [____]      │  │                      │ │
    │  │  Max Points     [____]              │  │                      │ │
    │  │  N Initial      [____]              │  └──────────────────────┘ │
    │  │                                     │                           │
    │  │  [ Start Scan ]  [ Stop / Abort ]   │                           │
    │  │  Status: Idle                       │                           │
    │  └─────────────────────────────────────┘                           │
    └─────────────────────────────────────────────────────────────────────┘

Usage
-----
    python bayesian_viewer.py

Dependencies
------------
    PyQt5, matplotlib, numpy, bluesky, ophyd, gpytorch, torch
    plus bayesian_engine and bluesky_plan from this package.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Optional

import numpy as np

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use("Qt5Agg")  # noqa: E402 – must be before pyplot import
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # noqa: F401 – kept for colormaps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stylesheet (dark, modern)
# ---------------------------------------------------------------------------
_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    top: -7px;
    color: #89b4fa;
}
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #585b70;
    border-radius: 4px;
    padding: 4px 6px;
    color: #cdd6f4;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #89b4fa;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 5px;
    padding: 7px 20px;
    font-weight: bold;
}
QPushButton:hover { background-color: #b4befe; }
QPushButton:pressed { background-color: #74c7ec; }
QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QPushButton#stop_btn {
    background-color: #f38ba8;
}
QPushButton#stop_btn:hover { background-color: #eba0ac; }
QPushButton#stop_btn:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QLabel#status_label {
    color: #a6e3a1;
    font-style: italic;
}
"""


# ---------------------------------------------------------------------------
# Scan worker thread
# ---------------------------------------------------------------------------

class ScanWorker(QThread):
    """
    Runs the Bluesky RunEngine in a background thread.

    Signals
    -------
    scan_update(int, np.ndarray, np.ndarray, list, list)
        Emitted after each ``optimizer.tell()`` call (i.e., after each
        measurement) with:
            n_pts    : int   – total measurements so far
            mean_grid: 2-D np.ndarray – GP predictive mean
            std_grid : 2-D np.ndarray – GP predictive std
            xs       : list[float]    – all physical X coords measured
            ys       : list[float]    – all physical Y coords measured
    scan_error(str)
        Emitted if an exception propagates out of the RunEngine.
    scan_finished()
        Emitted when the plan completes normally.
    """

    # Qt signals (must be class-level attributes)
    scan_update = pyqtSignal(int, object, object, list, list)
    scan_error = pyqtSignal(str)
    scan_finished = pyqtSignal()

    def __init__(
        self,
        optimizer,
        detector,
        x_motor,
        x_lo: float,
        x_hi: float,
        y_motor,
        y_lo: float,
        y_hi: float,
        maxpts: int,
        n_initial: int,
        scalar_key: Optional[str],
        parent=None,
    ):
        super().__init__(parent)

        self.optimizer = optimizer
        self.detector = detector
        self.x_motor = x_motor
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.y_motor = y_motor
        self.y_lo = y_lo
        self.y_hi = y_hi
        self.maxpts = maxpts
        self.n_initial = n_initial
        self.scalar_key = scalar_key
        self._abort = False

    def request_abort(self) -> None:
        """Thread-safe abort request.  The RunEngine is stopped via RE.abort()."""
        self._abort = True
        if hasattr(self, "_re") and self._re is not None:
            try:
                self._re.abort(reason="User requested abort from GUI")
            except Exception:
                pass

    def run(self) -> None:
        """Entry point executed in the background thread."""
        # ----------------------------------------------------------------
        # Lazy import of bluesky inside the thread so that the RE and its
        # asyncio event loop are created on the worker thread (not the GUI
        # thread), which avoids cross-thread event-loop conflicts.
        # ----------------------------------------------------------------
        try:
            from .bluesky_compat import ensure_bluesky
            ensure_bluesky()
            from bluesky import RunEngine
        except (ImportError, RuntimeError) as exc:
            self.scan_error.emit(
                f"bluesky import failed: {exc}\n"
                "Ensure the 6idb-bits conda env is available."
            )
            return

        try:
            from .bluesky_plan import bayesian2d, _extract_scalar  # noqa: F401
        except ImportError as exc:
            self.scan_error.emit(f"bluesky_plan import failed: {exc}")
            return

        # Wrap the optimizer's tell() to emit the update signal after each call
        original_tell = self.optimizer.tell

        def _patched_tell(x, y, value):
            original_tell(x, y, value)
            # Fetch the updated prediction grid
            try:
                mean_g, std_g, _, _ = self.optimizer.get_prediction_grid()
            except RuntimeError:
                mean_g = np.zeros((self.optimizer.grid_nx, self.optimizer.grid_ny))
                std_g = mean_g.copy()

            self.scan_update.emit(
                self.optimizer.n_observations,
                mean_g,
                std_g,
                list(self.optimizer.observed_x),
                list(self.optimizer.observed_y),
            )

        self.optimizer.tell = _patched_tell

        # Build RunEngine
        self._re = RunEngine()
        # Subscribe to documents for debugging (optional)
        self._re.subscribe(lambda name, doc: logger.debug("RE doc: %s", name))

        try:
            self._re(
                bayesian2d(
                    optimizer=self.optimizer,
                    detector=self.detector,
                    x_motor=self.x_motor,
                    x_lo=self.x_lo,
                    x_hi=self.x_hi,
                    y_motor=self.y_motor,
                    y_lo=self.y_lo,
                    y_hi=self.y_hi,
                    maxpts=self.maxpts,
                    n_initial=self.n_initial,
                    scalar_key=self.scalar_key or None,
                )
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Scan failed:\n%s", tb)
            self.scan_error.emit(str(exc))
        finally:
            # Restore original tell method
            self.optimizer.tell = original_tell
            self.scan_finished.emit()


# ---------------------------------------------------------------------------
# Matplotlib canvas (embedded in Qt)
# ---------------------------------------------------------------------------

class _MplCanvas(FigureCanvas):
    """
    A Matplotlib figure embedded in a Qt widget.

    Displays:
    - A 2-D colourmap of the GP predictive mean (``pcolor`` / ``imshow``).
    - Red 'X' scatter markers at every physically sampled point.
    """

    def __init__(self, parent=None, width: int = 6, height: int = 5):
        # Dark background to match the overall stylesheet
        self.fig = Figure(figsize=(width, height), facecolor="#1e1e2e")
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        # State
        self._im = None     # imshow AxesImage
        self._cbar = None   # colorbar
        self._scatter = None  # scanned points scatter

    def _style_axes(self) -> None:
        self.ax.set_facecolor("#181825")
        self.ax.tick_params(colors="#cdd6f4")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#45475a")
        self.ax.set_xlabel("X (motor units)", color="#cdd6f4")
        self.ax.set_ylabel("Y (motor units)", color="#cdd6f4")
        self.ax.set_title("GP Predictive Mean  (awaiting data…)", color="#89b4fa", pad=10)

    def update_plot(
        self,
        mean_grid: np.ndarray,
        xi: np.ndarray,
        yi: np.ndarray,
        xs: list,
        ys: list,
        n_pts: int,
    ) -> None:
        """
        Refresh the 2-D heatmap and the scatter of sampled points.

        Parameters
        ----------
        mean_grid : np.ndarray, shape (Nx, Ny)
            GP predictive mean values.
        xi : np.ndarray, shape (Nx,)
            Physical X axis coordinates.
        yi : np.ndarray, shape (Ny,)
            Physical Y axis coordinates.
        xs, ys : list of float
            Physical coordinates of all measurements so far.
        n_pts : int
            Total number of measurements (used in title).
        """
        extent = [xi[0], xi[-1], yi[0], yi[-1]]

        # ---- heatmap ---------------------------------------------------
        if self._im is None:
            # First call: create the image
            self._im = self.ax.imshow(
                mean_grid.T,          # transpose: imshow expects (rows=Y, cols=X)
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            # Colorbar
            self._cbar = self.fig.colorbar(
                self._im, ax=self.ax, fraction=0.046, pad=0.04
            )
            self._cbar.set_label(
                "Predicted Mean\n(0 = region 0, 1 = region 1, 0.5 = boundary)",
                color="#cdd6f4",
                fontsize=10,
            )
            self._cbar.ax.yaxis.set_tick_params(color="#cdd6f4")
            plt.setp(self._cbar.ax.yaxis.get_ticklabels(), color="#cdd6f4")
        else:
            # Subsequent calls: update array only (faster than re-creating)
            self._im.set_data(mean_grid.T)
            self._im.set_extent(extent)

        # ---- scatter of scanned points ---------------------------------
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None

        if len(xs) > 0:
            self._scatter = self.ax.scatter(
                xs, ys,
                marker="x",
                color="#f38ba8",    # Catppuccin red
                s=60,
                linewidths=2,
                zorder=5,
                label=f"Measured (n={n_pts})",
            )
            # Rebuild legend each time so count stays updated
            legend = self.ax.legend(
                loc="upper right",
                facecolor="#313244",
                edgecolor="#45475a",
                labelcolor="#cdd6f4",
            )

        self.ax.set_title(
            f"GP Predictive Mean  |  Measurements: {n_pts}",
            color="#89b4fa",
            pad=10,
        )
        self.ax.set_xlabel("X (motor units)", color="#cdd6f4")
        self.ax.set_ylabel("Y (motor units)", color="#cdd6f4")

        self.fig.tight_layout()
        self.draw()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BayesianViewer(QtWidgets.QMainWindow):
    """
    Main application window.

    Provides:
    - A control panel (left) for entering scan parameters.
    - A live 2-D GP prediction display (right).
    - Start / Stop buttons tied to the ScanWorker thread.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bayesian 2-D Scan Viewer  –  APS 6-ID")
        self.resize(1200, 700)
        self.setStyleSheet(_STYLE)

        # Worker thread (created fresh each scan)
        self._worker: Optional[ScanWorker] = None

        # Build UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setSpacing(12)
        root_layout.setContentsMargins(12, 12, 12, 12)

        root_layout.addWidget(self._build_control_panel(), 0)
        root_layout.addWidget(self._build_display_panel(), 1)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_control_panel(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("Scan Parameters")
        form = QtWidgets.QFormLayout(grp)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)

        def _line(placeholder=""):
            w = QtWidgets.QLineEdit()
            w.setPlaceholderText(placeholder)
            return w

        def _dspin(lo, hi, val, dec=3, step=0.1):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setValue(val)
            w.setDecimals(dec)
            w.setSingleStep(step)
            return w

        def _spin(lo, hi, val):
            w = QtWidgets.QSpinBox()
            w.setRange(lo, hi)
            w.setValue(val)
            return w

        # PV / device name fields
        self._x_pv = _line("e.g. 6IDA:m1  or  sample_x")
        self._y_pv = _line("e.g. 6IDA:m2  or  sample_y")
        self._det_pv = _line("e.g. 6IDD:det  or  my_detector")
        self._scalar_key = _line("e.g. stats1_total  (blank = auto)")

        # Bounds
        self._x_lo = _dspin(-100, 100, 0.0)
        self._x_hi = _dspin(-100, 100, 5.0)
        self._y_lo = _dspin(-100, 100, 0.0)
        self._y_hi = _dspin(-100, 100, 5.0)

        # Scan parameters
        self._maxpts = _spin(5, 10000, 100)
        self._n_init = _spin(1, 1000, 10)
        self._grid_nx = _spin(8, 256, 64)
        self._grid_ny = _spin(8, 256, 64)

        form.addRow("X Motor PV / Name:", self._x_pv)
        form.addRow("Y Motor PV / Name:", self._y_pv)
        form.addRow("Detector PV / Name:", self._det_pv)
        form.addRow("Scalar Signal Key:", self._scalar_key)

        sep_bounds = QtWidgets.QFrame()
        sep_bounds.setFrameShape(QtWidgets.QFrame.HLine)
        sep_bounds.setStyleSheet("color: #45475a;")
        form.addRow(sep_bounds)

        # X bounds in one row
        x_row = QtWidgets.QWidget()
        x_lay = QtWidgets.QHBoxLayout(x_row)
        x_lay.setContentsMargins(0, 0, 0, 0)
        x_lay.addWidget(self._x_lo)
        x_lay.addWidget(QtWidgets.QLabel("→"))
        x_lay.addWidget(self._x_hi)
        form.addRow("X Range:", x_row)

        y_row = QtWidgets.QWidget()
        y_lay = QtWidgets.QHBoxLayout(y_row)
        y_lay.setContentsMargins(0, 0, 0, 0)
        y_lay.addWidget(self._y_lo)
        y_lay.addWidget(QtWidgets.QLabel("→"))
        y_lay.addWidget(self._y_hi)
        form.addRow("Y Range:", y_row)

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setStyleSheet("color: #45475a;")
        form.addRow(sep2)

        form.addRow("Max Points:", self._maxpts)
        form.addRow("N Initial Random:", self._n_init)
        form.addRow("Grid Nx:", self._grid_nx)
        form.addRow("Grid Ny:", self._grid_ny)

        sep3 = QtWidgets.QFrame()
        sep3.setFrameShape(QtWidgets.QFrame.HLine)
        sep3.setStyleSheet("color: #45475a;")
        form.addRow(sep3)

        # Buttons
        btn_row = QtWidgets.QWidget()
        btn_lay = QtWidgets.QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)

        self._start_btn = QtWidgets.QPushButton("▶  Start Scan")
        self._stop_btn = QtWidgets.QPushButton("■  Stop / Abort")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.setEnabled(False)

        btn_lay.addWidget(self._start_btn)
        btn_lay.addWidget(self._stop_btn)
        form.addRow(btn_row)

        # Status label
        self._status_lbl = QtWidgets.QLabel("Status: Idle")
        self._status_lbl.setObjectName("status_label")
        form.addRow(self._status_lbl)

        # Connect buttons
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)

        return grp

    def _build_display_panel(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("Live GP Prediction Map")
        lay = QtWidgets.QVBoxLayout(grp)

        self._canvas = _MplCanvas(width=7, height=6)
        lay.addWidget(self._canvas)

        return grp

    # ------------------------------------------------------------------
    # Slot: Start scan
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        """
        Validate inputs, construct Ophyd objects, create BayesianOptimizer,
        and launch ScanWorker.
        """
        # --- Validate bounds ---
        if self._x_lo.value() >= self._x_hi.value():
            self._show_error("X range error", "x_lo must be less than x_hi.")
            return
        if self._y_lo.value() >= self._y_hi.value():
            self._show_error("Y range error", "y_lo must be less than y_hi.")
            return
        if self._n_init.value() >= self._maxpts.value():
            self._show_error(
                "Parameter error", "n_initial must be less than maxpts."
            )
            return

        # --- Resolve Ophyd objects ---
        x_motor, y_motor, detector = self._resolve_devices()
        if x_motor is None:
            return  # error already shown

        scalar_key = self._scalar_key.text().strip() or None

        # --- Build optimizer ---
        from .bayesian_engine import BayesianOptimizer

        optimizer = BayesianOptimizer(
            x_bounds=(self._x_lo.value(), self._x_hi.value()),
            y_bounds=(self._y_lo.value(), self._y_hi.value()),
            grid_nx=self._grid_nx.value(),
            grid_ny=self._grid_ny.value(),
        )
        # Cache xi/yi for plot axis ticks
        self._xi = optimizer._xi
        self._yi = optimizer._yi

        # --- Launch worker ---
        self._worker = ScanWorker(
            optimizer=optimizer,
            detector=detector,
            x_motor=x_motor,
            x_lo=self._x_lo.value(),
            x_hi=self._x_hi.value(),
            y_motor=y_motor,
            y_lo=self._y_lo.value(),
            y_hi=self._y_hi.value(),
            maxpts=self._maxpts.value(),
            n_initial=self._n_init.value(),
            scalar_key=scalar_key,
        )
        self._worker.scan_update.connect(self._on_scan_update)
        self._worker.scan_error.connect(self._on_scan_error)
        self._worker.scan_finished.connect(self._on_scan_finished)
        self._worker.start()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_lbl.setText("Status: Scanning…")
        self._status_lbl.setStyleSheet("color: #fab387;")   # orange while running

    # ------------------------------------------------------------------
    # Slot: Stop / Abort  
    # ------------------------------------------------------------------

    def _on_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._status_lbl.setText("Status: Aborting…")
            self._worker.request_abort()

    # ------------------------------------------------------------------
    # Slots: scan feedback
    # ------------------------------------------------------------------

    def _on_scan_update(
        self, n_pts: int, mean_grid, std_grid, xs: list, ys: list
    ) -> None:
        """Slot called (in the GUI thread) after each optimizer.tell()."""
        self._status_lbl.setText(f"Status: Measuring… ({n_pts}/{self._maxpts.value()})")
        self._canvas.update_plot(
            mean_grid=mean_grid,
            xi=self._xi,
            yi=self._yi,
            xs=xs,
            ys=ys,
            n_pts=n_pts,
        )

    def _on_scan_error(self, msg: str) -> None:
        """Slot called if the RunEngine raises an exception."""
        self._status_lbl.setText("Status: Error")
        self._status_lbl.setStyleSheet("color: #f38ba8;")
        self._reset_buttons()
        QtWidgets.QMessageBox.critical(
            self, "Scan Error",
            f"The scan encountered an error:\n\n{msg}\n\n"
            "Check the terminal / log for the full traceback.",
        )

    def _on_scan_finished(self) -> None:
        """Slot called when the plan completes normally or is aborted."""
        self._status_lbl.setText("Status: Complete ✓")
        self._status_lbl.setStyleSheet("color: #a6e3a1;")
        self._reset_buttons()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_buttons(self) -> None:
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _show_error(self, title: str, msg: str) -> None:
        QtWidgets.QMessageBox.warning(self, title, msg)

    def _resolve_devices(self):
        """
        Resolve the text-field entries to Ophyd device objects.

        Strategy:
        1. Try importing the name from the current IPython namespace via
           ``get_ipython().user_ns``.
        2. Fall back to creating an EpicsMotor / EpicsSignal from the PV
           string directly (requires EPICS connection).
        3. In **offline / demo** mode (no EPICS, no IPython), create
           ``ophyd.sim`` equivalents so the GUI can be tested.

        Returns (x_motor, y_motor, detector) or (None, None, None) on fail.
        """
        x_name = self._x_pv.text().strip()
        y_name = self._y_pv.text().strip()
        d_name = self._det_pv.text().strip()

        if not x_name or not y_name or not d_name:
            self._show_error(
                "Input error",
                "Please fill in all three device PV / name fields.",
            )
            return None, None, None

        # --- Try IPython namespace first (beamline session) ---
        try:
            ip = get_ipython()  # type: ignore[name-defined]
            ns = ip.user_ns
            x_motor = ns.get(x_name)
            y_motor = ns.get(y_name)
            detector = ns.get(d_name)
            if x_motor and y_motor and detector:
                logger.info("Resolved devices from IPython namespace.")
                return x_motor, y_motor, detector
        except NameError:
            pass  # not running inside IPython

        # --- Try direct EPICS construction ---
        try:
            from .bluesky_compat import ensure_bluesky
            ensure_bluesky()
            from ophyd import EpicsMotor, Device

            x_motor = EpicsMotor(x_name, name="x_motor")
            y_motor = EpicsMotor(y_name, name="y_motor")

            # Detector: if it looks like a PV, wrap with EpicsSignal
            # Real code would import the beamline-specific detector class.
            from ophyd import EpicsSignalRO

            class _SimpleDetector(Device):
                total = EpicsSignalRO(d_name, name="total", kind="hinted")

            detector = _SimpleDetector(name="detector")
            logger.info("Constructed Ophyd devices from PV strings.")
            return x_motor, y_motor, detector

        except Exception as exc:
            logger.warning("EPICS construction failed: %s – falling back to sim.", exc)

        # --- Offline / demo mode: ophyd.sim ---
        try:
            from .bluesky_compat import ensure_bluesky
            ensure_bluesky()
            from ophyd.sim import SynAxis, SynSignal

            x_motor = SynAxis(name=x_name or "sim_x")
            y_motor = SynAxis(name=y_name or "sim_y")

            # Use ophyd.sim.SynSignal which fully satisfies the Bluesky
            # protocol (trigger, read, describe, name, hints) so that
            # trigger_and_read works correctly.
            import random as _rng

            detector = SynSignal(
                func=lambda: _rng.uniform(0.0, 1.0),
                name="stats1_total",
                labels={"detectors"},
            )
            logger.warning(
                "Using SIMULATION devices – no real hardware connected."
            )
            QtWidgets.QMessageBox.information(
                self,
                "Simulation Mode",
                "Could not connect to EPICS or find devices in IPython namespace.\n"
                "Running in OFFLINE SIMULATION MODE.\n\n"
                "PV names entered are used as device names only.",
            )
            return x_motor, y_motor, detector

        except Exception as exc:
            self._show_error(
                "Device error",
                f"Could not resolve devices:\n{exc}\n\n"
                "Please check PV names and EPICS connection.",
            )
            return None, None, None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Ensure the worker thread is stopped before the window closes."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_abort()
            self._worker.wait(3000)  # 3-second grace period
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    )
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Bayesian 2-D Scan Viewer")
    window = BayesianViewer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
