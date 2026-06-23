"""
bayesian_viewer.py
==================
PyQt5 DashPVA GUI for Bayesian optimization driven by Bluesky's **blop**
(``blop.ax.Agent`` — an Ax/BoTorch Bayesian optimizer).

This viewer is a thin UI shell over :mod:`dashpva.viewer.bayesian.blop_adapter`:

* The control panel is **scalable** — add as many degrees of freedom (motors)
  as the experiment needs (tested to a dozen+), each with **GUI-editable search
  limits**, plus one or more objectives read from the detector.
* A background :class:`ScanWorker` owns the Bluesky ``RunEngine`` and runs
  :func:`~dashpva.viewer.bayesian.blop_adapter.blop_optimize_plan`, which asks
  the agent for points, moves the motors, reads the detector, and ingests the
  result.  After every measured point it emits ``point_measured`` so the GUI can
  update live (Qt signals are the only cross-thread channel).
* Because the search space can be high-dimensional, the live plots are a
  **convergence trace** (objective vs. evaluation, with best-so-far) and a
  **2-D projection** scatter where the user picks which DOF pair to view.

Configuration is persisted between launches via ``QSettings`` (the same store the
area-detector viewer uses), so the experiment setup is remembered.

Usage
-----
    python -m dashpva.viewer.bayesian.bayesian_viewer
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from dashpva.gui import configure_app
from dashpva.gui.theme_colors import (
    ERROR,
    INFO,
    SUCCESS,
    TEXT_MUTED,
    WARNING,
    status_style,
)
from dashpva.viewer.bayesian.blop_adapter import (
    DOFSpec,
    ObjectiveSpec,
    OptimizerConfig,
)

logger = logging.getLogger(__name__)

_SETTINGS_KEY = "bayesian/optimizer_config"
# The Bluesky conda-env path is persisted on its own (it is a per-machine setup
# detail, not part of the portable optimizer config blob).
_SETTINGS_BLUESKY_ENV = "bayesian/bluesky_env"


def _action_button_style(bg: str) -> str:
    """Colored primary-action button matching the app's colored buttons, with a
    native-looking disabled state (so disabled Start/Stop don't look active)."""
    return (
        f"QPushButton {{ background-color: {bg}; color: white; font-weight: 600; "
        f"border: none; border-radius: 4px; padding: 7px 16px; }}"
        f"QPushButton:disabled {{ background-color: palette(button); "
        f"color: palette(mid); }}"
    )


# ---------------------------------------------------------------------------
# Scan worker thread
# ---------------------------------------------------------------------------

class ScanWorker(QThread):
    """Owns the Bluesky ``RunEngine`` and runs the blop optimization plan.

    Signals
    -------
    point_measured(object)
        Emitted after every measured point with the payload dict produced by
        :func:`blop_optimize_plan`'s ``on_point`` callback.
    scan_error(str)
        Emitted if an exception propagates out of the RunEngine.
    scan_finished()
        Emitted when the plan completes (normally or via abort).
    """

    point_measured = pyqtSignal(object)
    scan_error = pyqtSignal(str)
    scan_finished = pyqtSignal()
    agent_ready = pyqtSignal(object)  # the built blop agent (for model surfaces)

    def __init__(
        self,
        config: OptimizerConfig,
        bluesky_root: Optional[str],
        simulate: bool,
        agent=None,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.bluesky_root = bluesky_root
        self.simulate = simulate
        self._existing_agent = agent  # reuse to resume; None builds a fresh agent
        self._abort = False
        self._re = None

    def request_abort(self) -> None:
        """Thread-safe abort request; stops the RunEngine via ``RE.abort()``."""
        self._abort = True
        if self._re is not None:
            try:
                self._re.abort(reason="User requested abort from GUI")
            except Exception:  # noqa: BLE001
                pass

    def _ensure_bluesky(self) -> None:
        """Make ``bluesky`` importable.

        Prefer whatever is already on ``sys.path`` (e.g. the uv venv during
        offline development); only fall back to injecting the beamline conda
        env when bluesky is not already importable.
        """
        try:
            import bluesky  # noqa: F401
        except ImportError:
            from dashpva.viewer.bayesian.bluesky_compat import ensure_bluesky

            ensure_bluesky(root=self.bluesky_root)

    def run(self) -> None:
        try:
            self._ensure_bluesky()
            from bluesky import RunEngine
        except (ImportError, RuntimeError) as exc:
            self.scan_error.emit(
                f"bluesky import failed: {exc}\n"
                "Check the Bluesky Conda Env path in the viewer."
            )
            return

        try:
            from dashpva.viewer.bayesian.blop_adapter import (
                blop_optimize_plan,
                build_agent,
                resolve_devices,
            )

            actuators, readables, simulated = resolve_devices(
                self.config, simulate=self.simulate
            )
            # Reuse the existing agent to resume (keeps the Ax trial history);
            # otherwise build a fresh one.
            agent = self._existing_agent or build_agent(self.config, actuators)
            # Expose the agent so the GUI can keep it for resume + model surfaces
            # (used only after the scan; never while this worker mutates it).
            self.agent_ready.emit(agent)
        except Exception as exc:  # noqa: BLE001
            logger.error("blop setup failed:\n%s", traceback.format_exc())
            self.scan_error.emit(f"Failed to set up optimizer/devices:\n{exc}")
            return

        def _on_point(payload: dict) -> None:
            self.point_measured.emit(payload)

        # context_managers=[] disables RunEngine's default SIGINT handler, which
        # can only be installed on the main thread; we run the RE in this worker
        # thread and drive aborts via request_abort() -> RE.abort() instead.
        self._re = RunEngine(context_managers=[])
        try:
            self._re(
                blop_optimize_plan(
                    agent=agent,
                    actuators=actuators,
                    readables=readables,
                    config=self.config,
                    on_point=_on_point,
                )
            )
        except Exception as exc:  # noqa: BLE001
            # A user-requested abort surfaces as a RunEngine exception; that's a
            # clean stop, not an error — don't raise an error dialog for it.
            if self._abort:
                logger.info("Scan aborted by user.")
            else:
                logger.error("Scan failed:\n%s", traceback.format_exc())
                self.scan_error.emit(str(exc))
        finally:
            self.scan_finished.emit()


class _SurfaceWorker(QThread):
    """Computes a model-prediction surface off the GUI thread (GP compute)."""

    done = pyqtSignal(object)   # surface payload dict
    failed = pyqtSignal(str)

    def __init__(self, agent, config, x_name, y_name, parent=None):
        super().__init__(parent)
        self._agent = agent
        self._config = config
        self._x = x_name
        self._y = y_name

    def run(self) -> None:
        try:
            from dashpva.viewer.bayesian.blop_adapter import predict_surface

            payload = predict_surface(self._agent, self._config, self._x, self._y)
            self.done.emit(payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("Surface prediction failed:\n%s", traceback.format_exc())
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Live plot panel (pyqtgraph)
# ---------------------------------------------------------------------------

_MODE_POINTS = "Measured points"
_MODE_MEAN = "Predicted surface"
_MODE_SIGMA = "Uncertainty (σ)"
_MODE_ACQ = "Acquisition (UCB)"
_SURFACE_MODES = (_MODE_MEAN, _MODE_SIGMA, _MODE_ACQ)
_MODE_TO_GRID = {_MODE_MEAN: "mean", _MODE_SIGMA: "sem", _MODE_ACQ: "acq"}


class _PlotPanel(QtWidgets.QWidget):
    """Convergence trace + a 2-D projection of the search space for a DOF pair.

    The projection shows either the measured points (colored by objective) or a
    model **surface** — predicted objective, uncertainty (σ), or an acquisition
    (UCB) proxy — computed on demand via the "Update surface" button.
    """

    # Emitted when the user requests a model surface for (x_dof, y_dof).
    surface_requested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Light plot theme (white bg / black foreground), matching the rest of
        # DashPVA's analysis plots (e.g. the phase fitter) rather than a dark theme.
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        # Resizable split so the convergence plot isn't squished by the
        # projection panel (drag the divider to rebalance).
        splitter = QtWidgets.QSplitter(Qt.Vertical)
        lay.addWidget(splitter, 1)

        # ---- convergence plot -------------------------------------------
        self._conv = pg.PlotWidget(title="Objective vs. evaluation")
        self._conv.showGrid(x=True, y=True, alpha=0.2)
        self._conv.setLabel("bottom", "Evaluation #")
        self._conv.setLabel("left", "Objective")
        self._conv.setMinimumHeight(170)
        self._measured_curve = self._conv.plot(
            [], [], pen=None, symbol="o", symbolSize=6,
            symbolBrush=INFO, name="measured",
        )
        self._best_curve = self._conv.plot(
            [], [], pen=pg.mkPen(SUCCESS, width=2), name="best so far",
        )
        splitter.addWidget(self._conv)

        # ---- 2-D projection ---------------------------------------------
        proj_box = QtWidgets.QGroupBox("2-D projection")
        proj_lay = QtWidgets.QVBoxLayout(proj_box)

        # View mode + on-demand surface update.
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("View:"))
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItems([_MODE_POINTS, _MODE_MEAN, _MODE_SIGMA, _MODE_ACQ])
        self._mode_combo.setToolTip(
            "Measured points: where you sampled (color = objective value).\n"
            "Predicted surface: the model's learned objective landscape.\n"
            "Uncertainty (σ): where the model is unsure.\n"
            "Acquisition (UCB): optimistic estimate the optimizer is drawn toward."
        )
        row1.addWidget(self._mode_combo)
        self._update_btn = QtWidgets.QPushButton("Update surface")
        self._update_btn.setToolTip(
            "Compute the model surface for the selected DOF pair from the current "
            "model (run an optimization first; other DOFs are fixed at the best point)."
        )
        self._update_btn.setEnabled(False)  # enabled once a model exists (after a run)
        row1.addWidget(self._update_btn)
        row1.addStretch(1)
        proj_lay.addLayout(row1)

        # Axis selectors — populated with the user's DOF names.
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("X axis:"))
        self._x_combo = QtWidgets.QComboBox()
        row2.addWidget(self._x_combo)
        row2.addWidget(QtWidgets.QLabel("Y axis:"))
        self._y_combo = QtWidgets.QComboBox()
        row2.addWidget(self._y_combo)
        row2.addStretch(1)
        proj_lay.addLayout(row2)

        self._proj = pg.PlotWidget()
        self._proj.showGrid(x=True, y=True, alpha=0.2)
        self._cmap = pg.colormap.get("viridis")

        # Surface heatmap (behind the points) + colorbar.
        self._surface_img = pg.ImageItem()
        self._surface_img.setOpts(axisOrder="row-major")
        self._surface_img.setZValue(-10)
        self._surface_img.setVisible(False)
        self._proj.addItem(self._surface_img)
        try:
            self._colorbar = pg.ColorBarItem(colorMap=self._cmap)
            self._colorbar.setImageItem(self._surface_img,
                                        insert_in=self._proj.getPlotItem())
            self._colorbar.setVisible(False)
        except Exception:  # noqa: BLE001 - colorbar is a nicety, not essential
            self._colorbar = None

        self._scatter = pg.ScatterPlotItem(size=11, pen=pg.mkPen(None))
        self._proj.addItem(self._scatter)
        self._best_marker = pg.ScatterPlotItem(
            size=20, symbol="star",
            pen=pg.mkPen(WARNING, width=2), brush=pg.mkBrush(None),
        )
        self._proj.addItem(self._best_marker)
        proj_lay.addWidget(self._proj, 1)

        self._proj_hint = QtWidgets.QLabel("")
        self._proj_hint.setStyleSheet(status_style(TEXT_MUTED))
        self._proj_hint.setWordWrap(True)
        proj_lay.addWidget(self._proj_hint)
        splitter.addWidget(proj_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 380])

        self._x_combo.currentIndexChanged.connect(self._render_projection)
        self._y_combo.currentIndexChanged.connect(self._render_projection)
        self._mode_combo.currentIndexChanged.connect(self._render_projection)
        self._update_btn.clicked.connect(self._on_update_clicked)

        # data state
        self._dof_names: List[str] = []
        self._records: List[dict] = []   # each: {"params": {...}, "primary": float}
        self._best_idx: Optional[int] = None
        self._minimize = False
        self._surface: Optional[dict] = None        # last computed surface payload
        self._surface_key: Optional[tuple] = None   # (x_name, y_name) it was for

    # -- setup ------------------------------------------------------------
    def reset(self, dof_names: List[str], minimize: bool) -> None:
        self._dof_names = list(dof_names)
        self._records = []
        self._best_idx = None
        self._minimize = minimize
        self._surface = None
        self._surface_key = None
        self._measured_curve.setData([], [])
        self._best_curve.setData([], [])
        self._scatter.clear()
        self._best_marker.clear()
        self._surface_img.clear()
        self._surface_img.setVisible(False)
        if self._colorbar is not None:
            self._colorbar.setVisible(False)
        for combo in (self._x_combo, self._y_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._dof_names)
            combo.blockSignals(False)
        if len(self._dof_names) > 1:
            self._y_combo.setCurrentIndex(1)
        self._render_projection()

    def current_dof_pair(self) -> tuple:
        return self._x_combo.currentText(), self._y_combo.currentText()

    def set_update_enabled(self, enabled: bool) -> None:
        self._update_btn.setEnabled(enabled)

    # -- live update ------------------------------------------------------
    def add_point(self, payload: dict) -> None:
        self._records.append(payload)
        primaries = [r["primary"] for r in self._records]

        # best-so-far trace
        if self._minimize:
            best_run = np.minimum.accumulate(primaries)
            self._best_idx = int(np.argmin(primaries))
        else:
            best_run = np.maximum.accumulate(primaries)
            self._best_idx = int(np.argmax(primaries))
        xs = list(range(1, len(primaries) + 1))
        self._measured_curve.setData(xs, primaries)
        self._best_curve.setData(xs, list(best_run))

        self._render_projection()

    def best_record(self) -> Optional[dict]:
        if self._best_idx is None:
            return None
        return self._records[self._best_idx]

    def set_surface(self, payload: dict) -> None:
        """Store a computed model surface and display it (if axes/mode match)."""
        self._surface = payload
        self._surface_key = (payload["x_name"], payload["y_name"])
        self._render_projection()

    # -- projection -------------------------------------------------------
    def _on_update_clicked(self) -> None:
        xname, yname = self.current_dof_pair()
        if not xname or not yname or xname == yname:
            self._proj_hint.setText("Pick two different DOFs for X and Y first.")
            return
        self.surface_requested.emit(xname, yname)

    def _surface_matches_axes(self) -> bool:
        return self._surface is not None and self._surface_key == self.current_dof_pair()

    def _render_projection(self) -> None:
        xname, yname = self.current_dof_pair()
        self._proj.setLabel("bottom", xname or "")
        self._proj.setLabel("left", yname or "")
        mode = self._mode_combo.currentText()
        surface_mode = mode in _SURFACE_MODES

        self._draw_points_overlay(xname, yname, surface_mode=surface_mode)

        show_surface = surface_mode and self._surface_matches_axes()
        if show_surface:
            grid = np.asarray(self._surface[_MODE_TO_GRID[mode]], dtype=float)
            s = self._surface
            self._surface_img.setImage(grid)
            self._surface_img.setRect(QtCore.QRectF(
                s["x_lo"], s["y_lo"], s["x_hi"] - s["x_lo"], s["y_hi"] - s["y_lo"]))
            zmin, zmax = float(np.nanmin(grid)), float(np.nanmax(grid))
            if zmax <= zmin:
                zmax = zmin + 1e-9
            self._surface_img.setVisible(True)
            if self._colorbar is not None:
                self._colorbar.setLevels((zmin, zmax))
                self._colorbar.setVisible(True)
        else:
            self._surface_img.setVisible(False)
            if self._colorbar is not None:
                self._colorbar.setVisible(False)

        self._update_proj_hint()

    def _draw_points_overlay(self, xname, yname, *, surface_mode: bool) -> None:
        if not self._records or not xname or not yname:
            self._scatter.clear()
            self._best_marker.clear()
            return
        xs = np.array([r["params"].get(xname, np.nan) for r in self._records])
        ys = np.array([r["params"].get(yname, np.nan) for r in self._records])
        if surface_mode:
            # On a colored surface, draw plain outlined points so they stay visible.
            self._scatter.setData(
                x=xs, y=ys, brush=pg.mkBrush(255, 255, 255, 150),
                pen=pg.mkPen("k", width=0.5),
            )
        else:
            vals = np.array([r["primary"] for r in self._records], dtype=float)
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
            brushes = [pg.mkBrush(*self._cmap.map(float(n), mode="byte")) for n in norm]
            self._scatter.setData(x=xs, y=ys, brush=brushes, pen=pg.mkPen(None))
        if self._best_idx is not None:
            self._best_marker.setData(x=[xs[self._best_idx]], y=[ys[self._best_idx]])

    def _update_proj_hint(self) -> None:
        mode = self._mode_combo.currentText()
        if mode not in _SURFACE_MODES:
            self._proj_hint.setText("Color = objective value at each sampled point.")
        elif not self._surface_matches_axes():
            self._proj_hint.setText(
                "Click “Update surface” to compute the model surface for this DOF pair.")
        else:
            s = self._surface
            extra = ""
            if s.get("fixed"):
                extra = "; other DOFs fixed at best (" + ", ".join(
                    f"{k}={v:.4g}" for k, v in s["fixed"].items()) + ")"
            self._proj_hint.setText(f"{mode} of “{s['objective']}”{extra}.")


# ---------------------------------------------------------------------------
# DOF table
# ---------------------------------------------------------------------------

class _DOFTable(QtWidgets.QTableWidget):
    """Editable table of degrees of freedom (one motor per row)."""

    COLS = ["On", "Name", "Motor PV / Name", "Low limit", "High limit", "Type"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        hdr = self.horizontalHeader()
        hdr.setMinimumSectionSize(50)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)    # On
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)  # Name
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)  # Motor PV
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)    # Low limit
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.Fixed)    # High limit
        hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.Fixed)    # Type
        self.setColumnWidth(0, 44)
        self.setColumnWidth(3, 98)
        self.setColumnWidth(4, 98)
        self.setColumnWidth(5, 80)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setMinimumHeight(150)

    def add_dof(self, spec: Optional[DOFSpec] = None) -> None:
        spec = spec or DOFSpec(
            name=f"dof{self.rowCount() + 1}", pv="", lo=0.0, hi=1.0
        )
        r = self.rowCount()
        self.insertRow(r)

        # enable checkbox (centered)
        chk = QtWidgets.QCheckBox()
        chk.setChecked(spec.enabled)
        wrap = QtWidgets.QWidget()
        wl = QtWidgets.QHBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setAlignment(Qt.AlignCenter)
        wl.addWidget(chk)
        self.setCellWidget(r, 0, wrap)

        self.setItem(r, 1, QtWidgets.QTableWidgetItem(spec.name))
        self.setItem(r, 2, QtWidgets.QTableWidgetItem(spec.pv))

        lo = self._make_spin(spec.lo)
        hi = self._make_spin(spec.hi)
        self.setCellWidget(r, 3, lo)
        self.setCellWidget(r, 4, hi)

        kind = QtWidgets.QComboBox()
        kind.addItems(["float", "int"])
        kind.setCurrentText(spec.kind)
        self.setCellWidget(r, 5, kind)

    @staticmethod
    def _make_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        sb = QtWidgets.QDoubleSpinBox()
        sb.setDecimals(4)
        sb.setRange(-1e9, 1e9)
        sb.setValue(value)
        sb.setSingleStep(0.1)
        sb.setMinimumWidth(90)
        return sb

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True)
        if not rows and self.rowCount():
            rows = [self.rowCount() - 1]
        for r in rows:
            self.removeRow(r)

    def specs(self) -> List[DOFSpec]:
        out: List[DOFSpec] = []
        for r in range(self.rowCount()):
            chk = self.cellWidget(r, 0).findChild(QtWidgets.QCheckBox)
            name_item = self.item(r, 1)
            pv_item = self.item(r, 2)
            out.append(
                DOFSpec(
                    name=(name_item.text() if name_item else "").strip(),
                    pv=(pv_item.text() if pv_item else "").strip(),
                    lo=self.cellWidget(r, 3).value(),
                    hi=self.cellWidget(r, 4).value(),
                    kind=self.cellWidget(r, 5).currentText(),
                    enabled=chk.isChecked(),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Objective table
# ---------------------------------------------------------------------------

class _ObjectiveTable(QtWidgets.QTableWidget):
    """Editable table of objectives (usually one)."""

    COLS = ["Name", "Read PV", "Direction"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setMaximumHeight(120)

    def add_objective(self, spec: Optional[ObjectiveSpec] = None) -> None:
        # When the user clicks "Add objective" (no spec), give the new row a
        # unique default name so it doesn't collide with an existing objective
        # (objective names must be distinct; duplicates fail validation).
        if spec is None:
            spec = ObjectiveSpec(name=self._unique_default_name())
        r = self.rowCount()
        self.insertRow(r)
        self.setItem(r, 0, QtWidgets.QTableWidgetItem(spec.name))
        self.setItem(r, 1, QtWidgets.QTableWidgetItem(spec.pv))
        direction = QtWidgets.QComboBox()
        direction.addItems(["maximize", "minimize"])
        direction.setCurrentText("minimize" if spec.minimize else "maximize")
        self.setCellWidget(r, 2, direction)

    def _unique_default_name(self) -> str:
        existing = {
            (self.item(r, 0).text().strip() if self.item(r, 0) else "")
            for r in range(self.rowCount())
        }
        if "intensity" not in existing:
            return "intensity"
        i = 2
        while f"objective{i}" in existing:
            i += 1
        return f"objective{i}"

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True)
        if not rows and self.rowCount() > 1:
            rows = [self.rowCount() - 1]
        for r in rows:
            if self.rowCount() > 1:  # always keep at least one objective
                self.removeRow(r)

    def specs(self) -> List[ObjectiveSpec]:
        out: List[ObjectiveSpec] = []
        for r in range(self.rowCount()):
            name_item = self.item(r, 0)
            pv_item = self.item(r, 1)
            out.append(
                ObjectiveSpec(
                    name=(name_item.text() if name_item else "").strip(),
                    pv=(pv_item.text() if pv_item else "").strip(),
                    minimize=self.cellWidget(r, 2).currentText() == "minimize",
                )
            )
        return out


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BayesianViewer(QtWidgets.QMainWindow):
    """Main application window for blop-driven Bayesian optimization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bayesian Optimization  (blop)  –  DashPVA")
        self.resize(1320, 800)
        # No per-window stylesheet: inherit the global theme (theme.qss applied by
        # configure_app) so this viewer matches the rest of DashPVA.

        self._worker: Optional[ScanWorker] = None
        self._settings = QtCore.QSettings("DashPVA", "Viewer")
        # Model + config kept after a scan so model surfaces can be computed on
        # demand (the agent persists once the scan worker has finished).
        self._agent = None
        self._agent_config: Optional[OptimizerConfig] = None
        self._surface_worker: Optional[_SurfaceWorker] = None
        self._stopping = False  # True between a Stop request and the worker ending

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(self._build_control_panel(), 0)
        root.addWidget(self._build_display_panel(), 1)
        self._plots.surface_requested.connect(self._on_surface_requested)

        self._load_config()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_control_panel(self) -> QtWidgets.QWidget:
        from dashpva.viewer.bayesian.bluesky_compat import get_bluesky_root

        # ---- scrollable setup area -------------------------------------
        content = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(content)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(8)

        # Bluesky env
        env_row = QtWidgets.QFormLayout()
        env_row.setLabelAlignment(Qt.AlignRight)
        self._bluesky_env = QtWidgets.QLineEdit(get_bluesky_root())
        self._bluesky_env.setPlaceholderText("path to conda env with bluesky/ophyd")
        env_row.addRow("Bluesky Conda Env:", self._bluesky_env)
        outer.addLayout(env_row)

        # DOF table + buttons
        outer.addWidget(self._section_label("Degrees of freedom (motors)"))
        self._dof_table = _DOFTable()
        outer.addWidget(self._dof_table)
        dof_btns = QtWidgets.QHBoxLayout()
        add_dof = self._small_button("＋ Add DOF")
        rm_dof = self._small_button("－ Remove selected")
        add_dof.clicked.connect(lambda: self._dof_table.add_dof())
        rm_dof.clicked.connect(self._dof_table.remove_selected)
        dof_btns.addWidget(add_dof)
        dof_btns.addWidget(rm_dof)
        dof_btns.addStretch(1)
        outer.addLayout(dof_btns)

        # Objective table + buttons
        outer.addWidget(self._section_label("Objectives"))
        self._obj_table = _ObjectiveTable()
        outer.addWidget(self._obj_table)
        obj_btns = QtWidgets.QHBoxLayout()
        add_obj = self._small_button("＋ Add objective")
        rm_obj = self._small_button("－ Remove selected")
        add_obj.clicked.connect(lambda: self._obj_table.add_objective())
        rm_obj.clicked.connect(self._obj_table.remove_selected)
        obj_btns.addWidget(add_obj)
        obj_btns.addWidget(rm_obj)
        obj_btns.addStretch(1)
        outer.addLayout(obj_btns)

        # Run controls
        outer.addWidget(self._section_label("Run controls"))
        run_form = QtWidgets.QFormLayout()
        run_form.setLabelAlignment(Qt.AlignRight)
        self._iterations = QtWidgets.QSpinBox()
        self._iterations.setRange(1, 100000)
        self._iterations.setValue(30)
        self._iterations.setToolTip(
            "Number of optimization rounds. Each round: the model suggests "
            "point(s), the motors move there, the detector is read, and the model "
            "is updated with the result(s)."
        )
        self._n_points = QtWidgets.QSpinBox()
        self._n_points.setRange(1, 1000)
        self._n_points.setValue(1)
        self._n_points.setToolTip(
            "Batch size: how many points are suggested and measured each round "
            "BEFORE the model updates.\n"
            "• 1 = standard sequential optimization (measure one, learn, repeat) — "
            "most sample-efficient; recommended.\n"
            "• >1 = propose a batch of points from the current model at once "
            "(fewer model updates; useful for parallel/faster acquisition)."
        )
        self._total_lbl = QtWidgets.QLabel()
        self._total_lbl.setToolTip(
            "Total measurements ≈ iterations × points/iteration (an upper bound; "
            "the initial exploration rounds may return slightly fewer)."
        )
        self._iterations.valueChanged.connect(self._update_total)
        self._n_points.valueChanged.connect(self._update_total)
        # Captions kept as refs so they can switch to additive wording on resume.
        self._iter_caption = QtWidgets.QLabel("Iterations (rounds):")
        self._total_caption = QtWidgets.QLabel("Total evaluations:")
        run_form.addRow(self._iter_caption, self._iterations)
        run_form.addRow("Points / iteration (batch):", self._n_points)
        run_form.addRow(self._total_caption, self._total_lbl)
        self._acq_kwargs = QtWidgets.QLineEdit()
        self._acq_kwargs.setPlaceholderText("advanced: key=value, key2=value2")
        run_form.addRow("Acquisition options:", self._acq_kwargs)
        self._simulate = QtWidgets.QCheckBox("Simulate (offline, no EPICS)")
        run_form.addRow("", self._simulate)
        outer.addLayout(run_form)
        self._update_total()

        hint = QtWidgets.QLabel(
            "Ax auto-explores (Sobol) the first trials, then switches to the GP "
            "model. Limits above bound each motor's search range."
        )
        hint.setStyleSheet(status_style(TEXT_MUTED))
        hint.setWordWrap(True)
        outer.addWidget(hint)
        outer.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        # ---- fixed footer: Start/Stop/status always visible ------------
        footer = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(footer)
        fl.setContentsMargins(6, 8, 6, 6)
        fl.setSpacing(8)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color: palette(mid);")
        fl.addWidget(sep)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)
        self._start_btn = QtWidgets.QPushButton("▶  Start")
        self._start_btn.setStyleSheet(_action_button_style(SUCCESS))
        self._start_btn.setMinimumHeight(34)
        self._stop_btn = QtWidgets.QPushButton("■  Stop")
        self._stop_btn.setStyleSheet(_action_button_style(ERROR))
        self._stop_btn.setMinimumHeight(34)
        self._stop_btn.setEnabled(False)
        self._reset_btn = QtWidgets.QPushButton("↺  Reset")
        self._reset_btn.setMinimumHeight(34)
        self._reset_btn.setToolTip(
            "Stop any run and clear everything (plots, model, status) back to scratch.")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._reset_btn)
        fl.addLayout(btn_row)

        self._status_lbl = QtWidgets.QLabel("Status: Idle")
        self._status_lbl.setStyleSheet(status_style(TEXT_MUTED))
        fl.addWidget(self._status_lbl)
        self._best_lbl = QtWidgets.QLabel("")
        self._best_lbl.setWordWrap(True)
        fl.addWidget(self._best_lbl)

        # ---- assemble: scroll (stretch) + fixed footer -----------------
        container = QtWidgets.QWidget()
        container.setFixedWidth(610)
        col = QtWidgets.QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        col.addWidget(scroll, 1)
        col.addWidget(footer, 0)
        return container

    def _build_display_panel(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("Live optimization")
        lay = QtWidgets.QVBoxLayout(grp)
        self._plots = _PlotPanel()
        lay.addWidget(self._plots)
        return grp

    # -- small UI helpers -------------------------------------------------
    @staticmethod
    def _section_label(text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(f"color: {INFO}; font-weight: bold; margin-top: 6px;")
        return lbl

    @staticmethod
    def _small_button(text: str) -> QtWidgets.QPushButton:
        # Native button — inherits the global theme so it matches the rest of the app.
        return QtWidgets.QPushButton(text)

    def _update_total(self) -> None:
        total = self._iterations.value() * self._n_points.value()
        self._total_lbl.setText(f"{total}")

    # ------------------------------------------------------------------
    # Config <-> UI
    # ------------------------------------------------------------------

    def _gather_config(self) -> OptimizerConfig:
        return OptimizerConfig(
            dofs=self._dof_table.specs(),
            objectives=self._obj_table.specs(),
            iterations=self._iterations.value(),
            n_points=self._n_points.value(),
            acq_kwargs=_parse_kwargs(self._acq_kwargs.text()),
        )

    def _apply_config(self, cfg: OptimizerConfig) -> None:
        self._dof_table.setRowCount(0)
        for d in cfg.dofs:
            self._dof_table.add_dof(d)
        if self._dof_table.rowCount() == 0:
            self._dof_table.add_dof()
        self._obj_table.setRowCount(0)
        for o in cfg.objectives:
            self._obj_table.add_objective(o)
        if self._obj_table.rowCount() == 0:
            self._obj_table.add_objective()
        self._iterations.setValue(cfg.iterations)
        self._n_points.setValue(cfg.n_points)
        self._acq_kwargs.setText(_format_kwargs(cfg.acq_kwargs))
        self._update_total()

    def _load_config(self) -> None:
        raw = self._settings.value(_SETTINGS_KEY, "")
        cfg = None
        if raw:
            try:
                cfg = OptimizerConfig.from_dict(json.loads(raw))
            except (ValueError, TypeError) as exc:
                logger.warning("Could not load saved config: %s", exc)
        if cfg is None or not cfg.dofs:
            cfg = OptimizerConfig(
                dofs=[
                    DOFSpec(name="x", pv="", lo=0.0, hi=5.0),
                    DOFSpec(name="y", pv="", lo=0.0, hi=5.0),
                ],
                objectives=[ObjectiveSpec()],
            )
        self._apply_config(cfg)
        # Restore the Bluesky conda-env path (its own key, separate from the
        # optimizer config); fall back to the get_bluesky_root() default set at
        # construction when nothing was saved.
        env = self._settings.value(_SETTINGS_BLUESKY_ENV, "", type=str)
        if env:
            self._bluesky_env.setText(env)
        # Simulate is intentionally never persisted: always start OFF so the viewer
        # never reopens in simulation mode at the beamline.
        self._simulate.setChecked(False)

    def _save_config(self) -> None:
        try:
            cfg = self._gather_config()
            self._settings.setValue(_SETTINGS_KEY, json.dumps(cfg.to_dict()))
            self._settings.setValue(
                _SETTINGS_BLUESKY_ENV, self._bluesky_env.text().strip())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save config: %s", exc)

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        simulate = self._simulate.isChecked()
        # If a previous run is still tearing down, finish stopping it first so we
        # never run two RunEngines at once.
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_abort()
            if not self._worker.wait(5000):
                QtWidgets.QMessageBox.warning(
                    self, "Still stopping",
                    "The previous run is still stopping — try again in a moment.")
                return

        # Resume the existing run if a model is loaded (Stop→Start, or add more
        # iterations after a finish). Reset clears the model to force a fresh run.
        resuming = self._agent is not None and self._agent_config is not None
        if resuming:
            # Keep the model + plot history; pick up the current Iterations /
            # Points-per-iteration so you can add more. DOFs/objectives stay as
            # the running model's (use Reset to change the problem).
            run_cfg = self._agent_config
            run_cfg.iterations = self._iterations.value()
            run_cfg.n_points = self._n_points.value()
            if run_cfg.iterations < 1 or run_cfg.n_points < 1:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid", "Iterations and points/iteration must be >= 1.")
                return
        else:
            cfg = self._gather_config()
            err = cfg.validate(require_devices=not simulate)
            if err:
                QtWidgets.QMessageBox.warning(self, "Invalid configuration", err)
                return
            run_cfg = cfg
            self._agent_config = cfg
            self._plots.reset(
                [d.name for d in cfg.active_dofs()],
                cfg.active_objectives()[0].minimize,
            )
            self._best_lbl.setText("")

        self._save_config()
        self._stopping = False
        self._plots.set_update_enabled(False)  # can't query the model mid-scan

        self._worker = ScanWorker(
            config=run_cfg,
            bluesky_root=self._bluesky_env.text().strip() or None,
            simulate=simulate,
            agent=self._agent if resuming else None,
        )
        self._worker.point_measured.connect(self._on_point)
        self._worker.scan_error.connect(self._on_scan_error)
        self._worker.scan_finished.connect(self._on_scan_finished)
        self._worker.agent_ready.connect(self._on_agent_ready)
        self._worker.start()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_lbl.setText("Status: Resuming…" if resuming else "Status: Optimizing…")
        self._status_lbl.setStyleSheet(status_style(WARNING))

    def _on_agent_ready(self, agent) -> None:
        self._agent = agent

    def _on_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._stopping = True
            self._stop_btn.setEnabled(False)  # avoid repeated abort clicks
            self._status_lbl.setText("Status: Stopping…")
            self._status_lbl.setStyleSheet(status_style(WARNING))
            self._worker.request_abort()

    def _on_reset(self) -> None:
        """Stop any run and clear everything (plots, model, surface) to scratch."""
        if self._worker is not None and self._worker.isRunning():
            self._stopping = True
            self._worker.request_abort()
            self._worker.wait(5000)
        self._worker = None
        self._stopping = False
        self._agent = None
        self._agent_config = None
        self._plots.reset([], minimize=False)   # clear plots, surface, and axes
        self._plots.set_update_enabled(False)
        self._best_lbl.setText("")
        self._status_lbl.setText("Status: Idle")
        self._status_lbl.setStyleSheet(status_style(TEXT_MUTED))
        self._reset_buttons()

    # ------------------------------------------------------------------
    # Worker slots
    # ------------------------------------------------------------------

    def _on_point(self, payload: dict) -> None:
        self._plots.add_point(payload)
        total = self._iterations.value() * self._n_points.value()
        self._status_lbl.setText(f"Status: Measuring… ({payload['index']}/{total})")
        best = self._plots.best_record()
        if best is not None:
            params = ", ".join(f"{k}={v:.4g}" for k, v in best["params"].items())
            self._best_lbl.setText(
                f"Best so far: {best['primary']:.5g}  @  {params}"
            )

    def _on_scan_error(self, msg: str) -> None:
        self._status_lbl.setText("Status: Error")
        self._status_lbl.setStyleSheet(status_style(ERROR))
        self._reset_buttons()
        # Surfaces can be computed if a model was built before the failure.
        self._plots.set_update_enabled(self._agent is not None)
        QtWidgets.QMessageBox.critical(
            self, "Scan Error",
            f"The optimization encountered an error:\n\n{msg}\n\n"
            "Check the terminal / log for the full traceback.",
        )

    def _on_scan_finished(self) -> None:
        if self._stopping:
            self._status_lbl.setText("Status: Stopped")
            self._status_lbl.setStyleSheet(status_style(TEXT_MUTED))
        else:
            self._status_lbl.setText("Status: Complete ✓")
            self._status_lbl.setStyleSheet(status_style(SUCCESS))
        self._stopping = False
        self._reset_buttons()
        # Model is settled and no longer being mutated -> allow surface compute.
        self._plots.set_update_enabled(self._agent is not None)

    # ------------------------------------------------------------------
    # Model-surface slots (2-D projection)
    # ------------------------------------------------------------------

    def _on_surface_requested(self, x_name: str, y_name: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Optimization running",
                "Wait for the current run to finish before computing a surface.")
            return
        if self._agent is None or self._agent_config is None:
            QtWidgets.QMessageBox.information(
                self, "No model yet",
                "Run an optimization first, then click “Update surface”.")
            return
        self._plots.set_update_enabled(False)
        self._status_lbl.setText("Status: Computing surface…")
        self._status_lbl.setStyleSheet(status_style(WARNING))
        self._surface_worker = _SurfaceWorker(
            self._agent, self._agent_config, x_name, y_name)
        self._surface_worker.done.connect(self._on_surface_done)
        self._surface_worker.failed.connect(self._on_surface_failed)
        self._surface_worker.start()

    def _on_surface_done(self, payload: dict) -> None:
        self._plots.set_surface(payload)
        self._plots.set_update_enabled(True)
        self._status_lbl.setText("Status: Surface updated ✓")
        self._status_lbl.setStyleSheet(status_style(SUCCESS))

    def _on_surface_failed(self, msg: str) -> None:
        self._plots.set_update_enabled(True)
        self._status_lbl.setText("Status: Surface failed")
        self._status_lbl.setStyleSheet(status_style(ERROR))
        QtWidgets.QMessageBox.warning(
            self, "Surface error",
            f"Could not compute the model surface:\n\n{msg}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_buttons(self) -> None:
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        # When a model is loaded, the next Start *resumes* and the iteration/total
        # counts are additions to the existing run — reflect that in the labels.
        resuming = self._agent is not None
        self._start_btn.setText("▶  Resume" if resuming else "▶  Start")
        self._iter_caption.setText(
            "+ Iterations (rounds):" if resuming else "Iterations (rounds):")
        self._total_caption.setText(
            "+ Total evaluations:" if resuming else "Total evaluations:")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_config()
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_abort()
            self._worker.wait(3000)
        event.accept()


# ---------------------------------------------------------------------------
# kwargs parsing helpers (acquisition options field)
# ---------------------------------------------------------------------------

def _parse_kwargs(text: str) -> dict:
    """Parse a ``key=value, key2=value2`` string into a dict (typed)."""
    out: dict = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        # try int -> float -> bool -> str
        try:
            out[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(value)
            continue
        except ValueError:
            pass
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            out[key] = value
    return out


def _format_kwargs(kwargs: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in kwargs.items())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    )
    app = QtWidgets.QApplication(sys.argv)
    configure_app(app)
    app.setApplicationName("Bayesian Optimization (blop)")
    window = BayesianViewer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
