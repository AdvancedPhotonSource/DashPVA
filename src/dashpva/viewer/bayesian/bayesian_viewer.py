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
from typing import Any, Dict, List, Optional

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
from dashpva.viewer.bayesian import profile_store
from dashpva.viewer.bayesian.blop_adapter import (
    DOFSpec,
    ObjectiveSpec,
    OptimizerConfig,
)

logger = logging.getLogger(__name__)

# Legacy single local config blob (pre-named-setups); migrated to a "default" local
# setup on first launch, then unused.
_SETTINGS_KEY = "bayesian/optimizer_config"
# Per-machine, non-exportable local state (never part of the portable profile).
_SETTINGS_BLUESKY_ENV = "bayesian/bluesky_env"
# Last-used named setup, keyed per profile/local: f"{_SETTINGS_LAST_SETUP}/{label}".
_SETTINGS_LAST_SETUP = "bayesian/last_setup"
# Named local setups (QSettings JSON table) used when no central profile is active.
_SETTINGS_LOCAL_SETUPS = "bayesian/local_setups"


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
    """Computes a single-objective model surface off the GUI thread (GP compute)."""

    done = pyqtSignal(object)   # surface payload dict
    failed = pyqtSignal(str)

    def __init__(self, agent, config, x_name, y_name, *,
                 fixed_overrides=None, objective=None, parent=None):
        super().__init__(parent)
        self._agent = agent
        self._config = config
        self._x = x_name
        self._y = y_name
        self._fixed = fixed_overrides
        self._objective = objective

    def run(self) -> None:
        try:
            from dashpva.viewer.bayesian.blop_adapter import predict_surface

            payload = predict_surface(
                self._agent, self._config, self._x, self._y,
                fixed_overrides=self._fixed, objective=self._objective)
            self.done.emit(payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("Surface prediction failed:\n%s", traceback.format_exc())
            self.failed.emit(str(exc))


class _SurfaceMultiWorker(QThread):
    """Computes all optimized objectives over a 2-D grid (for the RGB phase map)."""

    done = pyqtSignal(object)   # multi-surface payload dict
    failed = pyqtSignal(str)

    def __init__(self, agent, config, x_name, y_name, *,
                 fixed_overrides=None, parent=None):
        super().__init__(parent)
        self._agent = agent
        self._config = config
        self._x = x_name
        self._y = y_name
        self._fixed = fixed_overrides

    def run(self) -> None:
        try:
            from dashpva.viewer.bayesian.blop_adapter import predict_surface_multi

            payload = predict_surface_multi(
                self._agent, self._config, self._x, self._y,
                fixed_overrides=self._fixed)
            self.done.emit(payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("Phase-map prediction failed:\n%s", traceback.format_exc())
            self.failed.emit(str(exc))


class _SliceWorker(QThread):
    """Computes a 1-D model slice (all optimized objectives vs one DOF) off-thread."""

    done = pyqtSignal(object)   # slice payload dict
    failed = pyqtSignal(str)

    def __init__(self, agent, config, x_name, *, fixed_overrides=None, parent=None):
        super().__init__(parent)
        self._agent = agent
        self._config = config
        self._x = x_name
        self._fixed = fixed_overrides

    def run(self) -> None:
        try:
            from dashpva.viewer.bayesian.blop_adapter import predict_slice_1d

            payload = predict_slice_1d(
                self._agent, self._config, self._x, fixed_overrides=self._fixed)
            self.done.emit(payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("Slice prediction failed:\n%s", traceback.format_exc())
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Live plot panel (pyqtgraph)
# ---------------------------------------------------------------------------

_MODE_POINTS = "Measured points"
_MODE_MEAN = "Predicted surface"
_MODE_SIGMA = "Uncertainty (σ)"
_MODE_ACQ = "Acquisition (UCB)"
_MODE_OBJ_VS_DOF = "Objectives vs DOF"
_SURFACE_MODES = (_MODE_MEAN, _MODE_SIGMA, _MODE_ACQ)
_MODE_TO_GRID = {_MODE_MEAN: "mean", _MODE_SIGMA: "sem", _MODE_ACQ: "acq"}

# Distinct colors for the per-objective series in the "Objectives vs DOF" view.
_OBJ_PALETTE = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
)

_MODE_PHASE_MAP = "Phase map (RGB)"


def phase_anchor_rgb(i: int, n: int) -> tuple:
    """Anchor RGB (0-255) for phase ``i`` of ``n``.

    n<=3 -> clean red/green/blue primaries (so mixtures read intuitively);
    n>3  -> evenly-spaced samples of the 'turbo' colormap (approximate blend).
    """
    if n <= 3:
        return ((230, 40, 40), (40, 180, 70), (50, 90, 230))[i]
    try:
        cmap = pg.colormap.get("turbo")
        pos = i / max(1, n - 1)
        c = cmap.map(float(pos), mode="byte")
        return (int(c[0]), int(c[1]), int(c[2]))
    except Exception:  # noqa: BLE001 - fall back to palette hex
        col = pg.mkColor(_OBJ_PALETTE[i % len(_OBJ_PALETTE)])
        return (col.red(), col.green(), col.blue())


def composition_rgb(fracs, n: int):
    """Blend per-item phase fractions into uint8 RGB via the phase anchors.

    ``fracs`` is an ``(m, n)`` array (m items, n phases).  Each row is clamped to
    >=0 and normalized to sum 1, then mapped to a weighted sum of the n anchor
    colors.  Returns an ``(m, 3)`` uint8 array.  A pure phase yields its anchor
    color; mixtures blend proportionally.
    """
    f = np.clip(np.asarray(fracs, dtype=float), 0.0, None)
    if f.ndim == 1:
        f = f.reshape(1, -1)
    s = f.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    w = f / s                                              # (m, n)
    anchors = np.array([phase_anchor_rgb(i, n) for i in range(n)], dtype=float)  # (n, 3)
    rgb = w @ anchors                                      # (m, 3)
    return np.clip(rgb, 0, 255).astype(np.uint8)


class _PlotPanel(QtWidgets.QWidget):
    """Convergence trace + a 2-D projection of the search space for a DOF pair.

    The projection shows either the measured points (colored by objective) or a
    model **surface** — predicted objective, uncertainty (σ), or an acquisition
    (UCB) proxy — computed on demand via the "Update surface" button.
    """

    # Emitted when the user requests a single-objective model surface (x_dof, y_dof).
    surface_requested = pyqtSignal(str, str)
    # Emitted when the user requests an RGB phase-map (all objectives) (x_dof, y_dof).
    surface_multi_requested = pyqtSignal(str, str)
    # Emitted when the user requests a 1-D model slice (objectives vs x_dof).
    slice_requested = pyqtSignal(str)

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
        self._mode_combo.addItems(
            [_MODE_POINTS, _MODE_PHASE_MAP, _MODE_MEAN, _MODE_SIGMA, _MODE_ACQ,
             _MODE_OBJ_VS_DOF])
        self._mode_combo.setToolTip(
            "Measured points: where you sampled (color = phase composition).\n"
            "Phase map (RGB): predicted phase-composition map over the DOF pair.\n"
            "Predicted surface: the model's learned surface for one objective.\n"
            "Uncertainty (σ): where the model is unsure.\n"
            "Acquisition (UCB): optimistic estimate the optimizer is drawn toward.\n"
            "Objectives vs DOF: each objective (phase fraction) vs the X DOF — "
            "measured points, plus a model curve per optimized objective on Update."
        )
        row1.addWidget(self._mode_combo)
        # Objective selector — used by the single-objective surface modes.
        self._obj_label = QtWidgets.QLabel("Objective:")
        row1.addWidget(self._obj_label)
        self._obj_combo = QtWidgets.QComboBox()
        self._obj_combo.setToolTip(
            "Which objective the Predicted surface / σ / UCB views show "
            "(ignored by the composition and phase-map views).")
        row1.addWidget(self._obj_combo)
        self._update_btn = QtWidgets.QPushButton("Update surface")
        self._update_btn.setToolTip(
            "Compute the model surface for the selected DOF pair from the current "
            "model (run an optimization first; other DOFs are fixed via the sliders "
            "below, defaulting to the best point)."
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
        self._y_label = QtWidgets.QLabel("Y axis:")
        row2.addWidget(self._y_label)
        self._y_combo = QtWidgets.QComboBox()
        row2.addWidget(self._y_combo)
        row2.addStretch(1)
        proj_lay.addLayout(row2)

        # Fixed-DOF sliders (for DOFs not on X/Y): populated by _rebuild_fixed_sliders.
        self._fixed_box = QtWidgets.QWidget()
        self._fixed_form = QtWidgets.QFormLayout(self._fixed_box)
        self._fixed_form.setContentsMargins(0, 0, 0, 0)
        self._fixed_form.setSpacing(2)
        proj_lay.addWidget(self._fixed_box)

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
        # Legend (used by the "Objectives vs DOF" view; harmless otherwise).
        self._legend = self._proj.addLegend(offset=(-10, 10))
        # Per-objective series for the "Objectives vs DOF" view (created lazily).
        self._obj_scatter: Dict[str, Any] = {}   # name -> ScatterPlotItem (measured)
        self._obj_curve: Dict[str, Any] = {}     # name -> PlotDataItem (model mean)
        proj_lay.addWidget(self._proj, 1)

        self._proj_hint = QtWidgets.QLabel("")
        self._proj_hint.setStyleSheet(status_style(TEXT_MUTED))
        self._proj_hint.setWordWrap(True)
        proj_lay.addWidget(self._proj_hint)
        splitter.addWidget(proj_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 380])

        self._x_combo.currentIndexChanged.connect(self._on_axes_changed)
        self._y_combo.currentIndexChanged.connect(self._on_axes_changed)
        self._mode_combo.currentIndexChanged.connect(self._render_projection)
        self._obj_combo.currentIndexChanged.connect(self._render_projection)
        self._update_btn.clicked.connect(self._on_update_clicked)

        # data state
        self._dof_names: List[str] = []
        self._obj_names: List[str] = []  # objective names (for the Objectives-vs-DOF view)
        self._records: List[dict] = []   # each: {"params": {...}, "primary": float}
        self._best_idx: Optional[int] = None
        self._minimize = False
        self._surface: Optional[dict] = None        # last computed surface payload
        self._surface_key: Optional[tuple] = None   # (x_name, y_name) it was for
        self._surface_multi: Optional[dict] = None      # last RGB phase-map payload
        self._surface_multi_key: Optional[tuple] = None
        self._slice: Optional[dict] = None          # last computed 1-D slice payload
        self._slice_key: Optional[str] = None       # x_name it was for
        # DOF bounds (name -> (lo, hi, kind)) for the fixed-DOF sliders.
        self._dof_bounds: Dict[str, tuple] = {}
        self._opt_names: List[str] = []  # optimized objective names (surface selector)
        # Fixed non-axis DOF values (name -> value) + their slider/label widgets.
        self._fixed_vals: Dict[str, float] = {}
        self._fixed_widgets: Dict[str, tuple] = {}  # name -> (slider, value_label)
        # Whether a model exists and no compute/scan is running (Update gating).
        self._compute_allowed = False

    # -- setup ------------------------------------------------------------
    def reset(self, dofs, minimize: bool,
              obj_names: Optional[List[str]] = None,
              opt_names: Optional[List[str]] = None) -> None:
        """Reset for a new run.

        ``dofs`` is a list of DOFSpec-like objects (``.name/.lo/.hi/.kind``);
        ``obj_names`` are ALL objectives (for composition color / points),
        ``opt_names`` the optimized ones (for the surface objective selector).
        """
        self._dof_names = [d.name for d in dofs]
        self._dof_bounds = {
            d.name: (float(d.lo), float(d.hi), getattr(d, "kind", "float"))
            for d in dofs
        }
        self._obj_names = list(obj_names or [])
        self._opt_names = list(opt_names or [])
        self._records = []
        self._best_idx = None
        self._minimize = minimize
        self._surface = None
        self._surface_key = None
        self._surface_multi = None
        self._surface_multi_key = None
        self._slice = None
        self._slice_key = None
        self._fixed_vals = {}
        self._measured_curve.setData([], [])
        self._best_curve.setData([], [])
        self._scatter.clear()
        self._best_marker.clear()
        self._surface_img.clear()
        self._surface_img.setVisible(False)
        if self._colorbar is not None:
            self._colorbar.setVisible(False)
        self._clear_obj_series()
        self._obj_combo.blockSignals(True)
        self._obj_combo.clear()
        self._obj_combo.addItems(self._opt_names)
        self._obj_combo.blockSignals(False)
        for combo in (self._x_combo, self._y_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._dof_names)
            combo.blockSignals(False)
        if len(self._dof_names) > 1:
            self._y_combo.setCurrentIndex(1)
        self._rebuild_fixed_sliders()
        self._render_projection()

    # -- fixed-DOF sliders ------------------------------------------------
    def _on_axes_changed(self) -> None:
        # X/Y selection changed -> the set of non-axis DOFs changed.
        self._rebuild_fixed_sliders()
        self._render_projection()

    def _rebuild_fixed_sliders(self) -> None:
        while self._fixed_form.rowCount():
            self._fixed_form.removeRow(0)
        self._fixed_widgets.clear()
        x, y = self._x_combo.currentText(), self._y_combo.currentText()
        others = [n for n in self._dof_names if n not in (x, y)]
        # Visibility is owned by _render_projection (mode-dependent); here we only
        # (re)build the widgets for the current non-axis DOFs.
        for name in others:
            lo, hi, kind = self._dof_bounds.get(name, (0.0, 1.0, "float"))
            val = self._fixed_vals.get(name, 0.5 * (lo + hi))
            self._fixed_vals[name] = val
            sld = QtWidgets.QSlider(Qt.Horizontal)
            sld.setRange(0, 1000)
            sld.setValue(int(round((val - lo) / (hi - lo) * 1000)) if hi > lo else 0)
            sld.setToolTip(f"Hold {name} constant at this value for the model surface.")
            lbl = QtWidgets.QLabel(f"{val:.4g}")
            lbl.setMinimumWidth(60)
            sld.valueChanged.connect(lambda v, nm=name: self._on_fixed_slider(nm, v))
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(sld, 1)
            rl.addWidget(lbl)
            self._fixed_form.addRow(f"{name} =", row)
            self._fixed_widgets[name] = (sld, lbl)

    def _on_fixed_slider(self, name: str, v: int) -> None:
        lo, hi, kind = self._dof_bounds.get(name, (0.0, 1.0, "float"))
        val = lo + (v / 1000.0) * (hi - lo)
        if kind == "int":
            val = float(round(val))
        self._fixed_vals[name] = val
        if name in self._fixed_widgets:
            self._fixed_widgets[name][1].setText(f"{val:.4g}")
        # A moved slider invalidates the computed model surface (different slice).
        self._surface_key = None
        self._surface_multi_key = None
        self._update_proj_hint()

    def fixed_overrides(self) -> Dict[str, float]:
        return dict(self._fixed_vals)

    def current_objective(self) -> Optional[str]:
        return self._obj_combo.currentText() or None

    def current_dof_pair(self) -> tuple:
        return self._x_combo.currentText(), self._y_combo.currentText()

    def set_update_enabled(self, enabled: bool) -> None:
        # "enabled" == a model exists and no compute/scan is in progress. The
        # actual button state also depends on the mode (see _refresh_update_btn).
        self._compute_allowed = enabled
        self._refresh_update_btn()

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

    def set_slice(self, payload: dict) -> None:
        """Store a computed 1-D model slice and display it (if axis/mode match)."""
        self._slice = payload
        self._slice_key = payload["x_name"]
        self._render_projection()

    def set_surface_multi(self, payload: dict) -> None:
        """Store a computed RGB phase-map surface and display it (if axes/mode match)."""
        self._surface_multi = payload
        self._surface_multi_key = (payload["x_name"], payload["y_name"])
        self._render_projection()

    def _set_legend(self, entries) -> None:
        """Rebuild the projection legend from (label, color) entries."""
        try:
            self._legend.clear()
        except Exception:  # noqa: BLE001
            return
        for label, color in entries:
            sample = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None),
                                        brush=pg.mkBrush(color))
            self._legend.addItem(sample, label)

    def _clear_obj_series(self) -> None:
        """Remove all per-objective scatter/curve items from the projection plot."""
        for item in list(self._obj_scatter.values()) + list(self._obj_curve.values()):
            try:
                self._proj.removeItem(item)
            except Exception:  # noqa: BLE001
                pass
        self._obj_scatter.clear()
        self._obj_curve.clear()
        # addLegend() is sticky; rebuild it so stale entries don't linger.
        try:
            self._legend.clear()
        except Exception:  # noqa: BLE001
            pass

    # -- projection -------------------------------------------------------
    def _on_update_clicked(self) -> None:
        mode = self._mode_combo.currentText()
        if mode == _MODE_OBJ_VS_DOF:
            xname = self._x_combo.currentText()
            if not xname:
                self._proj_hint.setText("Pick an X DOF first.")
                return
            self.slice_requested.emit(xname)
            return
        xname, yname = self.current_dof_pair()
        if not xname or not yname or xname == yname:
            self._proj_hint.setText("Pick two different DOFs for X and Y first.")
            return
        if mode == _MODE_PHASE_MAP:
            self.surface_multi_requested.emit(xname, yname)
        else:
            self.surface_requested.emit(xname, yname)

    def _surface_matches_axes(self) -> bool:
        return self._surface is not None and self._surface_key == self.current_dof_pair()

    def _surface_multi_matches_axes(self) -> bool:
        return (self._surface_multi is not None
                and self._surface_multi_key == self.current_dof_pair())

    def _slice_matches_axis(self) -> bool:
        return self._slice is not None and self._slice_key == self._x_combo.currentText()

    def _render_projection(self) -> None:
        mode = self._mode_combo.currentText()
        obj_vs_dof = mode == _MODE_OBJ_VS_DOF
        phase_map_mode = mode == _MODE_PHASE_MAP
        points_mode = mode == _MODE_POINTS
        single_obj_mode = mode in _SURFACE_MODES         # mean / sigma / acq
        uses_model = mode != _MODE_POINTS                # everything but raw points

        # --- consistent control state across ALL modes -----------------------
        # Y-axis picker: used by every 2-D mode; not by Objectives-vs-DOF (Y is
        # the objective values there).
        self._y_combo.setEnabled(not obj_vs_dof)
        self._y_label.setEnabled(not obj_vs_dof)
        self._y_combo.setToolTip(
            "Not used in this view — Y is each objective's value (one series per "
            "objective)." if obj_vs_dof else "")
        # Objective picker: only the single-objective surfaces (mean/σ/UCB) use it.
        self._obj_combo.setEnabled(single_obj_mode)
        self._obj_label.setEnabled(single_obj_mode)
        # Fixed-DOF sliders: only meaningful when a MODEL prediction is shown
        # (they slice the model); hidden for raw measured points.
        self._fixed_box.setVisible(bool(self._fixed_widgets) and uses_model)
        # Update button: nothing to compute for raw points; label matches the mode.
        self._refresh_update_btn()
        # Legend is rebuilt per mode; clear first so nothing stale lingers.
        self._set_legend([])

        # --- Objectives vs DOF: 1-D view -------------------------------------
        if obj_vs_dof:
            self._scatter.clear()
            self._best_marker.clear()
            self._surface_img.setVisible(False)
            if self._colorbar is not None:
                self._colorbar.setVisible(False)
            self._render_obj_vs_dof(self._x_combo.currentText())  # sets its own legend
            self._update_proj_hint()
            return

        # --- 2-D modes -------------------------------------------------------
        for item in list(self._obj_scatter.values()) + list(self._obj_curve.values()):
            item.setVisible(False)

        xname, yname = self.current_dof_pair()
        self._proj.setLabel("bottom", xname or "")
        self._proj.setLabel("left", yname or "")

        # Measured points: composition color in points/phase-map modes; on any
        # colored surface (scalar heatmap OR RGB map) draw an outline so they show.
        self._draw_points_overlay(
            xname, yname,
            surface_mode=(single_obj_mode or phase_map_mode),
            composition=(points_mode or phase_map_mode),
        )

        if phase_map_mode:
            self._render_phase_map(xname, yname)
        elif single_obj_mode and self._surface_matches_axes():
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

        # Composition legend (phase → anchor color) for the composition views.
        if points_mode or phase_map_mode:
            n = len(self._obj_names)
            self._set_legend(
                [(nm, phase_anchor_rgb(i, n)) for i, nm in enumerate(self._obj_names)])

        self._update_proj_hint()

    def _refresh_update_btn(self) -> None:
        """Enable/label the Update button consistently with the current mode.

        The button is only meaningful when a model prediction can be computed
        (``self._compute_allowed`` — set by the run lifecycle) AND the mode has
        something to compute (raw 'Measured points' has nothing).
        """
        mode = self._mode_combo.currentText()
        has_compute = mode != _MODE_POINTS
        self._update_btn.setEnabled(self._compute_allowed and has_compute)
        label = {
            _MODE_PHASE_MAP: "Update phase map",
            _MODE_OBJ_VS_DOF: "Update curves",
        }.get(mode, "Update surface")
        self._update_btn.setText(label)

    def _render_phase_map(self, xname: str, yname: str) -> None:
        """Render the predicted RGB phase-map (composition of optimized objectives)."""
        if not (xname and yname and xname != yname
                and self._surface_multi_matches_axes()):
            self._surface_img.setVisible(False)
            if self._colorbar is not None:
                self._colorbar.setVisible(False)
            return
        s = self._surface_multi
        names = self._obj_names
        n = len(names)
        # Sample any objective's grid for the shape (ny, nx).
        any_grid = next(iter(s["objectives"].values()))["mean"]
        ny, nx = any_grid.shape
        # Fraction stack over ALL phases (observe phases = 0, not modeled) so the
        # anchor mapping matches the measured-point composition colors.
        stack = np.zeros((ny * nx, n), dtype=float)
        for i, nm in enumerate(names):
            od = s["objectives"].get(nm)
            if od is not None:
                stack[:, i] = np.clip(np.asarray(od["mean"], dtype=float).ravel(), 0, None)
        rgb = composition_rgb(stack, n).reshape(ny, nx, 3)
        self._surface_img.setImage(rgb)
        self._surface_img.setRect(QtCore.QRectF(
            s["x_lo"], s["y_lo"], s["x_hi"] - s["x_lo"], s["y_hi"] - s["y_lo"]))
        self._surface_img.setVisible(True)
        if self._colorbar is not None:
            self._colorbar.setVisible(False)     # RGB image: scalar bar doesn't apply

    def _draw_points_overlay(self, xname, yname, *, surface_mode: bool,
                             composition: bool = False) -> None:
        if not self._records or not xname or not yname:
            self._scatter.clear()
            self._best_marker.clear()
            return
        xs = np.array([r["params"].get(xname, np.nan) for r in self._records])
        ys = np.array([r["params"].get(yname, np.nan) for r in self._records])
        if composition and self._obj_names:
            # Color each point by its measured phase composition (all objectives).
            n = len(self._obj_names)
            fr = np.array(
                [[r.get("objectives", {}).get(nm, 0.0) for nm in self._obj_names]
                 for r in self._records], dtype=float)
            rgb = composition_rgb(fr, n)          # (m, 3) uint8
            brushes = [pg.mkBrush(int(c[0]), int(c[1]), int(c[2])) for c in rgb]
            pen = pg.mkPen("k", width=0.5) if surface_mode else pg.mkPen(None)
            self._scatter.setData(x=xs, y=ys, brush=brushes, pen=pen)
        elif surface_mode:
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

    def _obj_color(self, idx: int) -> str:
        return _OBJ_PALETTE[idx % len(_OBJ_PALETTE)]

    def _render_obj_vs_dof(self, xname: str) -> None:
        """Draw each objective's value vs a single DOF (measured points + model curve).

        Measured points come from the recorded ``objectives`` per point (all
        objectives, including observe).  A model regression curve is overlaid for
        each objective present in the current 1-D slice payload (optimized only).
        """
        self._proj.setLabel("bottom", xname or "")
        self._proj.setLabel("left", "objective value")

        names = self._obj_names or (
            list(self._records[0]["objectives"].keys()) if self._records else [])
        show_model = self._slice_matches_axis()
        slice_objs = (self._slice or {}).get("objectives", {}) if show_model else {}

        legend_entries = []
        for i, name in enumerate(names):
            color = self._obj_color(i)
            legend_entries.append((name, color))
            # lazily create the two items for this objective (no name= -> the
            # legend is managed manually via _set_legend, avoiding duplicates)
            sc = self._obj_scatter.get(name)
            if sc is None:
                sc = pg.ScatterPlotItem(size=9, pen=pg.mkPen(None),
                                        brush=pg.mkBrush(color))
                self._proj.addItem(sc)
                self._obj_scatter[name] = sc
            cv = self._obj_curve.get(name)
            if cv is None:
                cv = self._proj.plot([], [], pen=pg.mkPen(color, width=2))
                self._obj_curve[name] = cv

            # measured points for this objective vs the chosen DOF
            xs, ys = [], []
            for r in self._records:
                if xname in r["params"] and name in r.get("objectives", {}):
                    xs.append(r["params"][xname])
                    ys.append(r["objectives"][name])
            sc.setData(x=np.array(xs), y=np.array(ys))
            sc.setVisible(True)

            # model regression curve (optimized objectives only)
            if name in slice_objs:
                cv.setData(np.asarray(self._slice["xi"], dtype=float),
                           np.asarray(slice_objs[name]["mean"], dtype=float))
                cv.setVisible(True)
            else:
                cv.setData([], [])
                cv.setVisible(False)
        self._set_legend(legend_entries)

    def _update_proj_hint(self) -> None:
        mode = self._mode_combo.currentText()
        if mode == _MODE_OBJ_VS_DOF:
            if self._slice_matches_axis():
                s = self._slice
                extra = ""
                if s.get("fixed"):
                    extra = "; other DOFs fixed at best (" + ", ".join(
                        f"{k}={v:.4g}" for k, v in s["fixed"].items()) + ")"
                self._proj_hint.setText(
                    "Points = measured phase fractions; lines = model curve for "
                    f"optimized objectives{extra}.")
            else:
                self._proj_hint.setText(
                    "Points = measured objective values vs the X DOF. Click "
                    "“Update surface” to overlay the model curve (optimized objectives).")
            return
        if mode == _MODE_PHASE_MAP:
            if self._surface_multi_matches_axes():
                s = self._surface_multi
                extra = ""
                if s.get("fixed"):
                    extra = "; fixed " + ", ".join(
                        f"{k}={v:.4g}" for k, v in s["fixed"].items())
                self._proj_hint.setText(
                    "Predicted phase composition (color = blend of optimized "
                    f"phases); points = measured composition{extra}.")
            else:
                self._proj_hint.setText(
                    "Points colored by measured phase composition. Click "
                    "“Update surface” for the predicted RGB phase map "
                    "(set the fixed-DOF sliders first).")
            return
        if mode == _MODE_POINTS:
            self._proj_hint.setText(
                "Color = measured phase composition at each sampled point "
                "(legend shows each phase's anchor color).")
        elif not self._surface_matches_axes():
            self._proj_hint.setText(
                "Click “Update surface” to compute the model surface for this DOF pair.")
        else:
            s = self._surface
            extra = ""
            if s.get("fixed"):
                extra = "; other DOFs fixed (" + ", ".join(
                    f"{k}={v:.4g}" for k, v in s["fixed"].items()) + ")"
            self._proj_hint.setText(f"{mode} of “{s['objective']}”{extra}.")


# ---------------------------------------------------------------------------
# DOF table
# ---------------------------------------------------------------------------

class _DOFTable(QtWidgets.QTableWidget):
    """Editable table of degrees of freedom (one motor per row)."""

    COLS = ["On", "Name", "Motor PV / Name", "Low limit", "High limit", "Type",
            "Protocol"]

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
        hdr.setSectionResizeMode(6, QtWidgets.QHeaderView.Fixed)    # Protocol
        self.setColumnWidth(0, 44)
        self.setColumnWidth(3, 98)
        self.setColumnWidth(4, 98)
        self.setColumnWidth(5, 80)
        self.setColumnWidth(6, 74)
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

        proto = QtWidgets.QComboBox()
        proto.addItems(["auto", "ca", "pva"])
        proto.setCurrentText(getattr(spec, "protocol", "auto") or "auto")
        proto.setToolTip(
            "I/O protocol for this motor's PV.\n"
            "auto = infer from a ca://|pva:// prefix (default Channel Access);\n"
            "ca = Channel Access; pva = PVAccess (driven via PvaSignal).")
        self.setCellWidget(r, 6, proto)

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
                    protocol=self.cellWidget(r, 6).currentText(),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Objective table
# ---------------------------------------------------------------------------

class _ObjectiveTable(QtWidgets.QTableWidget):
    """Editable table of objectives (usually one)."""

    COLS = ["Name", "Read PV", "Protocol", "Field", "Role"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)   # Name
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)   # Read PV
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)     # Protocol
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)   # Field
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)  # Role
        self.setColumnWidth(2, 74)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setMaximumHeight(180)

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

        proto = QtWidgets.QComboBox()
        proto.addItems(["auto", "ca", "pva"])
        proto.setCurrentText(getattr(spec, "protocol", "auto") or "auto")
        proto.setToolTip(
            "Read protocol. pva lets several objectives share ONE PVA object and "
            "each pick a different Field.")
        self.setCellWidget(r, 2, proto)

        # Field = which key/column of a multi-field PVA object this objective reads
        # (blank for a plain CA PV / single-value NTScalar).
        field_item = QtWidgets.QTableWidgetItem(getattr(spec, "field", "") or "")
        field_item.setToolTip(
            "For a multi-field PVA object, the field/column to optimize on "
            "(e.g. monoclinic / orthorhombic / tetragonal). Leave blank for a "
            "single-value channel.")
        self.setItem(r, 3, field_item)

        role = QtWidgets.QComboBox()
        role.addItems(["maximize", "minimize", "observe"])
        role.setCurrentText(getattr(spec, "role", None) or
                            ("minimize" if spec.minimize else "maximize"))
        role.setToolTip(
            "maximize / minimize are optimized by the model; observe is only "
            "recorded (plotted/logged), not optimized.")
        self.setCellWidget(r, 4, role)

    def split_pva_stream(self, channel: str, fields: List[str]) -> None:
        """Add one objective row per field, all bound to one PVA channel.

        A convenience for multi-fraction phase streams: pick the channel once,
        then get N rows (protocol=pva, same Read PV, distinct Field).  New rows
        default to ``observe`` so nothing is optimized until the user sets a
        maximize/minimize role — preventing an accidental "maximize everything".
        """
        existing = {
            (self.item(r, 0).text().strip() if self.item(r, 0) else "")
            for r in range(self.rowCount())
        }
        for fld in fields:
            fld = (fld or "").strip()
            if not fld:
                continue
            name = fld
            i = 2
            while name in existing:
                name = f"{fld}{i}"
                i += 1
            existing.add(name)
            self.add_objective(
                ObjectiveSpec(name=name, pv=channel, protocol="pva",
                              field=fld, role="observe")
            )

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
                    protocol=self.cellWidget(r, 2).currentText(),
                    field=(self.item(r, 3).text().strip() if self.item(r, 3) else ""),
                    role=self.cellWidget(r, 4).currentText(),
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
        self._surface_multi_worker: Optional[_SurfaceMultiWorker] = None
        self._slice_worker: Optional[_SliceWorker] = None
        self._stopping = False  # True between a Stop request and the worker ending
        # Central-profile config source (None -> local QSettings fallback) and the
        # name of the currently-loaded named setup within that profile.
        self._source = None
        self._source_label = ""
        self._current_setup: Optional[str] = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(self._build_control_panel(), 0)
        root.addWidget(self._build_display_panel(), 1)
        self._plots.surface_requested.connect(self._on_surface_requested)
        self._plots.surface_multi_requested.connect(self._on_surface_multi_requested)
        self._plots.slice_requested.connect(self._on_slice_requested)

        self._load_initial()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_control_panel(self) -> QtWidgets.QWidget:
        # ---- scrollable setup area -------------------------------------
        content = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(content)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(8)

        # Profile + named-setup picker (follows the app-selected profile)
        outer.addLayout(self._build_profile_row())

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
        split_obj = self._small_button("⑂ Split PVA stream…")
        split_obj.setToolTip(
            "Bind several objectives to ONE multi-field PVA object: pick the "
            "channel, then get one row per field (each selectable as "
            "maximize/minimize/observe).")
        add_obj.clicked.connect(lambda: self._obj_table.add_objective())
        rm_obj.clicked.connect(self._obj_table.remove_selected)
        split_obj.clicked.connect(self._on_split_pva)
        obj_btns.addWidget(add_obj)
        obj_btns.addWidget(rm_obj)
        obj_btns.addWidget(split_obj)
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
        self._simulate = QtWidgets.QCheckBox("Simulate (offline, no EPICS)")
        run_form.addRow("", self._simulate)
        outer.addLayout(run_form)
        self._update_total()

        # Advanced-Agent kwargs are no longer edited in the GUI (rarely useful);
        # any value from a loaded setup is carried through here and re-saved.
        self._acq_kwargs_value: Dict[str, Any] = {}

        # ---- Advanced (optional): not needed for a simple move→read scan -----
        adv_box = QtWidgets.QGroupBox("Advanced (optional) — leave blank for a simple scan")
        adv_form = QtWidgets.QFormLayout(adv_box)
        adv_form.setLabelAlignment(Qt.AlignRight)
        self._checkpoint_path = QtWidgets.QLineEdit()
        self._checkpoint_path.setPlaceholderText(
            "optional: save/resume path, e.g. /path/to/bo_checkpoint")
        self._checkpoint_path.setToolTip(
            "Optional. Persist the optimizer state (Ax trials + model) to this path "
            "so a campaign can be stopped and resumed. Blank = no checkpointing; "
            "the optimizer still runs (Ax: Sobol exploration → GP Bayesian).")
        adv_form.addRow("Checkpoint path:", self._checkpoint_path)
        self._commit_pv = QtWidgets.QLineEdit()
        self._commit_pv.setPlaceholderText("optional: commit PV, e.g. pva://FLA:fire")
        self._commit_pv.setToolTip(
            "Optional trigger written AFTER the DOFs move and BEFORE the read, each "
            "point (\"apply the values and measure\"). Blank = none (plain "
            "move→read; drive real device PVs directly). Point it at a commit "
            "bridge for experiments that need custom apply logic.")
        adv_form.addRow("Commit PV:", self._commit_pv)
        self._commit_done_pv = QtWidgets.QLineEdit()
        self._commit_done_pv.setPlaceholderText(
            "optional: commit-done counter PV, e.g. pva://FLA:done")
        self._commit_done_pv.setToolTip(
            "Optional counter the commit target advances ONLY after the commit fully "
            "completes. When set, the commit step blocks until it advances — a robust "
            "barrier independent of PVA put-completion timing.")
        adv_form.addRow("Commit-done counter PV:", self._commit_done_pv)
        self._commit_done_timeout = QtWidgets.QSpinBox()
        self._commit_done_timeout.setRange(5, 3600)
        self._commit_done_timeout.setValue(600)
        self._commit_done_timeout.setSuffix(" s")
        self._commit_done_timeout.setToolTip(
            "How long the commit step waits for the commit-done counter to advance. "
            "Set it comfortably ABOVE your real commit time (aligned move + flash "
            "power-up + fresh-fit wait) so slow moves/flashes don't time out.")
        adv_form.addRow("Commit timeout:", self._commit_done_timeout)
        # Bluesky env is only a fallback for a lean install that lacks bluesky.
        # With the --bayesian venv, bluesky is already importable and this is
        # ignored — leave blank.
        self._bluesky_env = QtWidgets.QLineEdit()
        self._bluesky_env.setPlaceholderText(
            "optional: /path/to/blueskyenv (blank = use DashPVA's own bluesky)")
        self._bluesky_env.setToolTip(
            "Optional override. DashPVA's .venv already has bluesky/ophyd/blop "
            "(from install.sh --bayesian), so the optimizer runs its own Bluesky "
            "RunEngine and this is IGNORED. Only set a conda-env path if bluesky "
            "is NOT importable in the DashPVA venv (a lean install); generic core "
            "bluesky/ophyd is enough.")
        adv_form.addRow("Bluesky env override:", self._bluesky_env)
        outer.addWidget(adv_box)

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
        btn_row.setSpacing(12)
        self._start_btn = QtWidgets.QPushButton("▶  Start")
        self._start_btn.setProperty("role", "success")
        self._start_btn.setMinimumHeight(34)
        self._stop_btn = QtWidgets.QPushButton("■  Stop")
        self._stop_btn.setProperty("role", "error")
        self._stop_btn.setMinimumHeight(34)
        self._stop_btn.setEnabled(False)
        self._reset_btn = QtWidgets.QPushButton("↺  Reset")
        self._reset_btn.setMinimumHeight(34)
        self._reset_btn.setToolTip(
            "Stop any run and clear everything (plots, model, status) back to scratch.")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        self._reset_btn.clicked.connect(self._on_reset)
        # Equal stretch so the three actions share the row width evenly (not
        # clustered/crowded) regardless of label length.
        btn_row.addWidget(self._start_btn, 1)
        btn_row.addWidget(self._stop_btn, 1)
        btn_row.addWidget(self._reset_btn, 1)
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
            acq_kwargs=dict(self._acq_kwargs_value),   # carried through (no GUI field)
            commit_pv=self._commit_pv.text().strip(),
            commit_done_pv=self._commit_done_pv.text().strip(),
            commit_done_timeout=float(self._commit_done_timeout.value()),
            checkpoint_path=self._checkpoint_path.text().strip(),
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
        self._acq_kwargs_value = dict(cfg.acq_kwargs)   # carried through (no GUI field)
        self._checkpoint_path.setText(getattr(cfg, "checkpoint_path", "") or "")
        self._commit_pv.setText(getattr(cfg, "commit_pv", "") or "")
        self._commit_done_pv.setText(getattr(cfg, "commit_done_pv", "") or "")
        self._commit_done_timeout.setValue(int(getattr(cfg, "commit_done_timeout", 600.0)))
        self._update_total()

    def _on_split_pva(self) -> None:
        """Bind several objectives to one multi-field PVA object.

        Ask for the channel, try to introspect its fields live (so the user can
        confirm the exact keys), then add one ``observe`` objective row per
        chosen field.  If introspection fails (channel offline), the user just
        types the field names.
        """
        channel, ok = QtWidgets.QInputDialog.getText(
            self, "Split PVA stream",
            "PVA channel holding several fields\n"
            "(e.g. pva://pvapy:phase:fractions):")
        if not ok or not channel.strip():
            return
        channel = channel.strip()

        discovered: List[str] = []
        try:
            from dashpva.viewer.bayesian.pva_signal import PvaSignal
            val = PvaSignal(channel).get()
            if isinstance(val, dict):
                discovered = [k for k, v in val.items()
                              if isinstance(v, (int, float))]
        except Exception as exc:  # noqa: BLE001 - offline is fine; user types fields
            logger.info("PVA introspection of %s failed: %s", channel, exc)

        prompt = (
            "Fields to add (comma-separated). Discovered live:"
            if discovered else
            "Channel not reachable for introspection — type the field names "
            "(comma-separated):")
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Split PVA stream", prompt, text=", ".join(discovered))
        if not ok or not text.strip():
            return
        fields = [f.strip() for f in text.split(",") if f.strip()]
        if not fields:
            return
        self._obj_table.split_pva_stream(channel, fields)
        self._status_lbl.setText(
            f"Status: added {len(fields)} objective(s) on {channel} "
            "(set roles: maximize/minimize/observe)")

    def _build_profile_row(self) -> QtWidgets.QVBoxLayout:
        """Active-profile label + named-setup picker (Load/Save/Save As/Delete)."""
        box = QtWidgets.QVBoxLayout()
        box.setSpacing(4)

        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(QtWidgets.QLabel("Profile:"))
        self._profile_label = QtWidgets.QLabel("—")
        self._profile_label.setStyleSheet("font-weight: 600;")
        self._profile_label.setToolTip(
            "The app-selected profile (set in the Workflow editor). The optimizer "
            "follows it; switch profiles there, then ↻ to re-read.")
        prow.addWidget(self._profile_label, 1)
        refresh = self._small_button("↻")
        refresh.setToolTip("Re-read the selected profile and its setups")
        refresh.clicked.connect(self._on_refresh_profile)
        prow.addWidget(refresh)
        box.addLayout(prow)

        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Setup:"))
        self._setup_combo = QtWidgets.QComboBox()
        self._setup_combo.setMinimumWidth(150)
        self._setup_combo.setToolTip("Named Bayesian setups stored in this profile")
        self._setup_combo.activated[str].connect(self._on_setup_selected)
        srow.addWidget(self._setup_combo, 1)
        self._btn_load = self._small_button("Load")
        self._btn_save = self._small_button("Save")
        self._btn_save_as = self._small_button("Save As…")
        self._btn_delete = self._small_button("Delete")
        self._btn_load.clicked.connect(self._on_load_setup)
        self._btn_save.clicked.connect(self._on_save_setup)
        self._btn_save_as.clicked.connect(self._on_save_as_setup)
        self._btn_delete.clicked.connect(self._on_delete_setup)
        for b in (self._btn_load, self._btn_save, self._btn_save_as, self._btn_delete):
            srow.addWidget(b)
        box.addLayout(srow)
        return box

    # ------------------------------------------------------------------
    # Config persistence: follow the selected profile, else local QSettings
    # ------------------------------------------------------------------

    def _default_config(self) -> OptimizerConfig:
        return OptimizerConfig(
            dofs=[
                DOFSpec(name="x", pv="", lo=0.0, hi=5.0),
                DOFSpec(name="y", pv="", lo=0.0, hi=5.0),
            ],
            objectives=[ObjectiveSpec()],
        )

    def _last_setup_key(self) -> str:
        label = (self._source_label or "local").replace("/", "_")
        return f"{_SETTINGS_LAST_SETUP}/{label}"

    def _refresh_source(self) -> None:
        """Resolve the app's active profile (None -> local QSettings setups)."""
        try:
            self._source, self._source_label = profile_store.active_source()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not resolve config source: %s", exc)
            self._source, self._source_label = None, ""
        self._profile_label.setText(
            self._source_label or "Local (no profile selected)")
        # Both profile and local modes hold named setups, so all controls apply.
        for b in (self._btn_load, self._btn_save, self._btn_save_as, self._btn_delete):
            b.setEnabled(True)

    def _populate_setups(self, names: list, pick: str) -> None:
        self._setup_combo.blockSignals(True)
        self._setup_combo.clear()
        self._setup_combo.addItems(names)
        if pick and pick in names:
            self._setup_combo.setCurrentText(pick)
        self._setup_combo.blockSignals(False)

    def _load_initial(self) -> None:
        """On launch: list the active store's setups and reload the last-used one."""
        self._refresh_source()
        if self._source is None:
            self._migrate_local_blob()
        names = self._store_list()
        last = self._settings.value(self._last_setup_key(), "", type=str)
        pick = last if last in names else (names[0] if names else "")
        self._populate_setups(names, pick)
        if pick:
            cfg = self._store_load(pick) or self._default_config()
            self._current_setup = pick
        else:
            cfg = self._default_config()
            self._current_setup = None
        self._apply_config(cfg)
        self._restore_local_ui()

    def _restore_local_ui(self) -> None:
        # Bluesky conda-env override (per-machine, local; optional).
        env = self._settings.value(_SETTINGS_BLUESKY_ENV, "", type=str)
        # Drop the legacy wrong APS default that used to be persisted, so it
        # doesn't reappear (the field is ignored anyway when bluesky is in .venv).
        if env and env.rstrip("/") == "/home/beams/USER6IDB/.conda/envs/6idb-bits":
            env = ""
            self._settings.setValue(_SETTINGS_BLUESKY_ENV, "")
        if env:
            self._bluesky_env.setText(env)
        # Simulate is never persisted: always start OFF (beamline safety).
        self._simulate.setChecked(False)

    # ---- setup store: profile (ConfigSource) or local (QSettings) ----------

    def _local_table(self) -> dict:
        raw = self._settings.value(_SETTINGS_LOCAL_SETUPS, "")
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except (ValueError, TypeError):
                pass
        return {}

    def _write_local_table(self, table: dict) -> None:
        self._settings.setValue(_SETTINGS_LOCAL_SETUPS, json.dumps(table))

    def _migrate_local_blob(self) -> None:
        """One-time: fold a legacy single local config into a named 'default' setup."""
        if self._local_table():
            return
        raw = self._settings.value(_SETTINGS_KEY, "")
        if not raw:
            return
        try:
            cfg = OptimizerConfig.from_dict(json.loads(raw))
        except (ValueError, TypeError):
            return
        if cfg.dofs:
            self._write_local_table({"default": cfg.to_dict()})

    def _store_list(self) -> list:
        if self._source is not None:
            return profile_store.list_setups(self._source)
        return sorted(self._local_table().keys())

    def _store_load(self, name: str) -> Optional[OptimizerConfig]:
        if self._source is not None:
            return profile_store.load_setup(self._source, name)
        data = self._local_table().get(name)
        if not isinstance(data, dict):
            return None
        try:
            return OptimizerConfig.from_dict(data)
        except (ValueError, TypeError) as exc:
            logger.warning("Could not parse local setup %r: %s", name, exc)
            return None

    def _store_save(self, name: str, cfg: OptimizerConfig) -> bool:
        if self._source is not None:
            return profile_store.save_setup(self._source, name, cfg)
        table = self._local_table()
        table[name] = cfg.to_dict()
        self._write_local_table(table)
        return True

    def _store_delete(self, name: str) -> bool:
        if self._source is not None:
            return profile_store.delete_setup(self._source, name)
        table = self._local_table()
        if name not in table:
            return False
        table.pop(name)
        self._write_local_table(table)
        return True

    def _save_env(self) -> None:
        self._settings.setValue(
            _SETTINGS_BLUESKY_ENV, self._bluesky_env.text().strip())

    def _persist_local_state(self) -> None:
        """Save local, non-exportable UI state: the env path + the last-used pick.

        Named setups (profile *and* local) are written only on explicit Save /
        Save As — closing or starting a run never auto-saves a setup.
        """
        self._save_env()
        if self._current_setup:
            self._settings.setValue(self._last_setup_key(), self._current_setup)

    # ---- setup actions ----------------------------------------------------

    def _on_refresh_profile(self) -> None:
        self._load_initial()
        self.statusBar().showMessage(
            f"Profile: {self._source_label or 'Local'}", 4000)

    def _on_setup_selected(self, name: str) -> None:
        if not name:
            return
        cfg = self._store_load(name)
        if cfg is None:
            return
        self._apply_config(cfg)
        self._current_setup = name
        self._settings.setValue(self._last_setup_key(), name)
        self.statusBar().showMessage(f"Loaded setup '{name}'", 4000)

    def _on_load_setup(self) -> None:
        name = self._setup_combo.currentText()
        if name:
            self._on_setup_selected(name)

    def _on_save_setup(self) -> None:
        if not self._current_setup:
            self._on_save_as_setup()
            return
        if self._store_save(self._current_setup, self._gather_config()):
            self._settings.setValue(self._last_setup_key(), self._current_setup)
            self.statusBar().showMessage(f"Saved setup '{self._current_setup}'", 4000)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Save failed", "Could not save the setup.")

    def _on_save_as_setup(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save setup as", "Setup name:", text=self._current_setup or "")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in self._store_list():
            if QtWidgets.QMessageBox.question(
                    self, "Overwrite?",
                    f"Setup '{name}' already exists. Overwrite?"
            ) != QtWidgets.QMessageBox.Yes:
                return
        if self._store_save(name, self._gather_config()):
            self._current_setup = name
            self._populate_setups(self._store_list(), name)
            self._settings.setValue(self._last_setup_key(), name)
            self.statusBar().showMessage(f"Saved setup '{name}'", 4000)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Save failed", "Could not save the setup.")

    def _on_delete_setup(self) -> None:
        name = self._setup_combo.currentText()
        if not name:
            return
        where = self._source_label or "Local"
        if QtWidgets.QMessageBox.question(
                self, "Delete setup",
                f"Delete setup '{name}' from {where}?"
        ) != QtWidgets.QMessageBox.Yes:
            return
        self._store_delete(name)
        names = self._store_list()
        newpick = names[0] if names else ""
        self._populate_setups(names, newpick)
        if newpick:
            self._on_setup_selected(newpick)
        else:
            self._current_setup = None
            self._apply_config(self._default_config())
        self.statusBar().showMessage(f"Deleted setup '{name}'", 4000)

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
                cfg.active_dofs(),
                cfg.optimized_objectives()[0].minimize,
                [o.name for o in cfg.active_objectives()],
                [o.name for o in cfg.optimized_objectives()],
            )
            self._best_lbl.setText("")

        self._persist_local_state()
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
        self._plots.reset([], minimize=False, obj_names=[])  # clear plots, surface, axes
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
            self._agent, self._agent_config, x_name, y_name,
            fixed_overrides=self._plots.fixed_overrides(),
            objective=self._plots.current_objective())
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

    def _on_surface_multi_requested(self, x_name: str, y_name: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Optimization running",
                "Wait for the current run to finish before computing a phase map.")
            return
        if self._agent is None or self._agent_config is None:
            QtWidgets.QMessageBox.information(
                self, "No model yet",
                "Run an optimization first, then click “Update surface”.")
            return
        self._plots.set_update_enabled(False)
        self._status_lbl.setText("Status: Computing phase map…")
        self._status_lbl.setStyleSheet(status_style(WARNING))
        self._surface_multi_worker = _SurfaceMultiWorker(
            self._agent, self._agent_config, x_name, y_name,
            fixed_overrides=self._plots.fixed_overrides())
        self._surface_multi_worker.done.connect(self._on_surface_multi_done)
        self._surface_multi_worker.failed.connect(self._on_surface_multi_failed)
        self._surface_multi_worker.start()

    def _on_surface_multi_done(self, payload: dict) -> None:
        self._plots.set_surface_multi(payload)
        self._plots.set_update_enabled(True)
        self._status_lbl.setText("Status: Phase map updated ✓")
        self._status_lbl.setStyleSheet(status_style(SUCCESS))

    def _on_surface_multi_failed(self, msg: str) -> None:
        self._plots.set_update_enabled(True)
        self._status_lbl.setText("Status: Phase map failed")
        self._status_lbl.setStyleSheet(status_style(ERROR))
        QtWidgets.QMessageBox.warning(
            self, "Phase map error",
            f"Could not compute the phase map:\n\n{msg}")

    def _on_slice_requested(self, x_name: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Optimization running",
                "Wait for the current run to finish before computing a slice.")
            return
        if self._agent is None or self._agent_config is None:
            QtWidgets.QMessageBox.information(
                self, "No model yet",
                "Run an optimization first, then click “Update surface”.")
            return
        self._plots.set_update_enabled(False)
        self._status_lbl.setText("Status: Computing slice…")
        self._status_lbl.setStyleSheet(status_style(WARNING))
        self._slice_worker = _SliceWorker(
            self._agent, self._agent_config, x_name,
            fixed_overrides=self._plots.fixed_overrides())
        self._slice_worker.done.connect(self._on_slice_done)
        self._slice_worker.failed.connect(self._on_slice_failed)
        self._slice_worker.start()

    def _on_slice_done(self, payload: dict) -> None:
        self._plots.set_slice(payload)
        self._plots.set_update_enabled(True)
        self._status_lbl.setText("Status: Slice updated ✓")
        self._status_lbl.setStyleSheet(status_style(SUCCESS))

    def _on_slice_failed(self, msg: str) -> None:
        self._plots.set_update_enabled(True)
        self._status_lbl.setText("Status: Slice failed")
        self._status_lbl.setStyleSheet(status_style(ERROR))
        QtWidgets.QMessageBox.warning(
            self, "Slice error",
            f"Could not compute the model slice:\n\n{msg}")

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
        self._persist_local_state()
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_abort()
            self._worker.wait(3000)
        # Stop the model-surface thread too, and drop its signals, so it can't fire
        # into a destroyed widget if the window is closed mid "Update surface".
        if self._surface_worker is not None and self._surface_worker.isRunning():
            try:
                self._surface_worker.done.disconnect()
                self._surface_worker.failed.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._surface_worker.wait(2000)
        if self._slice_worker is not None and self._slice_worker.isRunning():
            try:
                self._slice_worker.done.disconnect()
                self._slice_worker.failed.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._slice_worker.wait(2000)
        if self._surface_multi_worker is not None and self._surface_multi_worker.isRunning():
            try:
                self._surface_multi_worker.done.disconnect()
                self._surface_multi_worker.failed.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._surface_multi_worker.wait(2000)
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
