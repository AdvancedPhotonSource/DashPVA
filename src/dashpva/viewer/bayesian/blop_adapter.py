"""
blop_adapter.py
===============
DashPVA-owned adapter around Bluesky's **blop** optimizer
(``blop.ax.Agent``, the Ax/BoTorch-backed Bayesian optimizer).

Why an adapter
--------------
``blop`` is the upstream optimization library (https://github.com/bluesky/blop)
and is mid-migration to its v1.0 ``blop.ax`` API.  Keeping every blop import and
every blop type-mapping in this one module means:

* the rest of DashPVA never imports blop directly (so the lean ``area-det``
  install never pulls torch/ax), and
* when blop's v1.0 API shifts, only this file changes.

How it is used
--------------
We drive blop through its *public optimizer* contract — ``agent.suggest()`` /
``agent.ingest()`` (see :class:`blop.protocols.Optimizer`) — inside our own
Bluesky plan (:func:`blop_optimize_plan`).  That reuses DashPVA's proven
"move motors → ``trigger_and_read`` → extract scalar" acquisition loop (ported
from the old ``bluesky_plan.bayesian2d``) and needs **no Tiled/Databroker**: the
detector value is read straight off the ``trigger_and_read`` reading.  The Agent
is used purely as the optimizer brain (``suggest``/``ingest``); its underlying Ax
model (``agent.ax_client``) powers the model-surface views (see
:func:`predict_surface`).

Adjustability
-------------
The optimization is described by plain DashPVA dataclasses
(:class:`DOFSpec`, :class:`ObjectiveSpec`, :class:`OptimizerConfig`) that the GUI
edits and persists.  Any number of motors (DOFs) and one-or-more objectives are
supported; mapping them onto blop ``RangeDOF`` / ``Objective`` objects happens in
:func:`build_agent`.

Heavy imports (``blop``, ``ophyd``, ``bluesky``) are performed lazily inside the
functions that need them so merely importing this module is cheap and safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Signal-name suffixes tried (in order) when an objective does not name an
# explicit ``signal_key``.  Ported from the old ``bluesky_plan._extract_scalar``.
PREFERRED_SUFFIXES: Tuple[str, ...] = (
    "stats1_total",
    "stats1_net",
    "stats1_mean_value",
    "total",
    "net",
    "mean_value",
    "intensity",
    "value",
)


# ---------------------------------------------------------------------------
# Configuration dataclasses (DashPVA-side, decoupled from blop's own classes)
# ---------------------------------------------------------------------------

@dataclass
class DOFSpec:
    """One degree of freedom (a motor the optimizer is allowed to move).

    Attributes
    ----------
    name : str
        Human/parameter name for the DOF (also the Ax parameter name when no
        live actuator is bound, e.g. simulation).  Keep it unique per config.
    pv : str
        Motor PV (or device name resolvable from the IPython namespace).
    lo, hi : float
        Inclusive search bounds, in motor units.  GUI-editable.
    kind : str
        ``"float"`` (continuous) or ``"int"`` (integer-valued motor).
    enabled : bool
        If ``False`` the row is ignored when building the agent.
    """

    name: str
    pv: str
    lo: float
    hi: float
    kind: str = "float"
    enabled: bool = True


@dataclass
class ObjectiveSpec:
    """One optimization objective read from the detector reading.

    Attributes
    ----------
    name : str
        Objective name reported to the optimizer (must be unique per config).
    signal_key : str
        Full or trailing signal name to pull from the detector reading
        (e.g. ``"stats1_total"``).  Blank → auto-detect via PREFERRED_SUFFIXES.
    minimize : bool
        ``False`` maximizes (peak/align), ``True`` minimizes.
    """

    name: str = "intensity"
    signal_key: str = ""
    minimize: bool = False


@dataclass
class OptimizerConfig:
    """Full description of a blop optimization run, as edited in the GUI."""

    dofs: List[DOFSpec] = field(default_factory=list)
    objectives: List[ObjectiveSpec] = field(default_factory=list)
    detector_pv: str = ""
    iterations: int = 30
    n_points: int = 1
    acq_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience -----------------------------------------------------
    def active_dofs(self) -> List[DOFSpec]:
        return [d for d in self.dofs if d.enabled]

    def active_objectives(self) -> List[ObjectiveSpec]:
        return [o for o in self.objectives] or [ObjectiveSpec()]

    def total_points(self) -> int:
        return max(1, self.iterations) * max(1, self.n_points)

    def validate(self, *, require_devices: bool = True) -> Optional[str]:
        """Return an error string if the config is invalid, else ``None``.

        ``require_devices`` may be set ``False`` for simulation runs, where motor
        and detector PVs are ignored (the adapter builds ``ophyd.sim`` devices).
        """
        dofs = self.active_dofs()
        if not dofs:
            return "At least one enabled DOF (motor) is required."
        seen = set()
        for d in dofs:
            if not d.name.strip():
                return "Every DOF needs a name."
            if d.name in seen:
                return f"Duplicate DOF name: {d.name!r}."
            seen.add(d.name)
            if require_devices and not d.pv.strip():
                return f"DOF {d.name!r} is missing a motor PV/name."
            if d.lo >= d.hi:
                return f"DOF {d.name!r}: low limit must be < high limit."
            if d.kind not in ("float", "int"):
                return f"DOF {d.name!r}: kind must be 'float' or 'int'."
        if require_devices and not self.detector_pv.strip():
            return "A detector PV/name is required."
        objs = self.active_objectives()
        onames = set()
        for o in objs:
            if not o.name.strip():
                return "Every objective needs a name."
            if o.name in onames:
                return f"Duplicate objective name: {o.name!r}."
            onames.add(o.name)
        if self.iterations < 1:
            return "Iterations must be >= 1."
        if self.n_points < 1:
            return "Points per iteration must be >= 1."
        return None

    # ---- (de)serialization for settings persistence ----------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "DOFS": [vars(d).copy() for d in self.dofs],
            "OBJECTIVES": [vars(o).copy() for o in self.objectives],
            "DETECTOR_PV": self.detector_pv,
            "ITERATIONS": self.iterations,
            "N_POINTS": self.n_points,
            "ACQ_KWARGS": dict(self.acq_kwargs),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        data = data or {}
        dofs = [
            DOFSpec(
                name=d.get("name", ""),
                pv=d.get("pv", ""),
                lo=float(d.get("lo", 0.0)),
                hi=float(d.get("hi", 1.0)),
                kind=d.get("kind", "float"),
                enabled=bool(d.get("enabled", True)),
            )
            for d in data.get("DOFS", [])
        ]
        objs = [
            ObjectiveSpec(
                name=o.get("name", "intensity"),
                signal_key=o.get("signal_key", ""),
                minimize=bool(o.get("minimize", False)),
            )
            for o in data.get("OBJECTIVES", [])
        ]
        return cls(
            dofs=dofs,
            objectives=objs,
            detector_pv=data.get("DETECTOR_PV", ""),
            iterations=int(data.get("ITERATIONS", 30)),
            n_points=int(data.get("N_POINTS", 1)),
            acq_kwargs=dict(data.get("ACQ_KWARGS", {})),
        )


# ---------------------------------------------------------------------------
# Scalar extraction from a Bluesky reading  (ported from old bluesky_plan)
# ---------------------------------------------------------------------------

def extract_scalar(reading: Optional[dict], signal_key: Optional[str] = None) -> float:
    """Pull a numeric value out of a Bluesky ``trigger_and_read`` reading.

    The reading is structured as ``{"<dev>_<sig>": {"value": x, "timestamp": t}}``.

    Parameters
    ----------
    reading : dict or None
        The reading returned by ``bps.trigger_and_read``.
    signal_key : str, optional
        Exact or trailing signal name to look up.  When ``None``/blank a set of
        common AreaDetector ROI-stat suffixes is tried, then the first numeric
        value found is used.

    Raises
    ------
    RuntimeError
        If ``reading`` is ``None``.
    KeyError
        If ``signal_key`` is given but not present.
    ValueError
        If no numeric value can be found.
    """
    if reading is None:
        raise RuntimeError(
            "trigger_and_read returned None; the detector reading may have been "
            "dropped. Check the RunEngine log."
        )

    if signal_key:
        for full_key, payload in reading.items():
            if full_key.endswith(signal_key):
                return float(payload["value"])
        raise KeyError(
            f"signal_key={signal_key!r} not found in detector reading. "
            f"Available keys: {list(reading.keys())}"
        )

    for suffix in PREFERRED_SUFFIXES:
        for full_key, payload in reading.items():
            if full_key.endswith(suffix):
                val = payload["value"]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    return float(val)

    for full_key, payload in reading.items():
        val = payload["value"]
        if isinstance(val, (int, float, np.integer, np.floating)):
            logger.warning(
                "extract_scalar(): fell back to key %r; set signal_key explicitly.",
                full_key,
            )
            return float(val)

    raise ValueError(
        "Could not extract a scalar from the detector reading. "
        f"Keys present: {list(reading.keys())}."
    )


# ---------------------------------------------------------------------------
# Device resolution (IPython namespace -> EPICS -> simulation)
# ---------------------------------------------------------------------------

def _ipython_ns() -> Optional[dict]:
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        return ip.user_ns
    except (NameError, AttributeError):
        return None


def resolve_devices(
    config: OptimizerConfig,
    *,
    simulate: bool = False,
) -> Tuple[Dict[str, Any], Any, bool]:
    """Resolve the config's DOF/detector names to ophyd objects.

    Strategy, per the old viewer ``_resolve_devices``:
      1. Look the names up in the live IPython namespace (beamline session).
      2. Construct EPICS devices (``EpicsMotor`` / ``EpicsSignalRO``).
      3. Fall back to ``ophyd.sim`` devices (a multi-Gaussian detector over the
         motors) so the GUI is fully testable offline.

    Returns
    -------
    (actuators, detector, simulated)
        ``actuators`` maps DOF name -> ophyd movable; ``detector`` is the
        readable; ``simulated`` is ``True`` when sim devices were used.
    """
    dofs = config.active_dofs()

    if not simulate:
        ns = _ipython_ns()
        if ns is not None:
            acts = {d.name: ns.get(d.pv) for d in dofs}
            det = ns.get(config.detector_pv)
            if all(acts.values()) and det is not None:
                logger.info("Resolved blop devices from IPython namespace.")
                return acts, det, False

        try:
            from ophyd import Device, EpicsMotor, EpicsSignalRO

            acts = {d.name: EpicsMotor(d.pv, name=d.name) for d in dofs}
            det_pv = config.detector_pv

            class _SimpleDetector(Device):
                total = EpicsSignalRO(det_pv, name="total", kind="hinted")

            det = _SimpleDetector(name="detector")
            logger.info("Constructed EPICS devices from PV strings.")
            return acts, det, False
        except Exception as exc:  # noqa: BLE001 - fall back to sim
            logger.warning("EPICS construction failed (%s); using simulation.", exc)

    motors, detector = _build_sim_devices(dofs)
    return motors, detector, True


def _build_sim_devices(dofs: List["DOFSpec"]):
    """Build ``ophyd.sim`` motors + a multi-Gaussian detector for offline tests.

    The detector returns a product of Gaussians peaked at the midpoint of each
    DOF's range, so a *maximize* run has a well-defined optimum to converge to.
    """
    from ophyd.sim import SynAxis, SynSignal

    motors = {d.name: SynAxis(name=d.name) for d in dofs}
    centers = {d.name: 0.5 * (d.lo + d.hi) for d in dofs}
    widths = {d.name: max(1e-6, 0.15 * (d.hi - d.lo)) for d in dofs}

    def _peak() -> float:
        val = 1.0
        for d in dofs:
            pos = motors[d.name].read()[d.name]["value"]
            val *= float(np.exp(-0.5 * ((pos - centers[d.name]) / widths[d.name]) ** 2))
        # Deterministic product-of-Gaussians peaked at the range midpoints.
        return 1000.0 * val

    detector = SynSignal(func=_peak, name="stats1_total", labels={"detectors"})
    return motors, detector


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def build_agent(config: OptimizerConfig, actuators: Dict[str, Any]):
    """Construct a ``blop.ax.Agent`` from the config and resolved actuators.

    The detector is *not* handed to the Agent as a blop ``sensor``: we read it
    ourselves in :func:`blop_optimize_plan` and feed values back via
    ``agent.ingest``.  A minimal no-op evaluation function satisfies the Agent
    constructor (it is never invoked on the suggest/ingest path).

    Parameters
    ----------
    config : OptimizerConfig
    actuators : dict
        DOF name -> ophyd movable (from :func:`resolve_devices`).
    """
    from blop.ax import Agent, Objective, RangeDOF

    dofs = []
    for d in config.active_dofs():
        actuator = actuators.get(d.name)
        if actuator is not None:
            dofs.append(
                RangeDOF(actuator=actuator, bounds=(d.lo, d.hi), parameter_type=d.kind)
            )
        else:
            dofs.append(
                RangeDOF(name=d.name, bounds=(d.lo, d.hi), parameter_type=d.kind)
            )

    objectives = [
        Objective(name=o.name, minimize=o.minimize) for o in config.active_objectives()
    ]

    def _noop_eval(uid: str, suggestions: list) -> list:  # pragma: no cover
        # Never called: we drive suggest()/ingest() manually in the plan.
        return [{"_id": s.get("_id"), objectives[0].name: 0.0} for s in suggestions]

    agent = Agent(
        sensors=[],
        dofs=dofs,
        objectives=objectives,
        evaluation_function=_noop_eval,
        **config.acq_kwargs,
    )
    return agent


# ---------------------------------------------------------------------------
# The Bluesky optimization plan (suggest -> move -> read -> ingest)
# ---------------------------------------------------------------------------

def blop_optimize_plan(
    agent,
    actuators: Dict[str, Any],
    detector,
    config: OptimizerConfig,
    on_point: Optional[Callable[[dict], None]] = None,
):
    """Bluesky plan that runs the blop optimization loop.

    For each of ``config.iterations`` iterations it asks the agent for
    ``config.n_points`` suggestions, moves the motors to each, triggers and
    reads the detector, extracts the objective scalar(s), and ingests the
    outcomes back into the agent.

    Parameters
    ----------
    agent : blop.ax.Agent
        Built via :func:`build_agent`.
    actuators : dict
        DOF name -> ophyd movable.  Keys must match the agent's DOF names.
    detector : ophyd readable
        Triggered/read at each point.
    config : OptimizerConfig
    on_point : callable, optional
        Called (in the RunEngine thread) after every measured point with a
        payload dict::

            {"params": {dof_name: pos, ...},
             "objectives": {obj_name: value, ...},
             "primary": float,        # first objective value
             "index": int}            # 1-based running count

        Use it to drive live plots (bridge to Qt via a queued signal).
    """
    import bluesky.plan_stubs as bps
    import bluesky.preprocessors as bpp

    active_dofs = config.active_dofs()
    objectives = config.active_objectives()
    read_devices = [detector] + [actuators[d.name] for d in active_dofs]

    _md = {
        "plan_name": "blop_optimize",
        "dofs": [d.name for d in active_dofs],
        "dof_pvs": [d.pv for d in active_dofs],
        "objectives": [o.name for o in objectives],
        "detector": getattr(detector, "name", config.detector_pv),
        "iterations": config.iterations,
        "n_points": config.n_points,
    }

    state = {"index": 0}

    @bpp.run_decorator(md=_md)
    def _inner():
        for it in range(config.iterations):
            yield from bps.checkpoint()

            suggestions = agent.suggest(config.n_points)
            outcomes = []
            for sug in suggestions:
                # Move every motor to its suggested position.
                moves: List[Any] = []
                for d in active_dofs:
                    pos = sug[d.name]
                    pos = float(np.clip(pos, d.lo, d.hi))  # safety clamp
                    moves += [actuators[d.name], pos]
                yield from bps.mv(*moves)

                reading = yield from bps.trigger_and_read(read_devices, name="primary")

                outcome = {"_id": sug["_id"]}
                obj_values: Dict[str, float] = {}
                for o in objectives:
                    val = extract_scalar(reading, o.signal_key or None)
                    outcome[o.name] = val
                    obj_values[o.name] = val
                outcomes.append(outcome)

                state["index"] += 1
                if on_point is not None:
                    params = {d.name: float(sug[d.name]) for d in active_dofs}
                    on_point(
                        {
                            "params": params,
                            "objectives": obj_values,
                            "primary": obj_values[objectives[0].name],
                            "index": state["index"],
                        }
                    )

            agent.ingest(outcomes)

    yield from _inner()


# ---------------------------------------------------------------------------
# Model-surface prediction (for the 2-D projection's contour/uncertainty views)
# ---------------------------------------------------------------------------

# Confidence multiplier for the Upper-Confidence-Bound acquisition view.
_ACQ_KAPPA = 2.0


def predict_surface(
    agent,
    config: OptimizerConfig,
    x_name: str,
    y_name: str,
    *,
    grid_n: int = 40,
    kappa: float = _ACQ_KAPPA,
) -> Dict[str, Any]:
    """Evaluate the agent's model over a 2-D grid for the primary objective.

    Uses ``agent.ax_client.predict`` to get the predicted mean and standard error
    across a grid spanning the two named DOFs (all other DOFs are fixed at the
    current best point, or each range's midpoint if no best point exists yet).
    Returns grids for three views:

    * ``mean`` — the predicted objective surface (the learned landscape),
    * ``sem``  — the model's uncertainty (standard error),
    * ``acq``  — an Upper/Lower-Confidence-Bound acquisition proxy
      (``mean + kappa*sem`` when maximizing, ``mean - kappa*sem`` when minimizing),
      i.e. the optimistic estimate the optimizer is drawn toward.

    Intended to run off the GUI thread (it does GP compute) and only when a scan
    is **not** actively mutating the agent.

    Returns
    -------
    dict with keys: ``x_name, y_name, xi, yi, x_lo, x_hi, y_lo, y_hi, mean, sem,
    acq, objective, minimize, kappa, fixed``.
    """
    dofs = config.active_dofs()
    by_name = {d.name: d for d in dofs}
    xd, yd = by_name[x_name], by_name[y_name]
    obj = config.active_objectives()[0]

    # Fix non-selected DOFs at the best point if available, else at the midpoint.
    # Use the Ax Client API (stable across blop/ax versions) rather than
    # Agent.get_best_points (only present on newer blop).
    best_params: Dict[str, Any] = {}
    try:
        result = agent.ax_client.get_best_parameterization()
        best_params = dict(result[0])  # (parameters, metrics, ...)
    except Exception:  # noqa: BLE001 - fall back to midpoints
        best_params = {}

    fixed: Dict[str, Any] = {}
    for d in dofs:
        if d.name in (x_name, y_name):
            continue
        val = best_params.get(d.name, 0.5 * (d.lo + d.hi))
        fixed[d.name] = int(round(val)) if d.kind == "int" else float(val)

    def _coerce(d: DOFSpec, v: float):
        return int(round(v)) if d.kind == "int" else float(v)

    xi = np.linspace(xd.lo, xd.hi, grid_n)
    yi = np.linspace(yd.lo, yd.hi, grid_n)
    points = []
    for yv in yi:                       # y outer, x inner -> row-major (ny, nx)
        for xv in xi:
            p = dict(fixed)
            p[x_name] = _coerce(xd, xv)
            p[y_name] = _coerce(yd, yv)
            points.append(p)

    preds = agent.ax_client.predict(points)   # list of {metric: (mean, sem)}
    name = obj.name
    mean = np.array([pt[name][0] for pt in preds], dtype=float).reshape(grid_n, grid_n)
    sem = np.array([pt[name][1] for pt in preds], dtype=float).reshape(grid_n, grid_n)
    acq = (mean - kappa * sem) if obj.minimize else (mean + kappa * sem)

    return {
        "x_name": x_name,
        "y_name": y_name,
        "xi": xi,
        "yi": yi,
        "x_lo": xd.lo,
        "x_hi": xd.hi,
        "y_lo": yd.lo,
        "y_hi": yd.hi,
        "mean": mean,
        "sem": sem,
        "acq": acq,
        "objective": name,
        "minimize": obj.minimize,
        "kappa": kappa,
        "fixed": fixed,
    }
