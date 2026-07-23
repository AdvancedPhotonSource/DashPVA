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
    protocol : str
        ``"auto"`` (infer from an optional ``ca://``/``pva://`` prefix on ``pv``,
        default Channel Access), ``"ca"``, or ``"pva"``.  A PVA DOF is driven
        through :class:`PvaSignal` (put-completion == the move/settle wait).
    """

    name: str
    pv: str
    lo: float
    hi: float
    kind: str = "float"
    enabled: bool = True
    protocol: str = "auto"


@dataclass
class ObjectiveSpec:
    """One optimization objective, read from a CA or PVA channel.

    Attributes
    ----------
    name : str
        Objective name reported to the optimizer (must be unique per config).
        Also names the readable, so the value is looked up by this name.
    pv : str
        The read channel for this objective's value
        (e.g. ``"detPV:Stats1:Total_RBV"``, ``"12idc:scaler1.S2"``, or a PVA
        channel like ``"pva://pvapy:phase:fractions"``).  Several objectives may
        share ONE PVA channel and each pick a different ``field`` from it.
    role : str
        ``"maximize"``, ``"minimize"``, or ``"observe"``.  ``observe`` objectives
        are read and recorded (plots/metadata) but **not** optimized — so one
        multi-field PVA stream can be, e.g., maximize=ortho / minimize=mono /
        observe=tetra.  ``role`` is the source of truth; ``minimize`` is kept in
        sync for back-compat.
    protocol : str
        ``"auto"`` (infer from a ``ca://``/``pva://`` prefix, default CA),
        ``"ca"``, or ``"pva"``.
    field : str
        For a multi-field PVA object, the key/column to select as this
        objective's scalar (blank for a plain NTScalar channel).
    minimize : bool
        Back-compat mirror of ``role == "minimize"``.
    """

    name: str = "intensity"
    pv: str = ""
    role: str = "maximize"
    protocol: str = "auto"
    field: str = ""
    minimize: bool = False

    def __post_init__(self) -> None:
        # ``role`` is the source of truth.  Two back-compat fallbacks:
        #   * an unknown/blank role derives from the legacy ``minimize`` flag;
        #   * legacy direct construction ``ObjectiveSpec(minimize=True)`` (role
        #     left at its "maximize" default) is honored as minimize.
        if self.role not in ("maximize", "minimize", "observe"):
            self.role = "minimize" if self.minimize else "maximize"
        elif self.role == "maximize" and self.minimize:
            self.role = "minimize"
        self.minimize = (self.role == "minimize")

    @property
    def optimized(self) -> bool:
        """True if this objective is handed to blop (maximize/minimize)."""
        return self.role in ("maximize", "minimize")


@dataclass
class OptimizerConfig:
    """Full description of a blop optimization run, as edited in the GUI."""

    dofs: List[DOFSpec] = field(default_factory=list)
    objectives: List[ObjectiveSpec] = field(default_factory=list)
    iterations: int = 30
    n_points: int = 1
    acq_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Optional explicit "fire"/commit channel put between the DOF move and the
    # objective read (e.g. the flash trigger).  Blank -> no fire step.  Its
    # put-completion is the fire-done / wait-for-fresh-result barrier.
    # Optional "commit" trigger written AFTER the DOF moves and BEFORE the read
    # each point (generic: "apply the DOF values and take a measurement").  Blank
    # -> no commit step (plain move->read).  Point it at a commit bridge, or leave
    # blank to drive real device PVs directly.
    commit_pv: str = ""
    commit_value: float = 1.0
    # Optional counter channel the commit target advances only after the commit
    # fully completes (e.g. a bridge's DONE counter).  When set, the commit device
    # blocks until it advances -> a robust move->commit->read barrier independent
    # of PVA put-completion timing.  Blank -> rely on put/settle only.
    commit_done_pv: str = ""
    # Optional path to persist/resume the Ax optimization state (blop Agent
    # checkpoint).  Blank -> no checkpointing.
    checkpoint_path: str = ""
    # Max seconds the commit step waits for commit_done_pv to advance.  Must
    # comfortably exceed the real commit time (aligned move + flash power-up +
    # fresh-fit wait); default is generous so slow moves/flashes don't time out.
    commit_done_timeout: float = 600.0

    # ---- convenience -----------------------------------------------------
    def active_dofs(self) -> List[DOFSpec]:
        return [d for d in self.dofs if d.enabled]

    def active_objectives(self) -> List[ObjectiveSpec]:
        """Every objective that is read each point (optimized AND observed)."""
        return [o for o in self.objectives] or [ObjectiveSpec()]

    def optimized_objectives(self) -> List[ObjectiveSpec]:
        """Objectives actually handed to blop (maximize/minimize only)."""
        opt = [o for o in self.active_objectives() if o.optimized]
        return opt or [ObjectiveSpec()]

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
        objs = self.active_objectives()
        onames = set()
        for o in objs:
            if not o.name.strip():
                return "Every objective needs a name."
            if o.name in onames:
                return f"Duplicate objective name: {o.name!r}."
            onames.add(o.name)
            if o.role not in ("maximize", "minimize", "observe"):
                return f"Objective {o.name!r}: role must be maximize/minimize/observe."
            if o.protocol not in ("auto", "ca", "pva"):
                return f"Objective {o.name!r}: protocol must be auto/ca/pva."
            if require_devices and not o.pv.strip():
                return f"Objective {o.name!r} is missing a read PV."
        if not any(o.optimized for o in objs):
            return "At least one objective must be maximize or minimize."
        for d in dofs:
            if d.protocol not in ("auto", "ca", "pva"):
                return f"DOF {d.name!r}: protocol must be auto/ca/pva."
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
            "ITERATIONS": self.iterations,
            "N_POINTS": self.n_points,
            "ACQ_KWARGS": dict(self.acq_kwargs),
            "COMMIT_PV": self.commit_pv,
            "COMMIT_VALUE": self.commit_value,
            "COMMIT_DONE_PV": self.commit_done_pv,
            "COMMIT_DONE_TIMEOUT": self.commit_done_timeout,
            "CHECKPOINT_PATH": self.checkpoint_path,
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
                protocol=d.get("protocol", "auto"),
            )
            for d in data.get("DOFS", [])
        ]
        objs = []
        for o in data.get("OBJECTIVES", []):
            # ``role`` wins when present; else derive from the legacy ``minimize``.
            role = o.get("role")
            if role not in ("maximize", "minimize", "observe"):
                role = "minimize" if bool(o.get("minimize", False)) else "maximize"
            objs.append(
                ObjectiveSpec(
                    name=o.get("name", "intensity"),
                    pv=o.get("pv", ""),
                    role=role,
                    protocol=o.get("protocol", "auto"),
                    field=o.get("field", ""),
                )
            )
        return cls(
            dofs=dofs,
            objectives=objs,
            iterations=int(data.get("ITERATIONS", 30)),
            n_points=int(data.get("N_POINTS", 1)),
            acq_kwargs=dict(data.get("ACQ_KWARGS", {})),
            # New COMMIT_* keys; fall back to the legacy FIRE_* keys for setups
            # saved before the rename.
            commit_pv=data.get("COMMIT_PV", data.get("FIRE_PV", "")),
            commit_value=float(data.get("COMMIT_VALUE", data.get("FIRE_VALUE", 1.0))),
            commit_done_pv=data.get("COMMIT_DONE_PV", data.get("FIRE_DONE_PV", "")),
            commit_done_timeout=float(data.get("COMMIT_DONE_TIMEOUT", 600.0)),
            checkpoint_path=data.get("CHECKPOINT_PATH", ""),
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


def select_field(value: Any, field: Optional[str] = None) -> float:
    """Coerce a reading *value* (scalar or multi-field dict) to one float.

    Used for PVA objectives that read a whole structure once: ``value`` may be a
    ``{field: scalar}`` dict (a multi-fraction phase stream), a list/array, or a
    plain number.  ``field`` names the wanted key when ``value`` is a dict.

    Raises
    ------
    KeyError
        If ``field`` is given but not present in a dict value.
    ValueError
        If no numeric value can be resolved.
    """
    if isinstance(value, dict):
        if field:
            if field not in value:
                raise KeyError(
                    f"field={field!r} not in PVA object; available: {list(value)}"
                )
            return float(value[field])
        # No field named: take the first numeric leaf.
        for v in value.values():
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v)
        raise ValueError(f"No numeric field in PVA object; keys: {list(value)}")
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty array reading; cannot pick a scalar.")
        return float(value[0])
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise ValueError(f"Non-numeric reading value: {value!r}")


def _objective_reading_value(reading: dict, dev: Any, o: "ObjectiveSpec") -> float:
    """Pull objective *o*'s scalar out of a ``trigger_and_read`` reading.

    A PVA structure readable reports one key per field (``{name}_{field}``); a
    plain scalar reports a single ``{name}`` key.  Resolve the field-specific key
    first, then the device name, then a trailing-suffix match.  If the objective
    reads a decomposed structure but names no field, raise a clear error listing
    the available fields — never silently optimize on an arbitrary one.
    """
    base = getattr(dev, "name", o.name)
    if o.field:
        for key in (f"{base}_{o.field}", base, o.name):
            if key in reading:
                return select_field(reading[key]["value"], o.field)
        return select_field(extract_scalar(reading, o.field), o.field)
    if base in reading:
        return select_field(reading[base]["value"], o.field)
    fields = sorted(k[len(base) + 1:] for k in reading if k.startswith(base + "_"))
    if fields:
        raise ValueError(
            f"Objective {o.name!r} reads multi-field PV {o.pv!r}; set its Field to "
            f"one of {fields} (or use 'Split PVA stream…').")
    return select_field(extract_scalar(reading, o.name), o.field)


# ---------------------------------------------------------------------------
# Device resolution (IPython namespace -> EPICS -> simulation)
# ---------------------------------------------------------------------------

def _resolve_protocol(spec_protocol: str, pv: str) -> Tuple[str, str]:
    """Return ``(protocol, bare_pv)`` honoring an explicit spec protocol first.

    ``spec_protocol`` of ``"ca"``/``"pva"`` forces that protocol (any URI prefix
    on ``pv`` is still stripped).  ``"auto"`` infers from a ``ca://``/``pva://``
    prefix, defaulting to Channel Access.
    """
    from dashpva.viewer.bayesian.pva_signal import split_protocol

    proto, bare = split_protocol(pv, default="ca")
    if spec_protocol in ("ca", "pva"):
        return spec_protocol, bare
    return proto, bare

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
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Resolve the config's DOF + objective PVs to ophyd objects.

    Every field is a full PV:
      1. Look names up in the live IPython namespace (beamline session) first.
      2. Construct EPICS devices: each DOF as a settable ``EpicsSignal`` on the
         exact PV typed (write == read; put-completion is the move-wait), each
         objective as an ``EpicsSignalRO`` on its own read PV.
      3. Fall back to ``ophyd.sim`` devices so the GUI is testable offline.

    Returns
    -------
    (actuators, readables, simulated)
        ``actuators`` maps DOF name -> ophyd movable; ``readables`` maps objective
        name -> ophyd readable; ``simulated`` is ``True`` when sim devices were used.
    """
    dofs = config.active_dofs()
    objectives = config.active_objectives()

    if not simulate:
        ns = _ipython_ns()
        if ns is not None:
            acts = {d.name: ns.get(d.pv) for d in dofs}
            reads = {o.name: ns.get(o.pv) for o in objectives}
            if all(acts.values()) and all(reads.values()):
                logger.info("Resolved blop devices from IPython namespace.")
                _add_commit_device(config, acts, ns=ns)
                return acts, reads, False

        try:
            acts = {d.name: _make_dof_device(d) for d in dofs}
            reads = _make_objective_readables(objectives)
            _add_commit_device(config, acts)
        except Exception as exc:  # noqa: BLE001 - fall back to sim
            logger.warning("Device construction failed (%s); using simulation.", exc)
        else:
            # Actionable error at Start (propagates, not swallowed into sim) when a
            # PVA objective targets a structure without a valid field.
            _validate_pva_objective_fields(objectives, reads)
            logger.info("Constructed CA/PVA devices from channel strings.")
            return acts, reads, False

    motors, readables = _build_sim_devices(dofs, objectives)
    return motors, readables, True


def _make_dof_device(d: "DOFSpec"):
    """Build a settable device for one DOF (CA ``EpicsSignal`` or ``PvaSignal``)."""
    proto, bare = _resolve_protocol(d.protocol, d.pv)
    if proto == "pva":
        from dashpva.viewer.bayesian.pva_signal import PvaSignal
        return PvaSignal(bare, name=d.name)
    from ophyd import EpicsSignal
    return EpicsSignal(bare, name=d.name, put_complete=True)


def _make_objective_readables(objectives: List["ObjectiveSpec"]) -> Dict[str, Any]:
    """Map each objective name -> a readable device.

    CA objectives get their own ``EpicsSignalRO``.  PVA objectives that share a
    channel share ONE cached :class:`PvaSignal` (field=None), so the channel is
    read once per point; a structure is reported as one scalar data_key per field
    (``{channel}_{field}``) and each objective selects its field by name
    downstream (see :func:`_objective_reading_value`).
    """
    from ophyd import EpicsSignalRO

    reads: Dict[str, Any] = {}
    pva_cache: Dict[str, Any] = {}
    for o in objectives:
        proto, bare = _resolve_protocol(o.protocol, o.pv)
        if proto == "pva":
            dev = pva_cache.get(bare)
            if dev is None:
                from dashpva.viewer.bayesian.pva_signal import PvaSignal
                # name by channel so a shared reading has stable per-field keys.
                dev = PvaSignal(bare, name=bare, field=None)
                pva_cache[bare] = dev
            reads[o.name] = dev
        else:
            reads[o.name] = EpicsSignalRO(bare, name=o.name, kind="hinted")
    return reads


def _validate_pva_objective_fields(
    objectives: List["ObjectiveSpec"], reads: Dict[str, Any]
) -> None:
    """Fail fast (before the RunEngine) on a mis-targeted multi-field PVA objective.

    Best-effort: probe each shared PVA signal once.  A not-yet-connected PV is
    skipped (the RunEngine surfaces connection errors later); only a *connected*
    structure whose objective names a blank or unknown field raises — turning a
    cryptic mid-scan event-model schema error into an actionable message at Start.
    """
    from dashpva.viewer.bayesian.pva_signal import PvaSignal

    for o in objectives:
        dev = reads.get(o.name)
        if not isinstance(dev, PvaSignal):
            continue
        try:
            val = dev.get()
        except Exception:  # noqa: BLE001 - not connected yet; defer to runtime
            continue
        if not isinstance(val, dict):
            continue
        fld = (o.field or "").strip()
        if not fld:
            raise ValueError(
                f"Objective {o.name!r} reads multi-field PV {o.pv!r}; set its Field "
                f"to one of {sorted(val)} (or use 'Split PVA stream…').")
        if fld not in val:
            raise ValueError(
                f"Objective {o.name!r}: field {fld!r} not found in {o.pv!r}; "
                f"available: {sorted(val)}.")


def _add_commit_device(config: OptimizerConfig, acts: Dict[str, Any], ns=None) -> None:
    """Attach the optional commit device under the ``"__commit__"`` key.

    Stored in ``acts`` (never a DOF, so :func:`build_agent` ignores it) and
    written by :func:`blop_optimize_plan` between the DOF move and the read.
    """
    if not (config.commit_pv or "").strip():
        return
    if ns is not None:
        dev = ns.get(config.commit_pv)
        if dev is not None:
            acts["__commit__"] = dev
            return
    proto, bare = _resolve_protocol("auto", config.commit_pv)
    done = (config.commit_done_pv or "").strip() or None
    if proto == "pva":
        from dashpva.viewer.bayesian.pva_signal import PvaSignal
        acts["__commit__"] = PvaSignal(
            bare, name="__commit__", done_channel=done,
            done_timeout=config.commit_done_timeout)
    else:
        from ophyd import EpicsSignal
        acts["__commit__"] = EpicsSignal(bare, name="__commit__", put_complete=True)


def _build_sim_devices(dofs: List["DOFSpec"], objectives: List["ObjectiveSpec"]):
    """Build ``ophyd.sim`` motors + one synthetic readable per objective.

    Each objective readable returns a product of Gaussians peaked at the midpoint
    of every DOF's range, so a *maximize* run has a well-defined optimum to
    converge to (a *minimize* run heads for the edges).  Each readable is named by
    its objective, so the plan looks values up via ``reading[obj_name]``.
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

    readables = {
        o.name: SynSignal(func=_peak, name=o.name, labels={"detectors"})
        for o in (objectives or [ObjectiveSpec()])
    }
    return motors, readables


# ---------------------------------------------------------------------------
# Move motors outside a plan (Move-to-best / click-to-move)
# ---------------------------------------------------------------------------

def resolve_actuators(
    config: OptimizerConfig, *, simulate: bool = False
) -> Tuple[Dict[str, Any], bool]:
    """Resolve ONLY the DOF motors (never objectives) to ophyd movables.

    Used to move motors outside a Bluesky plan.  Resolving DOFs alone means a
    misconfigured *objective* PV cannot trigger the simulation fallback and
    silently move sim motors instead of the real ones.  Returns
    ``(actuators, simulated)``.
    """
    dofs = config.active_dofs()
    if not simulate:
        ns = _ipython_ns()
        if ns is not None:
            acts = {d.name: ns.get(d.pv) for d in dofs}
            if all(acts.values()):
                return acts, False
        try:
            return {d.name: _make_dof_device(d) for d in dofs}, False
        except Exception as exc:  # noqa: BLE001 - fall back to sim
            logger.warning("DOF construction failed (%s); using simulation.", exc)
    motors, _ = _build_sim_devices(dofs, [])
    return motors, True


def move_to_point(
    config: OptimizerConfig,
    params: Dict[str, float],
    *,
    simulate: bool = False,
    settle_timeout: float = 120.0,
) -> Tuple[Dict[str, float], bool]:
    """Drive each active DOF to ``params[name]`` (clamped to ``[lo, hi]``), blocking
    until settled.  Returns ``(moved, simulated)`` where ``moved`` maps DOF name ->
    the clamped position commanded.  DOFs absent from ``params`` are skipped; a
    failed/timed-out move raises ``RuntimeError``.
    """
    actuators, simulated = resolve_actuators(config, simulate=simulate)
    moved: Dict[str, float] = {}
    for d in config.active_dofs():
        if d.name not in params:
            continue
        dev = actuators.get(d.name)
        if dev is None:
            continue
        pos = float(np.clip(params[d.name], d.lo, d.hi))
        try:
            dev.set(pos).wait(timeout=settle_timeout)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Move of DOF {d.name!r} to {pos} failed: {exc}") from exc
        moved[d.name] = pos
    return moved, simulated


def read_positions(actuators: Dict[str, Any], names) -> Dict[str, float]:
    """Best-effort read of the current value of each named actuator.

    Returns ``{name: value}`` for those that read successfully (used to snapshot
    the DOFs' starting positions so a later Reset can offer to restore them)."""
    out: Dict[str, float] = {}
    for name in names:
        dev = actuators.get(name)
        if dev is None:
            continue
        try:
            reading = dev.read()
            out[name] = float(next(iter(reading.values()))["value"])
        except Exception:  # noqa: BLE001
            logger.warning("Could not read position of DOF %r", name)
    return out


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

    # Only maximize/minimize objectives go to blop; ``observe`` objectives are
    # read and recorded in the plan but never handed to the optimizer.
    opt_specs = config.optimized_objectives()
    objectives = [
        Objective(name=o.name, minimize=o.minimize) for o in opt_specs
    ]

    def _noop_eval(uid: str, suggestions: list) -> list:  # pragma: no cover
        # Never called: we drive suggest()/ingest() manually in the plan.
        return [{"_id": s.get("_id"), objectives[0].name: 0.0} for s in suggestions]

    # checkpoint_path is an explicit Agent param; take it from the dedicated
    # config field (drop any stale copy in acq_kwargs to avoid a duplicate kwarg).
    extra = dict(config.acq_kwargs)
    extra.pop("checkpoint_path", None)
    ckpt = (config.checkpoint_path or "").strip() or None

    agent = Agent(
        sensors=[],
        dofs=dofs,
        objectives=objectives,
        evaluation_function=_noop_eval,
        checkpoint_path=ckpt,
        **extra,
    )
    return agent


# ---------------------------------------------------------------------------
# The Bluesky optimization plan (suggest -> move -> read -> ingest)
# ---------------------------------------------------------------------------

def blop_optimize_plan(
    agent,
    actuators: Dict[str, Any],
    readables: Dict[str, Any],
    config: OptimizerConfig,
    on_point: Optional[Callable[[dict], None]] = None,
):
    """Bluesky plan that runs the blop optimization loop.

    For each of ``config.iterations`` iterations it asks the agent for
    ``config.n_points`` suggestions, moves the motors to each, triggers and
    reads the objective readables, extracts each objective's scalar, and ingests
    the outcomes back into the agent.

    Parameters
    ----------
    agent : blop.ax.Agent
        Built via :func:`build_agent`.
    actuators : dict
        DOF name -> ophyd movable.  Keys must match the agent's DOF names.
    readables : dict
        Objective name -> ophyd readable; all triggered/read at each point.
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
    optimized = [o for o in objectives if o.optimized]
    commit_dev = actuators.get("__commit__")

    # Unique readable set: PVA objectives sharing a channel share one device, so
    # dedup by object identity to read each channel exactly once per point.
    read_devices: List[Any] = []
    _seen_ids = set()
    for dev in list(readables.values()) + [actuators[d.name] for d in active_dofs]:
        if id(dev) not in _seen_ids:
            _seen_ids.add(id(dev))
            read_devices.append(dev)

    _md = {
        "plan_name": "blop_optimize",
        "dofs": [d.name for d in active_dofs],
        "dof_pvs": [d.pv for d in active_dofs],
        "dof_protocols": [d.protocol for d in active_dofs],
        "objectives": [o.name for o in objectives],
        "objective_pvs": [o.pv for o in objectives],
        "objective_roles": [o.role for o in objectives],
        "objective_fields": [o.field for o in objectives],
        "optimized": [o.name for o in optimized],
        "commit_pv": config.commit_pv,
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
                # BARRIER: bps.mv waits every DOF's Status -> all DOFs settled.
                yield from bps.mv(*moves)

                # Optional commit step AFTER the move, BEFORE the read.  Its
                # put-completion (or done-counter barrier) is what guarantees the
                # applied change is complete and a fresh result is available,
                # i.e. move -> commit -> read ordering.
                if commit_dev is not None:
                    yield from bps.mv(commit_dev, config.commit_value)

                reading = yield from bps.trigger_and_read(read_devices, name="primary")

                outcome = {"_id": sug["_id"]}
                obj_values: Dict[str, float] = {}
                for o in objectives:
                    val = _objective_reading_value(
                        reading, readables.get(o.name), o)
                    obj_values[o.name] = val
                    # Only optimized objectives are ingested into blop; observe
                    # objectives are recorded (on_point/plots) but not optimized.
                    if o.optimized:
                        outcome[o.name] = val
                outcomes.append(outcome)

                state["index"] += 1
                if on_point is not None:
                    params = {d.name: float(sug[d.name]) for d in active_dofs}
                    primary_name = (optimized or objectives)[0].name
                    on_point(
                        {
                            "params": params,
                            "objectives": obj_values,
                            "primary": obj_values[primary_name],
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


def _best_params(agent) -> Dict[str, Any]:
    """Best parameterization from the Ax client, or ``{}`` if none yet."""
    try:
        result = agent.ax_client.get_best_parameterization()
        return dict(result[0])  # (parameters, metrics, ...)
    except Exception:  # noqa: BLE001 - fall back to midpoints
        return {}


def _grid_points_2d(
    config: OptimizerConfig,
    x_name: str,
    y_name: str,
    grid_n: int,
    best_params: Dict[str, Any],
    fixed_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, List[dict], Dict[str, Any]]:
    """Build the (xi, yi, points, fixed) for a 2-D DOF slice.

    Non-axis DOFs are fixed at ``fixed_overrides[name]`` when provided, else the
    best point, else the range midpoint.  ``points`` is row-major (y outer, x
    inner) so a reshape to ``(grid_n, grid_n)`` is (ny, nx).
    """
    dofs = config.active_dofs()
    by_name = {d.name: d for d in dofs}
    xd, yd = by_name[x_name], by_name[y_name]
    overrides = fixed_overrides or {}

    def _coerce(d: DOFSpec, v: float):
        return int(round(v)) if d.kind == "int" else float(v)

    fixed: Dict[str, Any] = {}
    for d in dofs:
        if d.name in (x_name, y_name):
            continue
        if d.name in overrides:
            val = overrides[d.name]
        else:
            val = best_params.get(d.name, 0.5 * (d.lo + d.hi))
        fixed[d.name] = _coerce(d, val)

    xi = np.linspace(xd.lo, xd.hi, grid_n)
    yi = np.linspace(yd.lo, yd.hi, grid_n)
    points = []
    for yv in yi:                       # y outer, x inner -> row-major (ny, nx)
        for xv in xi:
            p = dict(fixed)
            p[x_name] = _coerce(xd, xv)
            p[y_name] = _coerce(yd, yv)
            points.append(p)
    return xi, yi, points, fixed


def predict_surface(
    agent,
    config: OptimizerConfig,
    x_name: str,
    y_name: str,
    *,
    grid_n: int = 40,
    kappa: float = _ACQ_KAPPA,
    fixed_overrides: Optional[Dict[str, Any]] = None,
    objective: Optional[str] = None,
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
    by_name = {d.name: d for d in config.active_dofs()}
    xd, yd = by_name[x_name], by_name[y_name]
    opt = config.optimized_objectives()
    obj = next((o for o in opt if o.name == objective), opt[0])

    xi, yi, points, fixed = _grid_points_2d(
        config, x_name, y_name, grid_n, _best_params(agent), fixed_overrides)

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


def predict_surface_multi(
    agent,
    config: OptimizerConfig,
    x_name: str,
    y_name: str,
    *,
    grid_n: int = 40,
    fixed_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Predict every optimized objective over a 2-D DOF grid (for the RGB map).

    One ``ax_client.predict`` over the grid; returns per-optimized-objective
    ``mean``/``sem`` grids shaped ``(grid_n, grid_n)`` (ny, nx).  The GUI blends
    these per-pixel into a composition (phase-map) image.

    Returns
    -------
    dict with keys ``x_name, y_name, xi, yi, x_lo, x_hi, y_lo, y_hi, fixed`` and
    ``objectives``: ``{name: {"mean": (ny,nx), "sem": (ny,nx), "minimize": bool}}``.
    """
    by_name = {d.name: d for d in config.active_dofs()}
    xd, yd = by_name[x_name], by_name[y_name]

    xi, yi, points, fixed = _grid_points_2d(
        config, x_name, y_name, grid_n, _best_params(agent), fixed_overrides)
    preds = agent.ax_client.predict(points)

    objectives: Dict[str, Any] = {}
    for o in config.optimized_objectives():
        name = o.name
        try:
            mean = np.array([pt[name][0] for pt in preds], dtype=float).reshape(grid_n, grid_n)
            sem = np.array([pt[name][1] for pt in preds], dtype=float).reshape(grid_n, grid_n)
        except (KeyError, TypeError):
            continue
        objectives[name] = {"mean": mean, "sem": sem, "minimize": o.minimize}

    return {
        "x_name": x_name,
        "y_name": y_name,
        "xi": xi,
        "yi": yi,
        "x_lo": xd.lo,
        "x_hi": xd.hi,
        "y_lo": yd.lo,
        "y_hi": yd.hi,
        "fixed": fixed,
        "objectives": objectives,
    }


def predict_slice_1d(
    agent,
    config: OptimizerConfig,
    x_name: str,
    *,
    grid_n: int = 60,
    kappa: float = _ACQ_KAPPA,
    fixed_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Predict every optimized objective along a 1-D slice of one DOF.

    Complements :func:`predict_surface` (which is a 2-D grid for the *primary*
    objective only).  Here we sweep a single DOF ``x_name`` — fixing all other
    DOFs at the current best point (or each range's midpoint if none yet) — and
    return the model's predicted mean and standard error for **each** optimized
    objective.  The GUI overlays these as one regression curve per phase on top
    of the measured phase-fraction scatter ("Objectives vs DOF" view).

    Only ``maximize``/``minimize`` objectives are Ax metrics, so only those get a
    model curve; ``observe`` phases are shown by the panel as measured points.

    Returns
    -------
    dict with keys ``x_name, xi, x_lo, x_hi, fixed, kappa`` and ``objectives``:
    a mapping ``{obj_name: {"mean": ndarray, "sem": ndarray, "minimize": bool}}``.
    """
    dofs = config.active_dofs()
    by_name = {d.name: d for d in dofs}
    xd = by_name[x_name]
    opt_objs = config.optimized_objectives()
    overrides = fixed_overrides or {}

    best_params = _best_params(agent)

    def _coerce(d: DOFSpec, v: float):
        return int(round(v)) if d.kind == "int" else float(v)

    # Fix non-x DOFs at an override, else the best point, else the midpoint.
    fixed: Dict[str, Any] = {}
    for d in dofs:
        if d.name == x_name:
            continue
        if d.name in overrides:
            val = overrides[d.name]
        else:
            val = best_params.get(d.name, 0.5 * (d.lo + d.hi))
        fixed[d.name] = _coerce(d, val)

    xi = np.linspace(xd.lo, xd.hi, grid_n)
    points = []
    for xv in xi:
        p = dict(fixed)
        p[x_name] = _coerce(xd, xv)
        points.append(p)

    preds = agent.ax_client.predict(points)   # list of {metric: (mean, sem)}
    objectives: Dict[str, Any] = {}
    for o in opt_objs:
        name = o.name
        try:
            mean = np.array([pt[name][0] for pt in preds], dtype=float)
            sem = np.array([pt[name][1] for pt in preds], dtype=float)
        except (KeyError, TypeError):
            continue  # metric not modeled yet; skip its curve
        objectives[name] = {"mean": mean, "sem": sem, "minimize": o.minimize}

    return {
        "x_name": x_name,
        "xi": xi,
        "x_lo": xd.lo,
        "x_hi": xd.hi,
        "fixed": fixed,
        "kappa": kappa,
        "objectives": objectives,
    }
