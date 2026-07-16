"""
pva_signal.py
=============
A minimal **ophyd-compatible** signal backed by EPICS **PVAccess** (pvaccess),
so DashPVA's blop optimizer can drive DOFs and read objectives over PVA in
exactly the same plan that already works over Channel Access.

Why
---
DashPVA is PVA-native (the detector image, the streaming pyFAI 1-D, and the
phase-fraction stream all arrive as PVA objects), but ``blop_adapter`` originally
resolved every DOF/objective to an ``ophyd.EpicsSignal`` (Channel Access only).
``PvaSignal`` fills the gap with just enough of the ophyd Signal contract for
``bluesky.plan_stubs.mv`` and ``trigger_and_read`` to use it:

    read()               -> {name: {"value": ..., "timestamp": ...}}
    describe()           -> {name: {...}}
    set(value)           -> Status   (put + wait; used for DOFs by bps.mv)
    trigger()            -> Status   (immediately done; PVA is already live)
    get() / put(value)   -> scalar convenience
    read_configuration() / describe_configuration() -> {}
    wait_for_connection()

Multi-field PVA objects
-----------------------
One PVA channel can carry many scalars (e.g. a phase-fraction NTTable/structure
with ``monoclinic``/``orthorhombic``/``tetragonal``/``crystalline``).  A
``PvaSignal`` reads the **whole** structure once; a *field* selects the scalar.

* Construct with ``field=None`` (default) to expose the whole object under
  ``read()[name]["value"]`` (a plain dict of ``{field: scalar}`` when the object
  is a structure/NTTable, or the raw scalar for an NTScalar).  The adapter then
  caches ONE ``PvaSignal`` per unique channel and each objective picks its field
  from that single reading — so N objectives sharing a channel read it once.
* Or construct with an explicit ``field`` to make ``read()`` return just that
  scalar (handy for a DOF or a single-objective channel).

This module imports ``pvaccess`` lazily so merely importing it is cheap and does
not require the PVA stack to be present (e.g. on a dev laptop).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# URI-style protocol prefixes understood by ``split_protocol``.
_CA_PREFIX = "ca://"
_PVA_PREFIX = "pva://"


def split_protocol(pv: str, default: str = "ca") -> tuple[str, str]:
    """Return ``(protocol, bare_pv)`` for a possibly URI-prefixed PV string.

    ``"pva://x:y"`` -> ``("pva", "x:y")``; ``"ca://x:y"`` -> ``("ca", "x:y")``;
    ``"x:y"`` -> ``(default, "x:y")``.
    """
    s = (pv or "").strip()
    low = s.lower()
    if low.startswith(_PVA_PREFIX):
        return "pva", s[len(_PVA_PREFIX):]
    if low.startswith(_CA_PREFIX):
        return "ca", s[len(_CA_PREFIX):]
    return default, s


def _status(done: bool = True):
    """Return a finished ophyd ``Status`` (falls back to a duck-typed stub)."""
    try:
        from ophyd.status import Status
        st = Status()
        if done:
            st.set_finished()
        return st
    except Exception:  # noqa: BLE001 - ophyd absent (dev machine): duck-type it
        class _Done:
            def __init__(self):
                self.done = True
                self.success = True

            def wait(self, timeout=None):  # noqa: D401, ANN001
                return None

            def add_callback(self, cb):  # noqa: ANN001
                try:
                    cb(self)
                except Exception:  # noqa: BLE001
                    pass

        return _Done()


# EPICS/PVA housekeeping fields ignored when flattening a structure.
_META_KEYS = ("timeStamp", "timestamp", "alarm", "display", "control", "frame_number")


def _is_seq(x: Any) -> bool:
    """True for a sequence-like value (list/tuple/numpy array), not str/dict.

    pvaccess ``toDict()`` returns array fields as numpy ndarrays, so a plain
    ``isinstance(x, (list, tuple))`` misses them — hence this ``__len__`` check.
    """
    return hasattr(x, "__len__") and not isinstance(x, (str, bytes, dict))

# (names-array, values-array) key pairs treated as a named bundle, so a
# multi-field stream published as two parallel arrays (DashPVA's phase_fitter
# emits ``phase_names`` + ``phase_fractions``) reads back as ``{name: value}``.
_NAME_VALUE_PAIRS = (
    ("phase_names", "phase_fractions"),
    ("names", "fractions"),
    ("names", "values"),
    ("labels", "values"),
    ("keys", "values"),
)


def _zip_named_arrays(d: dict) -> Optional[dict]:
    """If ``d`` holds a (names[], values[]) pair, zip it into ``{name: value}``.

    Recognizes the explicit pairs in :data:`_NAME_VALUE_PAIRS` as well as any
    ``<x>_names`` / ``<x>_fractions`` (or ``<x>_values``) sibling arrays.
    Returns ``None`` when no such pairing is present.
    """
    def _pair(nk, vk):
        n, v = d.get(nk), d.get(vk)
        if _is_seq(n) and _is_seq(v) and len(n) == len(v) and len(n) > 0:
            return {str(name): _first_scalar(val) for name, val in zip(n, v)}
        return None

    for nk, vk in _NAME_VALUE_PAIRS:
        got = _pair(nk, vk)
        if got is not None:
            return got
    # Generic <x>_names / <x>_fractions|values siblings.
    for k in d:
        if k.endswith("_names"):
            stem = k[: -len("_names")]
            for suffix in ("_fractions", "_values"):
                got = _pair(k, stem + suffix)
                if got is not None:
                    return got
    return None


def _pv_to_pydict(pv_object) -> Any:
    """Best-effort convert a pvaccess ``PvObject`` to a Python value/dict.

    * (names[], values[]) parallel arrays -> ``{name: value}`` (see
      :func:`_zip_named_arrays`; this is how a phase-fraction stream is read).
    * NTScalar-like -> the scalar (``pv['value']``).
    * NTScalarArray -> a list.
    * structure with a ``value`` sub-structure -> ``{field: scalar}``.
    * Otherwise -> a one-level-flattened ``pv.toDict()``.
    """
    # Whole structure first: a named-array bundle wins (phase fractions).
    try:
        d = pv_object.toDict()
    except Exception:  # noqa: BLE001
        d = None
    if isinstance(d, dict):
        zipped = _zip_named_arrays(d)
        if zipped is not None:
            return zipped

    # Common EPICS normative-type shape: a top-level ``value``.
    try:
        val = pv_object["value"]
    except Exception:  # noqa: BLE001
        val = None

    if isinstance(val, (int, float, str, bool)):
        return val
    if _is_seq(val):
        return list(val)
    if isinstance(val, dict):
        return {k: _first_scalar(v) for k, v in val.items()}

    # Fall back to the whole structure, flattened one level.
    if isinstance(d, dict):
        return {k: _first_scalar(v) for k, v in d.items() if k not in _META_KEYS}
    return pv_object if d is None else d


def _first_scalar(v: Any) -> Any:
    """Coerce a nested pv sub-value to a scalar (first element of arrays)."""
    if isinstance(v, (int, float, str, bool)):
        return v
    if _is_seq(v):
        return v[0] if len(v) else None
    if isinstance(v, dict):
        # e.g. {"index": [...], "value": [...]} NTTable column, or nested scalar
        for key in ("value", "values"):
            if key in v:
                return _first_scalar(v[key])
        # otherwise first scalar-ish leaf
        for vv in v.values():
            return _first_scalar(vv)
    return v


class PvaSignal:
    """Ophyd-like signal over a single PVAccess channel.

    Parameters
    ----------
    pv : str
        The PVA channel name (URI ``pva://`` prefix is stripped if present).
    name : str, optional
        Ophyd-facing name (defaults to ``pv``).  This is the key under which
        ``read()``/``describe()`` report the value.
    field : str, optional
        If given, ``get()``/``read()`` return only this field of a multi-field
        PVA object.  If ``None`` the whole object is returned (dict for
        structures, scalar for NTScalar).
    put_timeout, get_timeout : float
        Per-operation timeouts (seconds).
    settle_time : float
        Extra dwell after a successful put (lets a soft-PV bridge react).
    done_channel : str, optional
        A monotonically-increasing counter channel that a downstream server
        (e.g. ``flash_bridge``'s ``FLA:done``) advances **only when the effect of
        this put has fully completed**.  When set, :meth:`set` snapshots the
        counter, puts, then blocks until the counter advances (or ``done_timeout``
        elapses) — a robust barrier that does NOT rely on put-completion timing
        (pvapy processes put callbacks asynchronously).
    done_timeout : float
        Max seconds to wait for ``done_channel`` to advance.
    """

    def __init__(
        self,
        pv: str,
        name: Optional[str] = None,
        *,
        field: Optional[str] = None,
        put_timeout: float = 10.0,
        get_timeout: float = 5.0,
        settle_time: float = 0.0,
        provider: Optional[str] = None,
        done_channel: Optional[str] = None,
        done_timeout: float = 130.0,
    ):
        _, bare = split_protocol(pv, default="pva")
        self.pv = bare
        self.name = name or bare
        self.field = field
        self.put_timeout = put_timeout
        self.get_timeout = get_timeout
        self.settle_time = settle_time
        self._provider = provider
        self.done_channel = (split_protocol(done_channel, default="pva")[1]
                             if done_channel else None)
        self.done_timeout = done_timeout
        self._done_ch = None            # lazy pvaccess.Channel for done_channel
        self._channel = None            # lazy pvaccess.Channel
        self.parent = None
        # Ophyd reads ``.hints`` on hinted readables.
        self.hints = {"fields": [self.name]}
        self.kind = "hinted"

    # -- connection -------------------------------------------------------
    def _ch(self):
        if self._channel is None:
            import pvaccess as pva
            prov = self._provider or pva.PVA
            self._channel = pva.Channel(self.pv, prov)
        return self._channel

    def wait_for_connection(self, timeout: float = 5.0) -> None:
        # A get() is the simplest connectivity probe pvaccess offers.
        try:
            self._ch().get("")
        except Exception as exc:  # noqa: BLE001
            raise TimeoutError(f"PVA channel {self.pv!r} not connected: {exc}")

    # -- reads ------------------------------------------------------------
    def _raw_get(self):
        return self._ch().get("")

    def get(self) -> Any:
        """Return the current value (scalar, dict of fields, or selected field)."""
        val = _pv_to_pydict(self._raw_get())
        if self.field is not None and isinstance(val, dict):
            return val.get(self.field)
        return val

    def read(self) -> Dict[str, Dict[str, Any]]:
        return {self.name: {"value": self.get(), "timestamp": time.time()}}

    def describe(self) -> Dict[str, Dict[str, Any]]:
        val = self.get()
        dtype = (
            "number" if isinstance(val, (int, float)) else
            "array" if isinstance(val, (list, tuple)) else
            "object" if isinstance(val, dict) else
            "string"
        )
        shape = [len(val)] if isinstance(val, (list, tuple)) else []
        return {self.name: {"source": f"PVA:{self.pv}", "dtype": dtype, "shape": shape}}

    def read_configuration(self) -> Dict[str, Any]:
        return {}

    def describe_configuration(self) -> Dict[str, Any]:
        return {}

    # -- trigger (readables) ---------------------------------------------
    def trigger(self):
        """PVA monitors are already live -> immediately-done Status."""
        return _status(done=True)

    # -- writes (DOFs) ----------------------------------------------------
    def put(self, value: Any) -> None:
        """Blocking put of a scalar onto the channel's ``value`` field.

        Uses pvaccess's bare-scalar ``put`` so the value is coerced to the
        channel's native scalar type.  This matters: putting a typed wrapper
        whose type differs from the record (e.g. ``PvInt`` to a ``double``
        record) is silently DROPPED by pvaccess, whereas ``ch.put(1)`` coerces
        correctly.
        """
        ch = self._ch()
        try:
            ch.put(value)                       # coerces int/float/str/bool
        except Exception:  # noqa: BLE001 - fall back to explicit typed puts
            import pvaccess as pva
            if isinstance(value, bool):
                ch.put(pva.PvBoolean(bool(value)))
            elif isinstance(value, int):
                ch.putInt(int(value))
            elif isinstance(value, float):
                ch.putDouble(float(value))
            elif isinstance(value, str):
                ch.put(pva.PvString(str(value)))
            else:
                raise

    def _read_done_counter(self) -> Optional[int]:
        if self._done_ch is None:
            import pvaccess as pva
            self._done_ch = pva.Channel(self.done_channel,
                                        self._provider or pva.PVA)
        try:
            obj = self._done_ch.get("")
            d = _pv_to_pydict(obj)
            if isinstance(d, dict):
                # take first numeric leaf
                for v in d.values():
                    if isinstance(v, (int, float)):
                        return int(v)
                return None
            return int(d)
        except Exception as exc:  # noqa: BLE001
            logger.debug("done-counter read failed on %s: %s", self.done_channel, exc)
            return None

    def set(self, value: Any):
        """Ophyd movable contract: put, then return a Status once settled.

        Plain DOF: ``put`` (pvaccess put returns after the server acknowledges
        the write) + optional ``settle_time`` -> finished Status.

        Fire/commit DOF with a ``done_channel``: snapshot the counter, put, then
        block until the counter advances (the downstream server increments it
        only after the full effect — e.g. flash + fresh fit — is done).  This is
        the true move->fire->read barrier; it does not depend on put-completion
        timing.
        """
        try:
            if self.done_channel:
                before = self._read_done_counter()
                self.put(value)
                if before is not None:
                    deadline = time.time() + self.done_timeout
                    while time.time() < deadline:
                        cur = self._read_done_counter()
                        if cur is not None and cur > before:
                            break
                        time.sleep(0.1)
                    else:
                        raise TimeoutError(
                            f"done channel {self.done_channel!r} did not advance "
                            f"past {before} within {self.done_timeout}s")
                elif self.settle_time:
                    time.sleep(self.settle_time)
            else:
                self.put(value)
                if self.settle_time:
                    time.sleep(self.settle_time)
            return _status(done=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("PvaSignal.set(%r) failed on %s", value, self.pv)
            st = _status(done=False)
            try:
                st.set_exception(exc)
            except Exception:  # noqa: BLE001
                pass
            return st

    # -- subscriptions (best-effort no-ops; plots poll via read()) --------
    def subscribe(self, cb, event_type=None, run=True):  # noqa: ANN001
        if run:
            try:
                cb()
            except TypeError:
                try:
                    cb(obj=self)
                except Exception:  # noqa: BLE001
                    pass
        return 0

    def clear_sub(self, cb):  # noqa: ANN001
        return None

    def __repr__(self) -> str:
        f = f", field={self.field!r}" if self.field else ""
        return f"PvaSignal(pv={self.pv!r}, name={self.name!r}{f})"
