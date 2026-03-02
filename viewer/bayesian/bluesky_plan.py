"""
bluesky_plan.py
===============
Bluesky Plan for 2-D Bayesian Optimization at APS 6-ID.

This module provides the ``bayesian2d`` plan generator, which orchestrates
a two-phase adaptive scan:

Phase 1 – Random Initialisation
    Randomly sample ``n_initial`` (x, y) positions within the specified
    motor bounds.  This ensures the GP has enough data spread across the
    domain before the acquisition function becomes meaningful.

Phase 2 – Bayesian Optimisation
    Use the ``BayesianOptimizer.ask()`` method to select each successive
    measurement location and ``BayesianOptimizer.tell()`` to ingest
    the returned detector value, iterating until ``maxpts`` total
    measurements have been collected.

All I/O goes through standard Bluesky/Ophyd primitives so that data are
automatically saved by the RunEngine to a Databroker.

Dependencies
------------
- bluesky  >= 1.9
- ophyd    >= 1.7
- numpy

Typical usage (in an IPython / Jupyter session with RE configured)
------------------------------------------------------------------
    from bluesky import RunEngine
    from bluesky.callbacks.best_effort import BestEffortCallback
    from ophyd.sim import SynAxis, SynSignal  # for offline testing

    from bayesian_engine import BayesianOptimizer
    from bluesky_plan  import bayesian2d

    RE = RunEngine()
    RE.subscribe(BestEffortCallback())

    opt = BayesianOptimizer(
        x_bounds=(0.0, 5.0), y_bounds=(0.0, 5.0),
        grid_nx=64, grid_ny=64,
    )

    RE(bayesian2d(
        optimizer  = opt,
        detector   = my_area_detector,       # Ophyd Device
        x_motor    = sample_x,              # EpicsMotor
        x_lo=0.0,  x_hi=5.0,
        y_motor    = sample_y,
        y_lo=0.0,  y_hi=5.0,
        maxpts     = 100,
        n_initial  = 10,
        scalar_key = "stats1_total",        # hinted field from detector
    ))
"""

from __future__ import annotations

import logging
import random
from typing import Any, Generator, List, Optional

import numpy as np

# Bluesky plan stubs and messages
# Inject 6idb-bits conda env so bluesky is importable from the uv venv
from .bluesky_compat import ensure_bluesky
ensure_bluesky()

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.utils import Msg

from .bayesian_engine import BayesianOptimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: extract a scalar value from a Bluesky reading dict
# ---------------------------------------------------------------------------

def _extract_scalar(reading: Optional[dict], scalar_key: Optional[str] = None) -> float:
    """
    Pull a numeric value out of a Bluesky detector reading.

    Bluesky's ``trigger_and_read`` returns a dict structured as::

        {
          "<device_name>_<signal_name>": {
              "value":     <the data>,
              "timestamp": <unix timestamp>,
          },
          ...
        }

    Parameters
    ----------
    reading : dict or None
        The raw reading returned by ``bps.trigger_and_read``.
        May be None if the read was dropped due to an error.
    scalar_key : str, optional
        Explicit key to look up (e.g., ``"stats1_total"``).
        If *None* the function tries common AreaDetector ROI stat keys and,
        as a last resort, returns the first numeric value found.

    Returns
    -------
    float
        The extracted scalar intensity value.

    Raises
    ------
    RuntimeError
        If ``reading`` is None (trigger_and_read returned nothing).
    KeyError
        If ``scalar_key`` is provided but not present in the reading.
    ValueError
        If no numeric value can be found in the reading at all.
    """
    # Guard against None – Bluesky can return None if the read was dropped
    if reading is None:
        raise RuntimeError(
            "trigger_and_read returned None. The detector reading may have "
            "been dropped due to an upstream error.  Check the RunEngine log."
        )

    if scalar_key is not None:
        # User has specified the exact key – look for it
        for full_key, payload in reading.items():
            # Match the *end* of the signal name to support varying device names
            if full_key.endswith(scalar_key):
                return float(payload["value"])
        raise KeyError(
            f"scalar_key='{scalar_key}' not found in detector reading. "
            f"Available keys: {list(reading.keys())}"
        )

    # ----------------------------------------------------------------
    # Auto-detect: try common AreaDetector ROI stats signal names
    # ----------------------------------------------------------------
    preferred_suffixes = [
        "stats1_total",
        "stats1_net",
        "stats1_mean_value",
        "total",
        "net",
        "mean_value",
        "intensity",
        "value",
    ]

    for suffix in preferred_suffixes:
        for full_key, payload in reading.items():
            if full_key.endswith(suffix):
                val = payload["value"]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    logger.debug(
                        "_extract_scalar(): auto-selected key '%s' → %.4g",
                        full_key, val,
                    )
                    return float(val)

    # Fallback: return the first numeric scalar we find
    for full_key, payload in reading.items():
        val = payload["value"]
        if isinstance(val, (int, float, np.integer, np.floating)):
            logger.warning(
                "_extract_scalar(): fallback to key '%s' – consider setting scalar_key explicitly.",
                full_key,
            )
            return float(val)

    raise ValueError(
        "Could not extract a scalar value from the detector reading. "
        f"Keys present: {list(reading.keys())}. "
        "Set scalar_key explicitly in bayesian2d()."
    )


# ---------------------------------------------------------------------------
# Main plan
# ---------------------------------------------------------------------------

def bayesian2d(
    optimizer: BayesianOptimizer,
    detector,
    x_motor,
    x_lo: float,
    x_hi: float,
    y_motor,
    y_lo: float,
    y_hi: float,
    maxpts: int,
    n_initial: int = 10,
    scalar_key: Optional[str] = None,
    md: Optional[dict] = None,
    **kwargs: Any,
) -> Generator[Msg, None, None]:
    """
    Two-phase Bayesian adaptive scan over a 2-D motor grid.

    Parameters
    ----------
    optimizer : BayesianOptimizer
        A pre-constructed (but empty) BayesianOptimizer instance whose
        ``x_bounds`` and ``y_bounds`` should match ``[x_lo, x_hi]`` and
        ``[y_lo, y_hi]``.
    detector : Ophyd Device
        The area-detector (or any readable device) to trigger.
    x_motor : Ophyd positioner
        Motor controlling the X axis of the sample.
    x_lo, x_hi : float
        Physical limits of the X scan range (same units as the motor).
    y_motor : Ophyd positioner
        Motor controlling the Y axis of the sample.
    y_lo, y_hi : float
        Physical limits of the Y scan range.
    maxpts : int
        Total number of measurements to take (phase 1 + phase 2).
    n_initial : int
        Number of random (phase 1) measurements before switching to
        Bayesian optimisation.  Must be < maxpts.
    scalar_key : str, optional
        Full or partial signal name to extract from the detector reading
        (e.g., ``"stats1_total"``).  If *None*, auto-detection is used.
    md : dict, optional
        Additional metadata merged into the RunStart document.
    **kwargs
        Passed through to ``bps.trigger_and_read`` (e.g., extra devices).

    Yields
    ------
    Msg
        Bluesky plan messages consumed by the RunEngine.

    Notes
    -----
    Data flow per iteration
    ~~~~~~~~~~~~~~~~~~~~~~~
    For every measurement point the plan:

    1. Emits a ``checkpoint`` so Bluesky knows it is safe to pause or
       rewind at this boundary.
    2. Moves both motors simultaneously via ``bps.mv``.
    3. Triggers the detector **and** both motors and reads all hinted
       signals via ``bps.trigger_and_read`` – this automatically emits
       an *Event* document with motor positions alongside detector values,
       which is critical for Databroker to record where each measurement
       was taken.
    4. Extracts the scalar intensity with ``_extract_scalar``.
    5. Calls ``optimizer.tell(x, y, value)`` which:
         a. Normalises the coordinate pair,
         b. Appends to the internal training tensors,
         c. Re-runs Adam MLL optimisation for ``train_iters`` steps.

    Hardware-limit protection
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Motor move limits are enforced by Ophyd before ``bps.mv`` yields; the
    RunEngine will raise ``LimitError`` which propagates naturally to the
    caller.  As an additional safety net, Bayesian-suggested coordinates
    are clamped to the scan bounds before issuing the move.

    Interruption / abort handling
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Each step yields a ``checkpoint`` message, enabling the RunEngine to
      pause cleanly between measurements and resume from the correct point.
    - The ``@bpp.run_decorator`` wrapper ensures a *RunStop* document is
      always emitted, regardless of where in the loop the interrupt occurs,
      keeping the Databroker record consistent.
    """
    if n_initial >= maxpts:
        raise ValueError(
            f"n_initial={n_initial} must be less than maxpts={maxpts}."
        )

    # ----------------------------------------------------------------
    # Build RunStart metadata
    # ----------------------------------------------------------------
    _md = {
        "plan_name": "bayesian2d",
        "x_motor": x_motor.name,
        "y_motor": y_motor.name,
        "detector": detector.name,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "maxpts": maxpts,
        "n_initial": n_initial,
        "scalar_key": scalar_key,
    }
    if md is not None:
        _md.update(md)

    # ----------------------------------------------------------------
    # Devices list for trigger_and_read
    # Include motors so their positions are recorded in Event documents.
    # This is critical: without motors in the devices list, Databroker
    # would only store detector values with no record of where the
    # sample stage was positioned for each reading.
    # ----------------------------------------------------------------
    devices: List = [detector, x_motor, y_motor]

    @bpp.run_decorator(md=_md)
    def _inner():
        """
        Inner generator decorated with @bpp.run_decorator so that
        RunStart / RunStop documents are automatically emitted.
        """
        # ----------------------------------------------------------
        # Phase 1: Random initialisation
        # ----------------------------------------------------------
        logger.info(
            "bayesian2d: starting Phase 1 – %d random measurements", n_initial,
        )

        for i in range(n_initial):
            # Checkpoint: safe point for pause/resume
            yield from bps.checkpoint()

            x_rand = random.uniform(x_lo, x_hi)
            y_rand = random.uniform(y_lo, y_hi)

            logger.debug(
                "Phase 1 [%d/%d]: moving to x=%.4g, y=%.4g",
                i + 1, n_initial, x_rand, y_rand,
            )

            # Move both motors simultaneously (Bluesky moves them in parallel)
            yield from bps.mv(x_motor, x_rand, y_motor, y_rand)

            # Trigger detector + read motors & detector → emits Event document
            # Including motors here ensures their positions appear in the
            # Event data alongside the detector value in Databroker.
            reading = yield from bps.trigger_and_read(devices, name="primary")

            # Extract scalar intensity
            try:
                value = _extract_scalar(reading, scalar_key)
            except (KeyError, ValueError, RuntimeError) as exc:
                logger.error(
                    "Phase 1 [%d]: could not extract scalar – %s", i + 1, exc
                )
                raise

            logger.info(
                "Phase 1 [%d/%d]: x=%.4g, y=%.4g → value=%.4g",
                i + 1, n_initial, x_rand, y_rand, value,
            )

            # Update GP model (normalise + append + MLL optimisation)
            optimizer.tell(x_rand, y_rand, value)

        # ----------------------------------------------------------
        # Phase 2: Bayesian optimisation
        # ----------------------------------------------------------
        n_bayes = maxpts - n_initial
        logger.info(
            "bayesian2d: starting Phase 2 – %d Bayesian optimisation steps",
            n_bayes,
        )

        for i in range(n_bayes):
            # Checkpoint: safe point for pause/resume
            yield from bps.checkpoint()

            # Query the GP acquisition function for the next best point
            try:
                x_next, y_next = optimizer.ask()
            except RuntimeError as exc:
                logger.critical("optimizer.ask() failed: %s", exc)
                raise

            # Clamp to scan bounds (safety net in case normalisation
            # produces a floating-point value just outside the range)
            x_next = float(np.clip(x_next, x_lo, x_hi))
            y_next = float(np.clip(y_next, y_lo, y_hi))

            logger.debug(
                "Phase 2 [%d/%d]: Bayesian next point → x=%.4g, y=%.4g",
                i + 1, n_bayes, x_next, y_next,
            )

            # Move motors to the Bayesian-recommended position
            yield from bps.mv(x_motor, x_next, y_motor, y_next)

            # Trigger detector + read motors & detector → Event document
            reading = yield from bps.trigger_and_read(devices, name="primary")

            # Extract scalar intensity
            try:
                value = _extract_scalar(reading, scalar_key)
            except (KeyError, ValueError, RuntimeError) as exc:
                logger.error(
                    "Phase 2 [%d]: could not extract scalar – %s", i + 1, exc
                )
                raise

            logger.info(
                "Phase 2 [%d/%d]: x=%.4g, y=%.4g → value=%.4g",
                i + 1, n_bayes, x_next, y_next, value,
            )

            # Feed measured value back into the GP (tell step)
            optimizer.tell(x_next, y_next, value)

    # Delegate to the decorated inner plan
    yield from _inner()
