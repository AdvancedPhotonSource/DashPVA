"""End-to-end Bayesian-optimization simulation tests via the DashPVA blop integration.

These exercise the *real* ``blop.ax.Agent`` optimizer through DashPVA's adapter
(:func:`build_agent` -> :func:`blop_optimize_plan`) against simulated ophyd
devices whose detector computes a known analytic test function from the motor
positions.  They verify the full path —

    agent.suggest -> bps.mv -> bps.trigger_and_read -> extract_scalar -> agent.ingest

— actually converges toward the known optimum, for 1-D, 2-D and N-D problems,
single- and multi-point batches, and both maximize and minimize directions.

The 2-D Himmelblau case uses the same benchmark function as blop's own
``simple-experiment`` tutorial, so this is effectively blop's reference problem
run through DashPVA.

These are **slow** (each runs a real GP optimization loop).  They are skipped
automatically when blop / bluesky / ophyd are not installed::

    uv pip install -e '.[bayesian]'      # or '.[full]'
    pytest tests/integration/test_bayesian_simulation.py -v
"""

import math

import numpy as np
import pytest

pytest.importorskip("blop.ax")
pytest.importorskip("ophyd")
bluesky = pytest.importorskip("bluesky")

from bluesky import RunEngine  # noqa: E402
from ophyd.sim import SynAxis, SynSignal  # noqa: E402

from dashpva.viewer.bayesian.blop_adapter import (  # noqa: E402
    DOFSpec,
    ObjectiveSpec,
    OptimizerConfig,
    blop_optimize_plan,
    build_agent,
    predict_surface,
)

# ---------------------------------------------------------------------------
# Helpers: build simulated devices whose detector = analytic test function
# ---------------------------------------------------------------------------

def _make_devices(dof_names, func):
    """SynAxis motors + a detector that reads ``func({motor: pos})`` each trigger."""
    motors = {n: SynAxis(name=n) for n in dof_names}

    def _read() -> float:
        pos = {n: motors[n].read()[n]["value"] for n in dof_names}
        return float(func(pos))

    detector = SynSignal(func=_read, name="stats1_total", labels={"detectors"})
    return motors, detector


def _run(cfg, motors, detector):
    """Run the full DashPVA blop plan; return (points, agent)."""
    # Reduce run-to-run variance for stable assertions.
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except Exception:  # noqa: BLE001
        pass

    agent = build_agent(cfg, motors)
    points: list = []
    RE = RunEngine()
    RE(blop_optimize_plan(agent, motors, detector, cfg, on_point=points.append))
    return points, agent


def _best(points, minimize):
    fn = min if minimize else max
    return fn(points, key=lambda p: p["primary"])


# ---------------------------------------------------------------------------
# 2-D Himmelblau  (blop's own tutorial benchmark) — minimize
# ---------------------------------------------------------------------------

_HIMMELBLAU_MINIMA = [
    (3.0, 2.0),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
]


def _himmelblau(pos):
    x, y = pos["x"], pos["y"]
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


class TestHimmelblau2D:

    def test_minimize_converges_to_a_known_minimum(self):
        cfg = OptimizerConfig(
            dofs=[
                DOFSpec(name="x", pv="x", lo=-5.0, hi=5.0),
                DOFSpec(name="y", pv="y", lo=-5.0, hi=5.0),
            ],
            objectives=[ObjectiveSpec(name="himmelblau", minimize=True)],
            detector_pv="det",
            iterations=30,
            n_points=1,
        )
        motors, detector = _make_devices(["x", "y"], _himmelblau)
        points, _ = _run(cfg, motors, detector)

        assert len(points) == cfg.total_points()
        for p in points:
            assert -5.0 <= p["params"]["x"] <= 5.0
            assert -5.0 <= p["params"]["y"] <= 5.0
            assert math.isfinite(p["primary"])

        best = _best(points, minimize=True)
        # Himmelblau minima are 0; random sampling over [-5,5]^2 averages ~hundreds.
        assert best["primary"] < 25.0, f"did not converge: best={best['primary']:.3g}"

        # The best point should sit near one of the four global minima.
        bx, by = best["params"]["x"], best["params"]["y"]
        dist = min(math.hypot(bx - mx, by - my) for mx, my in _HIMMELBLAU_MINIMA)
        assert dist < 2.0, f"best ({bx:.2f},{by:.2f}) not near a known minimum"


# ---------------------------------------------------------------------------
# 2-D Gaussian peak — maximize
# ---------------------------------------------------------------------------

class TestGaussianPeak2D:

    def test_maximize_finds_the_peak(self):
        cx, cy, sigma, amp = 5.0, 5.0, 1.5, 1000.0

        def gauss(pos):
            return amp * math.exp(
                -0.5 * (((pos["x"] - cx) / sigma) ** 2 + ((pos["y"] - cy) / sigma) ** 2)
            )

        cfg = OptimizerConfig(
            dofs=[
                DOFSpec(name="x", pv="x", lo=0.0, hi=10.0),
                DOFSpec(name="y", pv="y", lo=0.0, hi=10.0),
            ],
            objectives=[ObjectiveSpec(name="intensity", minimize=False)],
            detector_pv="det",
            iterations=25,
            n_points=1,
        )
        motors, detector = _make_devices(["x", "y"], gauss)
        points, agent = _run(cfg, motors, detector)

        best = _best(points, minimize=False)
        assert best["primary"] > 100.0, f"did not climb the peak: {best['primary']:.3g}"
        assert math.hypot(best["params"]["x"] - cx, best["params"]["y"] - cy) < 3.0

        # The model's own best parameterization (stable Ax Client API) agrees the
        # optimum is near the peak.
        best_params, *_ = agent.ax_client.get_best_parameterization()
        assert abs(best_params["x"] - cx) < 3.0
        assert abs(best_params["y"] - cy) < 3.0


# ---------------------------------------------------------------------------
# N-D sphere — minimize (proves the GUI/adapter scale past 2 motors)
# ---------------------------------------------------------------------------

class TestHighDimensionalSphere:

    def test_four_dof_minimize_improves(self):
        names = ["a", "b", "c", "d"]
        centers = {n: 5.0 for n in names}

        def sphere(pos):
            return sum((pos[n] - centers[n]) ** 2 for n in names)

        cfg = OptimizerConfig(
            dofs=[DOFSpec(name=n, pv=n, lo=0.0, hi=10.0) for n in names],
            objectives=[ObjectiveSpec(name="sphere", minimize=True)],
            detector_pv="det",
            iterations=40,
            n_points=1,
        )
        motors, detector = _make_devices(names, sphere)
        points, _ = _run(cfg, motors, detector)

        assert len(points) == cfg.total_points()  # n_points=1 -> one per iteration
        primaries = [p["primary"] for p in points]
        best = min(primaries)
        # 4-D sphere min is 0 (worst corner ~100). Strong convergence proves the
        # adapter/agent scale past two motors.
        assert best < 1.0, f"did not converge in 4-D: best={best:.3g}"
        assert best <= np.median(primaries)


# ---------------------------------------------------------------------------
# Batch acquisition (n_points > 1)
# ---------------------------------------------------------------------------

class TestBatchAcquisition:

    def test_multi_point_per_iteration(self):
        cx, cy, sigma, amp = 5.0, 5.0, 2.0, 500.0

        def gauss(pos):
            return amp * math.exp(
                -0.5 * (((pos["x"] - cx) / sigma) ** 2 + ((pos["y"] - cy) / sigma) ** 2)
            )

        cfg = OptimizerConfig(
            dofs=[
                DOFSpec(name="x", pv="x", lo=0.0, hi=10.0),
                DOFSpec(name="y", pv="y", lo=0.0, hi=10.0),
            ],
            objectives=[ObjectiveSpec(name="intensity", minimize=False)],
            detector_pv="det",
            iterations=8,
            n_points=3,
        )
        motors, detector = _make_devices(["x", "y"], gauss)
        points, _ = _run(cfg, motors, detector)

        # Up to 8 iterations * 3 points = 24, but Ax may return fewer than
        # n_points in early (Sobol) batches, so total is an upper bound.
        assert cfg.iterations <= len(points) <= cfg.total_points()
        assert max(p["primary"] for p in points) > 50.0


# ---------------------------------------------------------------------------
# Direction handling — same function, opposite goals
# ---------------------------------------------------------------------------

class TestDirection:

    def test_minimize_vs_maximize_on_a_parabola(self):
        # f(x) = (x - 3)^2 on [0, 10]: min 0 at x=3, max 49 at x=10.
        def parabola(pos):
            return (pos["x"] - 3.0) ** 2

        def _cfg(minimize):
            return OptimizerConfig(
                dofs=[DOFSpec(name="x", pv="x", lo=0.0, hi=10.0)],
                objectives=[ObjectiveSpec(name="f", minimize=minimize)],
                detector_pv="det",
                iterations=18,
                n_points=1,
            )

        m1, d1 = _make_devices(["x"], parabola)
        min_points, _ = _run(_cfg(True), m1, d1)
        best_min = _best(min_points, minimize=True)
        assert best_min["primary"] < 1.0
        assert abs(best_min["params"]["x"] - 3.0) < 1.0

        m2, d2 = _make_devices(["x"], parabola)
        max_points, _ = _run(_cfg(False), m2, d2)
        best_max = _best(max_points, minimize=False)
        assert best_max["primary"] > 30.0
        assert best_max["params"]["x"] > 8.0


# ---------------------------------------------------------------------------
# Model surface (predicted mean / uncertainty / acquisition) for the projection
# ---------------------------------------------------------------------------

class TestModelSurface:

    def test_predict_surface_mean_peaks_at_optimum(self):
        cx, cy, sigma, amp = 5.0, 5.0, 1.5, 1000.0

        def gauss(pos):
            return amp * math.exp(
                -0.5 * (((pos["x"] - cx) / sigma) ** 2 + ((pos["y"] - cy) / sigma) ** 2)
            )

        cfg = OptimizerConfig(
            dofs=[
                DOFSpec(name="x", pv="x", lo=0.0, hi=10.0),
                DOFSpec(name="y", pv="y", lo=0.0, hi=10.0),
            ],
            objectives=[ObjectiveSpec(name="intensity", minimize=False)],
            detector_pv="det",
            iterations=20,
            n_points=1,
        )
        motors, detector = _make_devices(["x", "y"], gauss)
        _, agent = _run(cfg, motors, detector)

        s = predict_surface(agent, cfg, "x", "y", grid_n=25)

        # Shapes + finiteness for all three views.
        for key in ("mean", "sem", "acq"):
            assert s[key].shape == (25, 25)
            assert np.isfinite(s[key]).all()

        # The predicted-mean surface should peak near the true optimum (5, 5).
        j, i = np.unravel_index(np.argmax(s["mean"]), s["mean"].shape)
        assert abs(s["xi"][i] - cx) < 2.0
        assert abs(s["yi"][j] - cy) < 2.0

        # Uncertainty varies across the grid; acquisition is the UCB (maximize).
        assert s["sem"].max() > s["sem"].min()
        assert np.allclose(s["acq"], s["mean"] + s["kappa"] * s["sem"])
