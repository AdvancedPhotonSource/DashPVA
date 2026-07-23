"""Tests for dashpva.viewer.bayesian.blop_adapter.

The pure-Python config/extraction logic is tested directly. A blop integration
test (build an agent + run the optimization plan against simulated devices via a
RunEngine) is skipped when blop/bluesky/ophyd are not installed.
"""

import math

import pytest

from dashpva.viewer.bayesian.blop_adapter import (
    DOFSpec,
    ObjectiveSpec,
    OptimizerConfig,
    _make_objective_readables,
    _objective_reading_value,
    extract_scalar,
)
from dashpva.viewer.bayesian.pva_signal import PvaSignal

# ---------------------------------------------------------------------------
# extract_scalar
# ---------------------------------------------------------------------------

class TestExtractScalar:

    def test_explicit_key_suffix_match(self):
        reading = {"det_stats1_total": {"value": 42.0, "timestamp": 0.0}}
        assert extract_scalar(reading, "stats1_total") == 42.0

    def test_auto_detect_preferred_suffix(self):
        reading = {
            "det_stats1_total": {"value": 7.5, "timestamp": 0.0},
            "x_motor": {"value": 1.0, "timestamp": 0.0},
        }
        assert extract_scalar(reading, None) == 7.5

    def test_auto_detect_prefers_total_over_value(self):
        reading = {
            "det_value": {"value": 1.0, "timestamp": 0.0},
            "det_total": {"value": 99.0, "timestamp": 0.0},
        }
        # 'total' precedes 'value' in PREFERRED_SUFFIXES
        assert extract_scalar(reading, None) == 99.0

    def test_first_numeric_fallback(self):
        reading = {"weird_signal": {"value": 3.0, "timestamp": 0.0}}
        assert extract_scalar(reading, None) == 3.0

    def test_none_reading_raises(self):
        with pytest.raises(RuntimeError):
            extract_scalar(None, None)

    def test_missing_explicit_key_raises(self):
        reading = {"det_total": {"value": 1.0, "timestamp": 0.0}}
        with pytest.raises(KeyError):
            extract_scalar(reading, "stats9_missing")

    def test_no_numeric_value_raises(self):
        reading = {"det_str": {"value": "not-a-number", "timestamp": 0.0}}
        with pytest.raises(ValueError):
            extract_scalar(reading, None)


# ---------------------------------------------------------------------------
# PvaSignal struct decomposition (bridge -> Bluesky event-model conformance)
# ---------------------------------------------------------------------------

class _FakePv:
    """Minimal stand-in for a pvaccess PvObject: toDict() + ['value'] access."""

    def __init__(self, d):
        self._d = dict(d)

    def toDict(self):
        return dict(self._d)

    def __getitem__(self, key):
        return self._d[key]


class TestPvaSignalStruct:

    def test_struct_describe_emits_scalar_key_per_field(self):
        sig = PvaSignal("X:ManualROI:Stats", name="X:ManualROI:Stats")
        sig._raw_get = lambda: _FakePv(
            {"frameId": 3, "m1_active": True, "m1_total": 5.0, "m1_comx": 10.0})

        desc = sig.describe()
        assert set(desc) == {
            "X:ManualROI:Stats_frameId",
            "X:ManualROI:Stats_m1_active",
            "X:ManualROI:Stats_m1_total",
            "X:ManualROI:Stats_m1_comx",
        }
        # Every field maps to a valid event-model primitive (never 'object').
        assert desc["X:ManualROI:Stats_m1_total"]["dtype"] == "number"
        assert desc["X:ManualROI:Stats_m1_comx"]["dtype"] == "number"
        assert desc["X:ManualROI:Stats_m1_active"]["dtype"] == "boolean"
        assert desc["X:ManualROI:Stats_frameId"]["dtype"] == "integer"
        assert all(d["dtype"] != "object" for d in desc.values())

    def test_struct_read_keys_match_describe(self):
        sig = PvaSignal("X:ManualROI:Stats", name="X:ManualROI:Stats")
        sig._raw_get = lambda: _FakePv({"m1_total": 5.0, "m1_active": False})
        reading = sig.read()
        assert set(reading) == set(sig.describe())
        assert reading["X:ManualROI:Stats_m1_total"]["value"] == 5.0
        assert reading["X:ManualROI:Stats_m1_active"]["value"] is False

    def test_explicit_field_returns_single_scalar_key(self):
        sig = PvaSignal("X:ManualROI:Stats", name="roi1", field="m1_total")
        sig._raw_get = lambda: _FakePv({"m1_total": 7.0, "m1_comx": 10.0})
        assert sig.get() == 7.0
        assert sig.describe() == {
            "roi1": {"source": "PVA:X:ManualROI:Stats", "dtype": "number", "shape": []}
        }
        assert sig.read()["roi1"]["value"] == 7.0

    def test_plain_scalar_single_key(self):
        sig = PvaSignal("X:Total_RBV", name="X:Total_RBV")
        sig._raw_get = lambda: _FakePv({"value": 42.0})
        assert sig.describe()["X:Total_RBV"]["dtype"] == "number"
        assert sig.read()["X:Total_RBV"]["value"] == 42.0


# ---------------------------------------------------------------------------
# _objective_reading_value + _make_objective_readables
# ---------------------------------------------------------------------------

class _Dev:
    def __init__(self, name):
        self.name = name


class TestObjectiveReadingValue:

    @staticmethod
    def _struct_reading():
        return {
            "X:Stats_m1_total": {"value": 5.0, "timestamp": 0.0},
            "X:Stats_m1_comx": {"value": 10.0, "timestamp": 0.0},
        }

    def test_field_selects_matching_key(self):
        dev = _Dev("X:Stats")
        o = ObjectiveSpec(name="roi1", pv="X:Stats", protocol="pva", field="m1_total")
        assert _objective_reading_value(self._struct_reading(), dev, o) == 5.0

    def test_blank_field_on_struct_raises_listing_fields(self):
        dev = _Dev("X:Stats")
        o = ObjectiveSpec(name="roi", pv="X:Stats", protocol="pva", field="")
        with pytest.raises(ValueError) as ei:
            _objective_reading_value(self._struct_reading(), dev, o)
        msg = str(ei.value)
        assert "m1_total" in msg and "m1_comx" in msg

    def test_plain_scalar_single_key(self):
        dev = _Dev("det:Total_RBV")
        reading = {"det:Total_RBV": {"value": 42.0, "timestamp": 0.0}}
        o = ObjectiveSpec(name="i", pv="det:Total_RBV", protocol="pva", field="")
        assert _objective_reading_value(reading, dev, o) == 42.0


class TestMakeObjectiveReadables:

    def test_struct_objectives_share_one_field_none_signal(self):
        pytest.importorskip("ophyd")
        objs = [
            ObjectiveSpec(name="a", pv="X:Stats", protocol="pva", field="m1_total"),
            ObjectiveSpec(name="b", pv="X:Stats", protocol="pva", field="m1_comx"),
        ]
        reads = _make_objective_readables(objs)
        # One shared readable -> the channel is read exactly once per point.
        assert reads["a"] is reads["b"]
        assert reads["a"].field is None
        assert reads["a"].name == "X:Stats"


class TestMoveToPoint:

    @staticmethod
    def _cfg():
        return OptimizerConfig(
            dofs=[DOFSpec(name="x", pv="x", lo=0.0, hi=10.0),
                  DOFSpec(name="y", pv="y", lo=0.0, hi=10.0)],
            objectives=[ObjectiveSpec(name="i", pv="det")],
            iterations=1, n_points=1,
        )

    def test_moves_and_clamps(self):
        pytest.importorskip("ophyd")
        from dashpva.viewer.bayesian.blop_adapter import move_to_point
        moved, simulated = move_to_point(
            self._cfg(), {"x": 3.0, "y": 999.0}, simulate=True)  # y over hi -> clamped
        assert simulated is True
        assert moved == {"x": 3.0, "y": 10.0}

    def test_absent_dof_is_skipped(self):
        pytest.importorskip("ophyd")
        from dashpva.viewer.bayesian.blop_adapter import move_to_point
        moved, _ = move_to_point(self._cfg(), {"x": 5.0}, simulate=True)  # no "y"
        assert moved == {"x": 5.0}


class _ReadDev:
    def __init__(self, v):
        self._v = v

    def read(self):
        return {"m": {"value": self._v, "timestamp": 0.0}}


class _BadDev:
    def read(self):
        raise RuntimeError("boom")


class TestReadPositions:

    def test_reads_present_skips_absent(self):
        from dashpva.viewer.bayesian.blop_adapter import read_positions
        out = read_positions({"x": _ReadDev(3.0), "y": _ReadDev(4.0)}, ["x", "y", "z"])
        assert out == {"x": 3.0, "y": 4.0}   # z absent from the actuators

    def test_skips_erroring_device(self):
        from dashpva.viewer.bayesian.blop_adapter import read_positions
        out = read_positions({"x": _ReadDev(1.0), "y": _BadDev()}, ["x", "y"])
        assert out == {"x": 1.0}


# ---------------------------------------------------------------------------
# OptimizerConfig.validate
# ---------------------------------------------------------------------------

def _valid_config(**overrides) -> OptimizerConfig:
    cfg = OptimizerConfig(
        dofs=[
            DOFSpec(name="x", pv="6IDA:m1", lo=0.0, hi=5.0),
            DOFSpec(name="y", pv="6IDA:m2", lo=0.0, hi=5.0),
        ],
        objectives=[ObjectiveSpec(name="intensity", pv="6IDD:det:Stats1:Total_RBV")],
        iterations=20,
        n_points=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestConfigValidate:

    def test_valid(self):
        assert _valid_config().validate() is None

    def test_requires_a_dof(self):
        assert "DOF" in _valid_config(dofs=[]).validate()

    def test_disabled_dofs_dont_count(self):
        cfg = _valid_config(
            dofs=[DOFSpec(name="x", pv="m1", lo=0, hi=1, enabled=False)]
        )
        assert "DOF" in cfg.validate()

    def test_duplicate_dof_name(self):
        cfg = _valid_config(
            dofs=[
                DOFSpec(name="x", pv="m1", lo=0, hi=1),
                DOFSpec(name="x", pv="m2", lo=0, hi=1),
            ]
        )
        assert "Duplicate" in cfg.validate()

    def test_missing_pv(self):
        cfg = _valid_config(dofs=[DOFSpec(name="x", pv="", lo=0, hi=1)])
        assert "motor PV" in cfg.validate()

    def test_lo_ge_hi(self):
        cfg = _valid_config(dofs=[DOFSpec(name="x", pv="m1", lo=5, hi=1)])
        assert "low limit" in cfg.validate()

    def test_bad_kind(self):
        cfg = _valid_config(dofs=[DOFSpec(name="x", pv="m1", lo=0, hi=1, kind="double")])
        assert "kind" in cfg.validate()

    def test_missing_objective_pv(self):
        cfg = _valid_config(objectives=[ObjectiveSpec(name="intensity", pv="")])
        assert "read PV" in cfg.validate()

    def test_duplicate_objective(self):
        cfg = _valid_config(
            objectives=[
                ObjectiveSpec(name="i", pv="a"),
                ObjectiveSpec(name="i", pv="b"),
            ]
        )
        assert "Duplicate objective" in cfg.validate()

    def test_bad_iterations(self):
        assert "Iterations" in _valid_config(iterations=0).validate()

    def test_bad_n_points(self):
        assert "Points" in _valid_config(n_points=0).validate()

    def test_simulate_skips_device_requirements(self):
        # Simulation ignores motor/detector PVs, so require_devices=False passes
        # even with empty PVs, while the strict check still flags them.
        cfg = _valid_config(
            dofs=[DOFSpec(name="x", pv="", lo=0.0, hi=1.0)],
            objectives=[ObjectiveSpec(name="intensity", pv="")],
        )
        assert cfg.validate(require_devices=False) is None
        assert cfg.validate() is not None


# ---------------------------------------------------------------------------
# OptimizerConfig helpers + (de)serialization
# ---------------------------------------------------------------------------

class TestConfigSerialization:

    def test_total_points(self):
        assert _valid_config(iterations=10, n_points=3).total_points() == 30

    def test_active_objectives_defaults_when_empty(self):
        cfg = _valid_config(objectives=[])
        active = cfg.active_objectives()
        assert len(active) == 1
        assert isinstance(active[0], ObjectiveSpec)

    def test_roundtrip_preserves_everything(self):
        cfg = _valid_config(
            iterations=42,
            n_points=2,
            acq_kwargs={"name": "demo", "flag": True},
        )
        restored = OptimizerConfig.from_dict(cfg.to_dict())
        assert [d.name for d in restored.dofs] == [d.name for d in cfg.dofs]
        assert [d.lo for d in restored.dofs] == [d.lo for d in cfg.dofs]
        assert [d.hi for d in restored.dofs] == [d.hi for d in cfg.dofs]
        assert restored.iterations == 42
        assert restored.n_points == 2
        assert restored.acq_kwargs == {"name": "demo", "flag": True}
        assert restored.objectives[0].pv == cfg.objectives[0].pv

    def test_3dof_2objective_config_builds(self):
        # Proves the N-D / multi-objective shape is just a config change.
        cfg = OptimizerConfig(
            dofs=[
                DOFSpec(name=f"m{i}", pv=f"pv{i}", lo=0.0, hi=1.0)
                for i in range(3)
            ],
            objectives=[
                ObjectiveSpec(name="flux", pv="det:Stats1:Total_RBV", minimize=False),
                ObjectiveSpec(name="width", pv="det:Stats2:Total_RBV", minimize=True),
            ],
        )
        assert cfg.validate() is None
        assert len(cfg.active_dofs()) == 3
        assert len(cfg.active_objectives()) == 2


# ---------------------------------------------------------------------------
# Viewer thread-safety regression guard
# ---------------------------------------------------------------------------

class TestViewerThreadSafety:

    def test_runengine_created_without_signal_handlers(self):
        # The RunEngine runs in a worker QThread; it MUST be created with
        # context_managers=[] because its default SIGINT handler can only be
        # installed on the main thread ("signal only works in main thread").
        # Regression guard for that fix in ScanWorker.run.
        pytest.importorskip("PyQt5")
        pytest.importorskip("pyqtgraph")
        import inspect

        from dashpva.viewer.bayesian import bayesian_viewer

        src = inspect.getsource(bayesian_viewer.ScanWorker.run)
        assert "context_managers=[]" in src


class TestObjectiveTable:

    def test_add_objective_generates_unique_names(self):
        # Clicking "Add objective" must not create duplicate-named rows (which
        # would fail validation). Default first name is "intensity", then unique.
        pytest.importorskip("PyQt5")
        pytest.importorskip("pyqtgraph")
        from PyQt5 import QtWidgets

        from dashpva.viewer.bayesian.bayesian_viewer import _ObjectiveTable

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        table = _ObjectiveTable()
        table.add_objective()
        table.add_objective()
        table.add_objective()

        names = [o.name for o in table.specs()]
        assert names[0] == "intensity"
        assert len(names) == len(set(names)), f"duplicate objective names: {names}"


# ---------------------------------------------------------------------------
# blop integration (skipped if blop/bluesky/ophyd not available)
# ---------------------------------------------------------------------------

class TestBlopIntegration:

    def test_sim_optimization_runs_and_improves(self):
        pytest.importorskip("blop.ax")
        pytest.importorskip("ophyd")
        bluesky = pytest.importorskip("bluesky")

        from dashpva.viewer.bayesian.blop_adapter import (
            blop_optimize_plan,
            build_agent,
            resolve_devices,
        )

        cfg = OptimizerConfig(
            dofs=[
                DOFSpec(name="x", pv="x", lo=0.0, hi=10.0),
                DOFSpec(name="y", pv="y", lo=0.0, hi=10.0),
            ],
            objectives=[
                ObjectiveSpec(name="intensity", pv="det:Stats1:Total_RBV"),
            ],
            iterations=12,
            n_points=1,
        )

        actuators, readables, simulated = resolve_devices(cfg, simulate=True)
        assert simulated is True
        assert set(actuators) == {"x", "y"}

        agent = build_agent(cfg, actuators)

        points = []
        RE = bluesky.RunEngine()
        RE(
            blop_optimize_plan(
                agent, actuators, readables, cfg,
                on_point=lambda p: points.append(p),
            )
        )

        # Structural: one payload per evaluation, all within bounds, finite.
        assert len(points) == cfg.total_points()
        for p in points:
            assert 0.0 <= p["params"]["x"] <= 10.0
            assert 0.0 <= p["params"]["y"] <= 10.0
            assert math.isfinite(p["primary"])

        # The sim detector peaks (value 1000) at the center (5, 5). The best
        # point found should be meaningfully above a flat/edge response.
        best = max(points, key=lambda p: p["primary"])
        assert best["primary"] > 1.0
