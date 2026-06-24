"""Tests for the Bayesian ``profile_store`` (named setups in the central profile).

Uses a temporary ``.toml`` file as the ``ConfigSource`` locator, so it exercises the
real TOML backend without a database. Verifies create / list / load / save
(overwrite) / delete, that sibling setups are preserved, and that unrelated profile
sections (e.g. ``[ROI]``) are not clobbered by a Bayesian write.
"""

import pytest

toml = pytest.importorskip("toml")

from dashpva.utils.config.source import ConfigSource  # noqa: E402
from dashpva.viewer.bayesian import profile_store  # noqa: E402
from dashpva.viewer.bayesian.blop_adapter import (  # noqa: E402
    DOFSpec,
    ObjectiveSpec,
    OptimizerConfig,
)


def _cfg(dof_name, lo, hi, iterations):
    return OptimizerConfig(
        dofs=[DOFSpec(name=dof_name, pv=f"IOC:{dof_name}", lo=lo, hi=hi)],
        objectives=[ObjectiveSpec(name="flux", pv="IOC:Stats1:Total_RBV")],
        iterations=iterations,
    )


def _src(tmp_path):
    return ConfigSource(str(tmp_path / "profile.toml"))


def test_named_setups_roundtrip_and_isolation(tmp_path):
    src = _src(tmp_path)
    assert profile_store.list_setups(src) == []
    assert profile_store.load_setup(src, "missing") is None

    assert profile_store.save_setup(src, "beam_collimation", _cfg("th", -1.0, 1.0, 30))
    assert profile_store.save_setup(src, "crl_focusing", _cfg("m6", -1.5, 1.5, 20))
    assert profile_store.list_setups(src) == ["beam_collimation", "crl_focusing"]

    bc = profile_store.load_setup(src, "beam_collimation")
    assert bc.iterations == 30
    assert [d.name for d in bc.dofs] == ["th"]
    assert bc.dofs[0].pv == "IOC:th"
    assert bc.objectives[0].pv == "IOC:Stats1:Total_RBV"
    crl = profile_store.load_setup(src, "crl_focusing")
    assert crl.dofs[0].name == "m6"
    assert crl.dofs[0].hi == 1.5

    # Overwrite one; the sibling survives.
    assert profile_store.save_setup(src, "beam_collimation", _cfg("th", -2.0, 2.0, 42))
    assert profile_store.list_setups(src) == ["beam_collimation", "crl_focusing"]
    assert profile_store.load_setup(src, "beam_collimation").iterations == 42
    assert profile_store.load_setup(src, "crl_focusing").iterations == 20

    # Delete one; the other remains; deleting again is a no-op.
    assert profile_store.delete_setup(src, "crl_focusing")
    assert profile_store.list_setups(src) == ["beam_collimation"]
    assert profile_store.delete_setup(src, "crl_focusing") is False


def test_bayesian_write_preserves_other_sections(tmp_path):
    path = tmp_path / "profile.toml"
    path.write_text(
        toml.dumps(
            {
                "DETECTOR_PREFIX": "13SIM1",
                "ROI": {"ROI1": {"MIN_X": "13SIM1:ROI1:MinX"}},
            }
        )
    )
    src = ConfigSource(str(path))
    assert profile_store.save_setup(src, "s1", _cfg("x", 0.0, 1.0, 10))

    data = toml.loads(path.read_text())
    assert data["DETECTOR_PREFIX"] == "13SIM1"
    assert data["ROI"]["ROI1"]["MIN_X"] == "13SIM1:ROI1:MinX"
    assert "s1" in data["BAYESIAN"]


def test_no_source_is_safe():
    # active_source() returns None when nothing is resolvable; helpers no-op safely.
    assert profile_store.list_setups(None) == []
    assert profile_store.load_setup(None, "x") is None
    assert profile_store.save_setup(None, "x", _cfg("x", 0.0, 1.0, 5)) is False
    assert profile_store.delete_setup(None, "x") is False
