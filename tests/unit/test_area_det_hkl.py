"""Focused regressions for Area Detector live HKL setup."""

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from dashpva.viewer.area_det import area_det_viewer as mod  # noqa: E402

VIEWER = mod.DiffractionImageWindow


def _code(obj) -> str:
    return "\n".join(
        line for line in inspect.getsource(obj).splitlines()
        if not line.lstrip().startswith("#")
    )


def test_hkl_setup_does_not_request_removed_motor_name_fields():
    setup = _code(VIEWER.hkl_setup)
    assert "sample_circle_names" not in setup
    assert "det_circle_names" not in setup


def test_nan_motor_position_fails_setup_with_a_logged_error(monkeypatch, capsys):
    monkeypatch.setattr(
        mod, "axis_field_channels",
        lambda config, kind, field=None, **kw: [f"{kind}:{field}"],
    )
    errors = []
    window = SimpleNamespace(
        hkl_config={},
        stop_hkl=SimpleNamespace(isChecked=lambda: False),
        hkl_data={"sample:DIRECTION_AXIS": "z-", "sample:POSITION": float("nan")},
        logger=SimpleNamespace(error=errors.append),
        rsm_geometry_ready=True,
        _rsm_geometry_cache=object(),
        _rsm_geometry_cache_key=object(),
    )

    VIEWER.hkl_setup(window)

    reported = "".join(errors) + capsys.readouterr().out
    assert window.rsm_geometry_ready is False
    assert "sample:POSITION" in reported and "non-finite" in reported
