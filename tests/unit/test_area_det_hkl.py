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


def test_hkl_errors_are_logged_once_and_recovery_is_logged():
    messages = []
    window = SimpleNamespace(
        _hkl_log_errors={},
        logger=SimpleNamespace(
            error=lambda message: messages.append(("error", message)),
            info=lambda message: messages.append(("info", message)),
        ),
    )

    VIEWER._log_hkl_error(window, "setup", ValueError("missing Energy:Value"))
    VIEWER._log_hkl_error(window, "setup", ValueError("missing Energy:Value"))
    VIEWER._clear_hkl_error(window, "setup")

    assert messages == [
        ("error", "[HKL] setup failed: missing Energy:Value"),
        ("info", "[HKL] setup recovered"),
    ]


def test_hkl_setup_does_not_request_removed_motor_name_fields():
    setup = _code(VIEWER.hkl_setup)
    assert "sample_circle_names" not in setup
    assert "det_circle_names" not in setup


def test_successful_update_and_calculation_clear_previous_errors():
    update_source = _code(VIEWER.handle_hkl_data_update)
    calculation_source = _code(VIEWER.create_rsm)

    assert "_clear_hkl_error('update')" in update_source
    assert "_clear_hkl_error('RSM calculation')" in calculation_source


def test_last_missing_pv_clears_monitor_initialization_error():
    cleared = []
    emitted = []
    window = SimpleNamespace(
        hkl_data={'energy': 12.0, 'ub': None},
        _hkl_dynamic_channels=set(),
        _rsm_geometry_cache=object(),
        _rsm_geometry_cache_key=object(),
        qx=object(),
        qy=object(),
        qz=object(),
        _clear_hkl_error=cleared.append,
        hkl_data_updated=SimpleNamespace(emit=emitted.append),
    )

    VIEWER.hkl_ca_callback(window, 'ub', [1.0] * 9)

    assert cleared == ['monitor initialization']
    assert emitted == [True]
