"""Persistence test for the Bayesian viewer's QSettings round-trip.

The viewer remembers its setup between launches via ``QSettings``. The optimizer
config (DOFs, objectives, detector PV, run controls) round-trips through
``OptimizerConfig``; the Bluesky conda-env path is persisted under its own key.
The Simulate checkbox is deliberately NOT persisted -- it always reopens OFF so a
viewer never silently restarts in simulation mode at the beamline.

This drives ``_save_config()`` / ``_load_config()`` directly against an isolated
temp settings store, so the real user settings are never touched. Skipped when the
GUI stack is unavailable.
"""

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from dashpva.viewer.bayesian.bayesian_viewer import BayesianViewer  # noqa: E402
from dashpva.viewer.bayesian.blop_adapter import (  # noqa: E402
    DOFSpec,
    ObjectiveSpec,
    OptimizerConfig,
)


@pytest.fixture(scope="module")
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


def _temp_settings(tmp_path):
    """An isolated INI-backed settings store (never the user's real settings)."""
    return QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)


def test_bluesky_env_persists_and_simulate_resets(app, tmp_path):
    # --- first viewer: configure a setup and save it ----------------------
    v1 = BayesianViewer()
    v1._settings = _temp_settings(tmp_path)
    v1._apply_config(
        OptimizerConfig(
            dofs=[
                DOFSpec(name="th", pv="6IDB:th", lo=-1.0, hi=1.0),
                DOFSpec(name="chi", pv="6IDB:chi", lo=0.0, hi=10.0, kind="int"),
            ],
            objectives=[ObjectiveSpec(name="flux", pv="6IDB:det:Stats1:Total_RBV")],
            iterations=42,
            n_points=3,
        )
    )
    v1._bluesky_env.setText("/opt/conda/envs/6idb-bits")
    v1._simulate.setChecked(True)  # must NOT survive a reopen
    v1._save_config()
    v1._settings.sync()  # flush to the INI file before the second viewer reads it

    # --- second viewer: load from the same store --------------------------
    v2 = BayesianViewer()
    v2._settings = _temp_settings(tmp_path)
    v2._load_config()

    # Bluesky conda-env path is restored.
    assert v2._bluesky_env.text() == "/opt/conda/envs/6idb-bits"

    # The optimizer config (DOFs / objective / detector / run controls) round-trips.
    cfg = v2._gather_config()
    assert cfg.iterations == 42
    assert cfg.n_points == 3
    assert [d.name for d in cfg.dofs] == ["th", "chi"]
    assert [d.pv for d in cfg.dofs] == ["6IDB:th", "6IDB:chi"]
    assert cfg.dofs[1].kind == "int"
    assert cfg.objectives[0].name == "flux"
    assert cfg.objectives[0].pv == "6IDB:det:Stats1:Total_RBV"

    # Simulate is never persisted -- it always reopens OFF.
    assert v2._simulate.isChecked() is False
