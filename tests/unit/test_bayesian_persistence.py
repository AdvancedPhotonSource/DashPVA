"""Persistence tests for the Bayesian viewer.

Two paths:
- **Local fallback** (no central profile): the optimizer config + Bluesky env round-
  trip through an isolated QSettings store; Simulate always reopens OFF.
- **Profile mode**: when a profile is active, the viewer follows it — the setup
  dropdown lists the profile's named ``[BAYESIAN]`` setups and Load switches between
  them.

Both use offscreen Qt and never touch the user's real settings. Skipped when the GUI
stack is unavailable.
"""

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from dashpva.viewer.bayesian import profile_store  # noqa: E402
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


def _cfg(iterations, n_points=1):
    return OptimizerConfig(
        dofs=[
            DOFSpec(name="th", pv="6IDB:th", lo=-1.0, hi=1.0),
            DOFSpec(name="chi", pv="6IDB:chi", lo=0.0, hi=10.0, kind="int"),
        ],
        objectives=[ObjectiveSpec(name="flux", pv="6IDB:det:Stats1:Total_RBV")],
        iterations=iterations,
        n_points=n_points,
    )


def test_local_named_setup_round_trip_and_simulate_resets(app, tmp_path, monkeypatch):
    # Force "no central profile" so the viewer uses local QSettings setups.
    monkeypatch.setattr(profile_store, "active_source", lambda: (None, ""))

    v1 = BayesianViewer()
    v1._settings = _temp_settings(tmp_path)
    v1._apply_config(_cfg(42, n_points=3))
    v1._bluesky_env.setText("/opt/conda/envs/6idb-bits")
    v1._simulate.setChecked(True)                         # must NOT survive a reopen
    assert v1._store_save("mysetup", v1._gather_config())  # explicit save (named)
    v1._settings.setValue(v1._last_setup_key(), "mysetup")
    v1._save_env()
    v1._settings.sync()

    v2 = BayesianViewer()
    v2._settings = _temp_settings(tmp_path)
    v2._load_initial()

    assert v2._bluesky_env.text() == "/opt/conda/envs/6idb-bits"
    assert v2._setup_combo.currentText() == "mysetup"
    assert v2._current_setup == "mysetup"
    cfg = v2._gather_config()
    assert cfg.iterations == 42
    assert cfg.n_points == 3
    assert [d.name for d in cfg.dofs] == ["th", "chi"]
    assert [d.pv for d in cfg.dofs] == ["6IDB:th", "6IDB:chi"]
    assert cfg.dofs[1].kind == "int"
    assert cfg.objectives[0].pv == "6IDB:det:Stats1:Total_RBV"
    assert v2._simulate.isChecked() is False


def test_profile_mode_lists_and_loads_named_setups(app, tmp_path, monkeypatch):
    pytest.importorskip("toml")
    from dashpva.utils.config.source import ConfigSource

    # A profile (temp TOML) pre-loaded with two named Bayesian setups.
    src = ConfigSource(str(tmp_path / "profile.toml"))
    profile_store.save_setup(src, "beam_collimation", _cfg(15))
    profile_store.save_setup(src, "crl_focusing", _cfg(25))
    monkeypatch.setattr(profile_store, "active_source", lambda: (src, "myprofile"))

    v = BayesianViewer()
    v._settings = _temp_settings(tmp_path)
    v._load_initial()

    items = [v._setup_combo.itemText(i) for i in range(v._setup_combo.count())]
    assert items == ["beam_collimation", "crl_focusing"]
    assert v._profile_label.text() == "myprofile"
    # No last-used recorded yet -> first setup loads.
    assert v._current_setup == "beam_collimation"
    assert v._gather_config().iterations == 15

    # Selecting the other setup loads it.
    v._on_setup_selected("crl_focusing")
    assert v._current_setup == "crl_focusing"
    assert v._gather_config().iterations == 25


def test_local_mode_supports_named_setups(app, tmp_path, monkeypatch):
    # Regression: in local mode (no profile) the user must still be able to create
    # and name multiple setups — not be stuck on a single "(local)" slot.
    monkeypatch.setattr(profile_store, "active_source", lambda: (None, ""))

    v = BayesianViewer()
    v._settings = _temp_settings(tmp_path)
    v._load_initial()

    # All setup controls are enabled in local mode.
    assert v._btn_save_as.isEnabled()
    assert v._btn_delete.isEnabled()

    # Create two named local setups (what the Save As button drives).
    v._apply_config(_cfg(11))
    assert v._store_save("beam_collimation", v._gather_config())
    v._apply_config(_cfg(22))
    assert v._store_save("crl_focusing", v._gather_config())
    assert v._store_list() == ["beam_collimation", "crl_focusing"]
    assert v._store_load("crl_focusing").iterations == 22

    # Delete one; the other survives.
    assert v._store_delete("beam_collimation")
    assert v._store_list() == ["crl_focusing"]
