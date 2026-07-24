"""Primary-objective combo tracks the optimized objectives.

Regression for a bug where the Primary combo refreshed on add/remove/rename but
NOT when a row's Role combo changed (maximize <-> observe): an observe-only
objective wrongly lingered in the Primary list. The fix routes every change to
the optimized set through ``_ObjectiveTable.objectives_changed``.

Offscreen Qt; skipped when the GUI stack is unavailable.
"""

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5 import QtWidgets  # noqa: E402

from dashpva.viewer.bayesian.bayesian_viewer import BayesianViewer  # noqa: E402
from dashpva.viewer.bayesian.blop_adapter import ObjectiveSpec  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _combo_items(viewer):
    c = viewer._primary_combo
    return [c.itemText(i) for i in range(c.count())]


def test_primary_combo_follows_role_changes(app):
    v = BayesianViewer()
    v._obj_table.add_objective(ObjectiveSpec(name="flux", role="maximize"))

    # Added maximize objective shows up as a Primary candidate.
    assert "flux" in _combo_items(v)

    row = v._obj_table.rowCount() - 1
    role = v._obj_table.cellWidget(row, 4)

    # Flip to observe -> dropped from the Primary list.
    role.setCurrentText("observe")
    assert "flux" not in _combo_items(v)

    # Flip back to an optimized role -> reappears.
    role.setCurrentText("maximize")
    assert "flux" in _combo_items(v)


def test_primary_combo_updates_on_remove(app):
    v = BayesianViewer()
    v._obj_table.add_objective(ObjectiveSpec(name="throwaway", role="maximize"))
    assert "throwaway" in _combo_items(v)

    row = v._obj_table.rowCount() - 1
    v._obj_table.selectRow(row)
    v._obj_table.remove_selected()
    assert "throwaway" not in _combo_items(v)
