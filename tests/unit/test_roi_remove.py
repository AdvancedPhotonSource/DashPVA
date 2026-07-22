"""Remove-manual-ROI target selection (PR #117 review fix).

The 'Remove manual' button used to silently do nothing when the shared ROI
dropdown sat on an EPICS ROI. ``_manual_remove_target`` resolves a manual ROI
reliably: the dropdown selection if it's manual, else the most-recently-added
one. Pure static method, tested directly (no viewer/Qt widgets).
"""

import pytest


def _target(selected, manual):
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    from dashpva.viewer.roi_stats_panel import RoiStatsPanel

    return RoiStatsPanel._manual_remove_target(selected, manual)


_M = [{"n": 1, "key": "Manual1"}, {"n": 3, "key": "Manual3"}]


class TestManualRemoveTarget:

    def test_selected_manual_is_used(self):
        assert _target("Manual1", _M) == "Manual1"

    def test_epics_selection_falls_back_to_most_recent(self):
        assert _target("Stats2", _M) == "Manual3"   # highest slot = most recent

    def test_none_selection_falls_back(self):
        assert _target(None, _M) == "Manual3"

    def test_no_manual_rois_returns_none(self):
        assert _target("Stats1", []) is None
        assert _target(None, []) is None
