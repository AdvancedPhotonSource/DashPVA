"""Manual-ROI scatter aggregate — the scalarized Bayesian-objective helper.

``DiffractionImageWindow._scatter_aggregate`` collapses the active manual ROIs'
per-ROI totals/means into single ``scatter_total`` / ``scatter_mean`` targets so a
BO objective can minimize scatter across all ROIs at once. It's a pure static
method, tested directly (no viewer/Qt widgets).
"""

import pytest


def _agg(totals, means):
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    from dashpva.viewer.area_det.area_det_viewer import DiffractionImageWindow

    return DiffractionImageWindow._scatter_aggregate(totals, means)


class TestScatterAggregate:

    def test_empty_is_zero(self):
        out = _agg([], [])
        assert out == {"scatter_total": 0.0, "scatter_mean": 0.0, "n_active": 0}

    def test_sum_and_mean(self):
        out = _agg([10, 20, 30], [1, 2, 3])
        assert out["scatter_total"] == 60.0
        assert out["scatter_mean"] == 2.0   # area-normalized: mean of per-pixel means
        assert out["n_active"] == 3

    def test_mean_normalization_ignores_roi_size(self):
        # A huge-total ROI and a small one with equal per-pixel means average to
        # that shared mean — big ROI does not dominate scatter_mean.
        out = _agg([1_000_000, 5], [4.0, 4.0])
        assert out["scatter_mean"] == 4.0
        assert out["n_active"] == 2

    def test_returns_python_floats(self):
        out = _agg([2, 4], [2, 4])
        assert isinstance(out["scatter_total"], float)
        assert isinstance(out["scatter_mean"], float)
        assert isinstance(out["n_active"], int)
