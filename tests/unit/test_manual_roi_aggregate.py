# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

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
