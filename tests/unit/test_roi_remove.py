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
