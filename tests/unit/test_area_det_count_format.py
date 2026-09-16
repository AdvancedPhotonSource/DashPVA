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

import pytest

from dashpva.viewer.area_det.count_format import format_count


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0"),
        (999, "999"),
        (1000, "1,000"),
        (123456789, "123,456,789"),
        (1234567890, "1,234,567,890"),
        (12.6, "13"),
    ],
)
def test_format_integer_counts(value, expected):
    assert format_count(value) == expected


def test_format_fractional_detector_counts():
    assert format_count(1234567.5, 2) == "1,234,567.50"
    assert format_count(-1234.25, 2) == "-1,234.25"


def test_format_count_uses_scientific_notation_only_above_grouped_limit():
    assert format_count(9_999_999_999) == "9,999,999,999"
    assert format_count(10_000_000_000) == "1.00e+10"


def test_format_count_rejects_negative_precision():
    with pytest.raises(ValueError):
        format_count(1, -1)


def test_roi_table_preserves_fractional_statistics():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    from dashpva.viewer.roi_stats_panel import _format_stat_value

    assert _format_stat_value("Total_RBV", 1080.25) == "1,080.25"
    assert _format_stat_value("MeanValue_RBV", 1080.0) == "1,080.00"
