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

"""Workflow keeps each analysis processor paired with its real class."""

from pathlib import Path

import pytest

pytest.importorskip("PyQt5.QtWidgets")

import dashpva.settings as app_settings  # noqa: E402
from dashpva.workflow.workflow import Workflow  # noqa: E402


class _TextWidget:
    def __init__(self, value=""):
        self._value = value

    def text(self):
        return self._value

    def setText(self, value):
        self._value = value


class _SpinWidget:
    def __init__(self, value):
        self._value = value
        self._enabled = True
        self.tooltip = ""

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def isEnabled(self):
        return self._enabled

    def setEnabled(self, enabled):
        self._enabled = enabled

    def setToolTip(self, text):
        self.tooltip = text


class _AnalysisHarness:
    _on_analysis_processor_file_changed = (
        Workflow._on_analysis_processor_file_changed
    )
    _remember_analysis_consumer_count = Workflow._remember_analysis_consumer_count
    _processor_file_path = staticmethod(Workflow._processor_file_path)

    def __init__(self):
        self.lineEditProcessorClassAnalysis = _TextWidget("HpcRsmProcessor")
        self.spinBoxNConsumersAnalysis = _SpinWidget(5)
        self._analysis_non_grid_consumers = 5


def test_grid_processor_selection_fixes_class_and_consumer_count():
    workflow = _AnalysisHarness()

    workflow._on_analysis_processor_file_changed(
        "src/dashpva/consumers/hpc/analysis/hpc_rsm_grid_consumer.py"
    )

    assert workflow.lineEditProcessorClassAnalysis.text() == "HpcRsmGridProcessor"
    assert workflow.spinBoxNConsumersAnalysis.value() == 1
    assert not workflow.spinBoxNConsumersAnalysis.isEnabled()
    assert "exactly one" in workflow.spinBoxNConsumersAnalysis.tooltip


@pytest.mark.parametrize(
    ("filename", "processor_class"),
    (
        ("hpc_rsm_consumer.py", "HpcRsmProcessor"),
        ("hpc_spontaneous_analysis_consumer.py", "HpcAnalysisProcessor"),
        ("hpc_vectorized_analysis_consumer.py", "HpcAnalysisProcessor"),
    ),
)
def test_non_grid_processor_files_select_their_real_class(
    filename, processor_class
):
    workflow = _AnalysisHarness()
    workflow._on_analysis_processor_file_changed(filename)

    assert workflow.lineEditProcessorClassAnalysis.text() == processor_class
    assert workflow.spinBoxNConsumersAnalysis.isEnabled()


def test_switching_from_grid_restores_the_non_grid_consumer_count():
    workflow = _AnalysisHarness()
    workflow._on_analysis_processor_file_changed("hpc_rsm_grid_consumer.py")
    workflow.spinBoxNConsumersAnalysis.setValue(9)
    workflow._on_analysis_processor_file_changed("hpc_rsm_grid_consumer.py")

    assert workflow.spinBoxNConsumersAnalysis.value() == 1

    workflow._on_analysis_processor_file_changed("hpc_rsm_consumer.py")
    assert workflow.lineEditProcessorClassAnalysis.text() == "HpcRsmProcessor"
    assert workflow.spinBoxNConsumersAnalysis.value() == 5
    assert workflow.spinBoxNConsumersAnalysis.isEnabled()


def test_processor_file_is_resolved_independently_of_launch_directory():
    relative = Path(
        "src/dashpva/consumers/hpc/analysis/hpc_rsm_grid_consumer.py"
    )
    resolved = _AnalysisHarness._processor_file_path(relative)

    assert Path(resolved) == (app_settings.PROJECT_ROOT / relative).resolve()
