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


"""Status-bar RAM readout reports this process and colours by threshold."""

from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QApplication, QLabel

import dashpva.settings as app_settings
from dashpva.viewer.core.base_window import BaseWindow


@pytest.fixture
def label():
    app = QApplication.instance() or QApplication([])
    widget = QLabel()
    yield widget
    del app


def test_ram_readout_shows_resident_memory(label):
    BaseWindow._update_memory_label(SimpleNamespace(_mem_label=label))
    assert label.text().startswith("RAM: ")
    assert label.text().endswith("%)")
    assert label.property("memoryLevel") == "normal"


@pytest.mark.parametrize("warn, error, level", [(0.0, 1.0, "warning"), (0.0, 0.0, "error")])
def test_ram_readout_colours_by_threshold(label, monkeypatch, warn, error, level):
    monkeypatch.setattr(app_settings, "MEMORY_WARN_FRACTION", warn)
    monkeypatch.setattr(app_settings, "MEMORY_ERROR_FRACTION", error)
    BaseWindow._update_memory_label(SimpleNamespace(_mem_label=label))
    assert label.property("memoryLevel") == level
