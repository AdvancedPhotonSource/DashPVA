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


"""Workflow queue/cache memory estimates and the high-memory confirmation."""

from dashpva.workflow import workflow
from dashpva.workflow.workflow import Workflow


def test_frame_memory_text_scales_with_frame_count():
    one = Workflow._frame_memory_text(1)
    assert one.startswith("Estimated for 1 MP: images")
    assert "with HKL" in one
    assert "1.9 MiB" in one
    assert "24.8 MiB" in Workflow._frame_memory_text(1)
    assert "GiB" in Workflow._frame_memory_text(1000)


def test_frame_memory_rich_text_bolds_sizes():
    text = Workflow._frame_memory_text(1, rich_text=True)
    assert "images <b>1.9 MiB</b>" in text
    assert "with HKL <b>24.8 MiB</b>" in text


def test_small_buffers_start_without_asking(monkeypatch):
    asked = []
    monkeypatch.setattr(workflow.QMessageBox, "warning", lambda *a, **k: asked.append(a))
    assert Workflow._confirm_memory_risk(None, "Start", [("Server queue", 10)])
    assert asked == []


def test_large_buffers_ask_and_respect_no(monkeypatch):
    asked = []

    def _warning(*args, **_kwargs):
        asked.append((args[1], args[2]))
        return workflow.QMessageBox.No

    monkeypatch.setattr(workflow.QMessageBox, "warning", _warning)
    assert not Workflow._confirm_memory_risk(None, "Starting the Analysis Consumer",
                                             [("Server queue", 1000)])
    title, text = asked[0]
    assert title == "High Memory Setting"
    assert "1000 frame(s)" in text
