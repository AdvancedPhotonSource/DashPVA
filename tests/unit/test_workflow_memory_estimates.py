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
    assert Workflow._frame_memory_text(1) == "≈ 2 MB for 1 MP images (26 MB with HKL)"
    assert Workflow._frame_memory_text(100) == "≈ 200 MB for 1 MP images (2.6 GB with HKL)"
    assert "GB" in Workflow._frame_memory_text(1000)


def test_frame_memory_rich_text_bolds_sizes():
    text = Workflow._frame_memory_text(1, rich_text=True)
    assert text == "≈ <b>2 MB</b> for 1 MP images (<b>26 MB</b> with HKL)"


def test_small_buffers_start_without_asking(monkeypatch):
    asked = []
    monkeypatch.setattr(workflow.QMessageBox, "warning", lambda *a, **k: asked.append(a))
    assert Workflow._confirm_memory_risk(None, "Start", [("Server queue", 10, True)])
    assert asked == []


def test_image_only_queue_ignores_hkl_size(monkeypatch):
    asked = []
    monkeypatch.setattr(workflow.QMessageBox, "warning", lambda *a, **k: asked.append(a))
    # 100 image frames = 200 MB: no prompt, although 100 frames with HKL would be 2.6 GB
    assert Workflow._confirm_memory_risk(None, "The Metadata Associator", [("Server queue", 100, False)])
    assert asked == []


def test_large_buffers_ask_and_respect_no(monkeypatch):
    asked = []

    def _warning(*args, **_kwargs):
        asked.append((args[1], args[2]))
        return workflow.QMessageBox.No

    monkeypatch.setattr(workflow.QMessageBox, "warning", _warning)
    assert not Workflow._confirm_memory_risk(None, "The Analysis Consumer",
                                             [("Server queue", 1000, True)])
    title, text = asked[0]
    assert title == "Large Queue Size"
    assert "up to 1000 frames ≈ 26 GB (including HKL data)" in text
    assert "Start anyway?" in text
