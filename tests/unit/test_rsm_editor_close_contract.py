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

"""RSM editor close behavior when staged values fail validation."""

from __future__ import annotations

import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox

from dashpva.consumers.ioc_rsm_parameter import (
    _has_pending_rsm_change,
    _review_pending_change,
)
from dashpva.viewer.core.base_window import BaseWindow


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class _Event:
    accepted = None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class _InvalidRSMEditor(BaseWindow):
    def __init__(self):
        super().__init__(viewer_name="Invalid RSM Editor", visible_actions=None)
        self.save_calls = 0

    @staticmethod
    def _pending_change():
        raise ValueError("ENERGY_SOURCE_PV is required")

    def has_unsaved_changes(self):
        return _has_pending_rsm_change(self._pending_change, {})

    def unsaved_changes_rows(self):
        _pending, rows = _review_pending_change(self._pending_change)
        return rows

    def save_changes(self):
        self.save_calls += 1
        try:
            self._pending_change()
        except ValueError:
            return False
        return True


def test_invalid_rsm_save_keeps_close_event_open(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)
    monkeypatch.setattr(
        QMessageBox,
        "standardButton",
        lambda self, clicked: QMessageBox.Save,
    )
    window = _InvalidRSMEditor()
    event = _Event()

    window.closeEvent(event)

    assert event.accepted is False
    assert window.save_calls == 1
    assert window.unsaved_changes_rows()[0][1] == "RSM_CONFIGURATION_INVALID"
