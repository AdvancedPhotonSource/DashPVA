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

"""Save/Discard/Cancel and editable review behavior for staged windows."""

from __future__ import annotations

import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox

from dashpva.gui.change_review_dialog import ChangeReviewDialog
from dashpva.viewer.core.base_window import BaseWindow


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class _Event:
    def __init__(self):
        self.accepted = None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class _Editor(BaseWindow):
    def __init__(self, *, dirty=True, save_succeeds=True):
        super().__init__(viewer_name="Test Editor", visible_actions=None)
        self.dirty = dirty
        self.save_succeeds = save_succeeds
        self.save_calls = 0

    def has_unsaved_changes(self):
        return self.dirty

    def save_changes(self):
        self.save_calls += 1
        if self.save_succeeds:
            self.dirty = False
        return self.save_succeeds


def _answer(monkeypatch, button):
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)
    monkeypatch.setattr(QMessageBox, "standardButton", lambda self, clicked: button)


def test_clean_close_never_prompts(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "exec_",
        lambda self: pytest.fail("clean close must not prompt"),
    )
    event = _Event()

    _Editor(dirty=False).closeEvent(event)

    assert event.accepted is True


@pytest.mark.parametrize(
    ("answer", "save_succeeds", "accepted", "save_calls"),
    [
        (QMessageBox.Save, True, True, 1),
        (QMessageBox.Save, False, False, 1),
        (QMessageBox.Discard, True, True, 0),
        (QMessageBox.Cancel, True, False, 0),
    ],
)
def test_dirty_close_outcomes(
    qapp, monkeypatch, answer, save_succeeds, accepted, save_calls
):
    _answer(monkeypatch, answer)
    window = _Editor(save_succeeds=save_succeeds)
    event = _Event()

    window.closeEvent(event)

    assert event.accepted is accepted
    assert window.save_calls == save_calls


def test_removal_row_can_be_dropped_from_review(qapp):
    dialog = ChangeReviewDialog(
        None,
        [("remove", "DETECTOR_SETUP.SIZE", "[10, 10]", "")],
    )
    button = dialog.table.cellWidget(0, 3)

    assert button is not None
    button.click()
    kept, dropped = dialog.decisions()
    assert kept == []
    assert dropped == [
        ("remove", "DETECTOR_SETUP.SIZE", "[10, 10]", "")
    ]
