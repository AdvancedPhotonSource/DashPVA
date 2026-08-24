# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
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
