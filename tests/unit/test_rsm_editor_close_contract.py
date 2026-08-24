# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
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
