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

"""BaseWindow restores geometry, docks and inputs itself; subclasses call nothing.

Regression for the pattern where every subclass had to trigger its own restore and
one of them (the area-detector viewer) simply never did, so saved values were written
on close and ignored on open. The restore is deferred to the event loop, so these
tests pump it explicitly.
"""

from __future__ import annotations

import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QLineEdit

from dashpva.viewer.core.base_window import BaseWindow


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point every window's QSettings at a temp ini so the real store is untouched."""
    path = str(tmp_path / "state.ini")
    monkeypatch.setattr(BaseWindow, "_qsettings",
                        lambda _self: QSettings(path, QSettings.IniFormat))
    return path


class _Window(BaseWindow):
    """Minimal BaseWindow with one named input."""

    def __init__(self):
        super().__init__(viewer_name="Restore Test", visible_actions=None)
        self.field = QLineEdit(self)
        self.field.setObjectName("field")
        self.setCentralWidget(self.field)
        self.restored_order = []

    def restore_inputs(self):
        super().restore_inputs()
        self.restored_order.append("inputs")

    def on_session_restored(self):
        self.restored_order.append("hook")


class _ProfileWindow(_Window):
    """Fields come from elsewhere, so inputs must not be re-applied."""

    restore_inputs_on_start = False


class _OptedOut(_Window):
    persist_state = False


def _build(cls, qapp):
    """Construct a window and let the deferred restore run."""
    window = cls()
    qapp.processEvents()
    return window


def test_inputs_restore_without_the_subclass_asking(qapp, store):
    first = _build(_Window, qapp)
    first.field.setText("saved-value")
    first.save_layout()

    second = _build(_Window, qapp)
    assert second.field.text() == "saved-value"


def test_restore_inputs_on_start_false_leaves_inputs_alone(qapp, store):
    first = _build(_Window, qapp)
    first.field.setText("saved-value")
    first.save_layout()

    second = _build(_ProfileWindow, qapp)
    assert second.field.text() == ""


def test_persist_state_false_restores_nothing(qapp, store):
    first = _build(_Window, qapp)
    first.field.setText("saved-value")
    first.save_layout()

    second = _build(_OptedOut, qapp)
    assert second.field.text() == ""
    assert second.restored_order == []


def test_hook_runs_after_inputs_are_applied(qapp, store):
    first = _build(_Window, qapp)
    first.field.setText("saved-value")
    first.save_layout()

    second = _build(_Window, qapp)
    assert second.restored_order == ["inputs", "hook"]


class _LevelsWindow(_Window):
    """Stands in for the area-detector viewer's autoscale/limits pair.

    Mirrors its restore hazard: __init__ leaves autoscale on, the input walk
    applies spin boxes before check boxes, and the limit handler is passive
    while autoscale is on.
    """

    def __init__(self):
        super().__init__()
        self.levels = None
        self.chk_autoscale = QCheckBox(self)
        self.chk_autoscale.setObjectName("chk_autoscale")
        self.limit = QDoubleSpinBox(self)
        self.limit.setObjectName("limit")
        self.limit.setRange(-1e6, 1e6)
        self.limit.valueChanged.connect(self._limit_changed)
        self.chk_autoscale.setChecked(True)

    def _limit_changed(self):
        if not self.chk_autoscale.isChecked():
            self.levels = self.limit.value()

    def on_session_restored(self):
        super().on_session_restored()
        if not self.chk_autoscale.isChecked():
            self._limit_changed()


def test_restored_limits_take_effect_when_autoscale_was_saved_off(qapp, store):
    first = _build(_LevelsWindow, qapp)
    first.chk_autoscale.setChecked(False)
    first.limit.setValue(1234.0)
    first.save_layout()

    second = _build(_LevelsWindow, qapp)
    assert second.limit.value() == 1234.0
    assert second.chk_autoscale.isChecked() is False
    # Without the hook the spin box shows 1234 but nothing ever applied it.
    assert second.levels == 1234.0
