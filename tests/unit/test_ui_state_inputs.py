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


"""Named-input persistence built into BaseWindow.

Inputs are remembered per window type in Qt's per-user store, so these tests
redirect that store into tmp_path rather than touching the developer's own.
"""
from __future__ import annotations

import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QSpinBox,
)

from dashpva.viewer.core.base_window import BaseWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path))
    yield


class _Form(BaseWindow):
    """A BaseWindow with a few named inputs."""

    def __init__(self):
        super().__init__(viewer_name="Form")
        self.edit = QLineEdit(self)
        self.edit.setObjectName("edit")
        self.count = QSpinBox(self)
        self.count.setObjectName("count")
        self.flag = QCheckBox(self)
        self.flag.setObjectName("flag")
        self.choice = QComboBox(self)
        self.choice.setObjectName("choice")
        self.choice.addItems(["alpha", "beta"])


def test_inputs_round_trip(qapp):
    """Named inputs are saved on one BaseWindow and re-applied on the next."""
    w = _Form()
    w.edit.setText("prefix:1")
    w.count.setValue(7)
    w.flag.setChecked(True)
    w.choice.setCurrentIndex(1)
    w.save_inputs()

    reopened = _Form()
    reopened.restore_inputs()
    assert reopened.edit.text() == "prefix:1"
    assert reopened.count.value() == 7
    assert reopened.flag.isChecked()
    assert reopened.choice.currentText() == "beta"


def test_an_unnamed_or_qt_internal_widget_is_not_persisted(qapp):
    w = _Form()
    anon = QLineEdit(w)          # no objectName
    anon.setText("dropped")
    w.save_inputs()

    assert "" not in w.session_inputs()
    assert not any(k.startswith("qt_") for k in w.session_inputs())


def test_a_skipped_input_is_never_written(qapp):
    class _Guarded(_Form):
        persist_input_skip = {"edit"}

    w = _Guarded()
    w.edit.setText("secret")
    w.count.setValue(3)

    values = w.session_inputs()
    assert "edit" not in values
    assert values["count"] == 3


def test_a_combo_choice_that_no_longer_exists_is_not_injected(qapp):
    """Items are populated at runtime; a stale choice must not become a new item."""
    w = _Form()
    w.choice.addItem("gamma")
    w.choice.setCurrentIndex(2)
    w.save_inputs()

    reopened = _Form()            # only alpha/beta this time
    reopened.restore_inputs()
    assert reopened.choice.count() == 2
    assert reopened.choice.currentText() == "alpha"


def test_opting_out_writes_nothing(qapp):
    class _NoPersist(_Form):
        persist_state = False

    w = _NoPersist()
    w.edit.setText("x")
    w.save_inputs()
    assert QSettings("DashPVA", "_NoPersist").value("inputs") is None


def test_a_corrupt_stored_blob_is_ignored(qapp):
    w = _Form()
    QSettings("DashPVA", "_Form").setValue("inputs", "not json{")
    w.restore_inputs()            # must not raise
    assert w.edit.text() == ""


class _Window(BaseWindow):
    def __init__(self):
        super().__init__(viewer_name='InputsTest')
        self.edit = QLineEdit(self)
        self.edit.setObjectName("edit")


def test_closing_a_base_window_saves_its_inputs_too(qapp):
    """save_layout is the shared seam -- inputs ride along with geometry."""
    w = _Window()
    w.edit.setText("kept")
    w.close()

    keys = QSettings("DashPVA", "_Window").allKeys()
    assert "inputs" in keys and "geometry" in keys


def test_a_base_window_restores_inputs_on_start(qapp):
    """_restore_on_start is the other seam; without it inputs save but never load."""
    w = _Window()
    w.edit.setText("kept")
    w.close()

    reopened = _Window()
    reopened._restore_on_start()
    assert reopened.edit.text() == "kept"
