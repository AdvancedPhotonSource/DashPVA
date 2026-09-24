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


"""Machine-local window layout persistence.

A window reopens at the size, position and dock arrangement it was closed at.
State is written to Qt's per-user store, scoped per viewer type, so these tests
redirect that store into tmp_path rather than touching the developer's own.
"""
from __future__ import annotations

import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication, QDockWidget

from dashpva.viewer.core.base_window import BaseWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    """Point QSettings at tmp_path so a test can never read or write the real store."""
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path))
    yield


class _Docked(BaseWindow):
    """A window with one dock, so saveState has something to record."""

    def __init__(self):
        super().__init__(viewer_name='LayoutTest')
        self.dock = QDockWidget("d", self)
        self.dock.setObjectName("d")
        self.addDockWidget(0x1, self.dock)   # Qt.LeftDockWidgetArea


def test_close_saves_geometry_and_dock_state(qapp):
    w = _Docked()
    w.resize(640, 480)
    w.close()

    s = QSettings("DashPVA", "_Docked")
    assert sorted(s.allKeys()) == ["dock_state", "geometry"]


def test_a_window_reopens_at_the_size_it_was_closed_at(qapp):
    w = _Docked()
    w.resize(645, 485)
    w.move(90, 110)
    w.close()

    reopened = _Docked()
    reopened.restore_layout()
    assert (reopened.width(), reopened.height()) == (645, 485)
    assert (reopened.x(), reopened.y()) == (90, 110)


def test_a_dock_layout_saved_under_another_version_is_not_restored(qapp):
    """Qt drops a mismatched state, which is what makes a version bump retire it."""
    w = _Docked()
    w.dock.setFloating(True)
    w.close()

    reopened = _Docked()
    reopened.dock_state_version = _Docked.dock_state_version + 1
    state = QSettings("DashPVA", "_Docked").value("dock_state")
    assert state, "precondition: a layout was saved"
    assert reopened.restoreState(state, reopened.dock_state_version) is False


def test_the_dock_state_version_is_per_viewer():
    """One viewer raising its version must not retire another viewer's layout."""
    class _Other(BaseWindow):
        dock_state_version = BaseWindow.dock_state_version + 5

    assert _Other.dock_state_version != BaseWindow.dock_state_version
    assert _Docked.dock_state_version == BaseWindow.dock_state_version


def test_opting_out_writes_nothing(qapp):
    class _NoPersist(_Docked):
        persist_state = False

    w = _NoPersist()
    w.resize(600, 400)
    w.close()

    assert QSettings("DashPVA", "_NoPersist").allKeys() == []
