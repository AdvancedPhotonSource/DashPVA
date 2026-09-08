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


# --- Apply/close parity against live IOC records ----------------------------
# The close gate and Apply must agree on what is pending, including changes
# that came only from the IOC. A divergent edit stays a conflict and writes
# nothing -- these lock in the behaviour end to end on the real window.


class _RecordingSource:
    """Snapshot source that records whether a save was ever attempted."""

    def __init__(self, raw):
        self.raw = raw
        self.save_attempts = 0

    def load_snapshot(self):
        from dashpva.utils.config.revision import mapping_revision

        return self.raw, mapping_revision(self.raw)

    def replace_if_revision(self, full_config, revision):
        self.save_attempts += 1
        raise AssertionError("save must not be attempted while conflicted")


def _live_editor():
    """Build the real SimulatorWindow over a recording source."""
    import threading

    from dashpva.consumers.ioc_rsm_parameter import _build_gui_classes
    from dashpva.utils.rsm_parameter_config import (
        RSMParameterEditSession,
        default_parameter_mapping,
        profile_from_raw,
    )

    raw = {"IOC_PREFIX": "sim:", "IOC_RSM_PARAMETER": default_parameter_mapping()}
    source = _RecordingSource(raw)
    session = RSMParameterEditSession(source)
    profile = profile_from_raw(raw)
    _, _, simulator_window = _build_gui_classes()
    window = simulator_window(
        session, profile, lambda snapshot: None, {}, threading.Lock()
    )
    return window, source, profile


def _direction_record(profile):
    return f"{profile.prefix}PrimaryBeamDirection:AxisNumber1"


def test_ioc_only_change_makes_the_close_gate_dirty(qapp):
    """A caput nobody edited locally still counts as a pending change."""
    window, _source, profile = _live_editor()
    try:
        assert window.has_unsaved_changes() is False
        baseline = window._normalized_baseline["PRIMARY_BEAM_DIRECTION"][0]
        with window.pv_lock:
            window.pv_values[_direction_record(profile)] = baseline + 1

        pending = window._pending_change()
        assert pending["adopted"], "live change should be adopted"
        assert not pending["conflicts"]
        # Apply and close read the same calculation, so both must see it.
        assert window.has_unsaved_changes() is True
        assert window.unsaved_changes_rows()
    finally:
        window._stop_worker()


def test_conflicting_edit_writes_nothing(qapp, monkeypatch):
    """Form and IOC diverging on one record must save nothing at all."""
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: QMessageBox.Cancel)
    window, source, profile = _live_editor()
    try:
        record = _direction_record(profile)
        baseline = window._normalized_baseline["PRIMARY_BEAM_DIRECTION"][0]
        # Stage a local edit, then have the IOC move the same record elsewhere.
        parameters = window._parameters()
        parameters["PRIMARY_BEAM_DIRECTION"][0] = baseline + 5
        monkeypatch.setattr(window, "_parameters", lambda: parameters)
        with window.pv_lock:
            window.pv_values[record] = baseline + 9

        pending = window._pending_change()
        assert pending["conflicts"], "divergent edits must conflict"
        assert not pending["adopted"]
        # save_changes() is what BaseWindow's close gate calls.
        assert window.save_changes() is False
        assert source.save_attempts == 0
    finally:
        window._stop_worker()
