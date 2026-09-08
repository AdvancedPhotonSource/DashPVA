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

"""Qt smoke tests for the profile-driven RSM parameter IOC editor window.

SimulatorWindow/AxisTable/PollWorker are defined by the module-level
_build_gui_classes() factory rather than at import time (PyQt5 is kept out of
module scope so the same file can run as the pvaccess IOC subprocess without
ever importing it -- see ioc_rsm_parameter.py's docstring). Calling the
factory here is what makes these classes instantiable from a test at all.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox

import dashpva.consumers.ioc_rsm_parameter as iocmod
from dashpva.consumers.ioc_rsm_parameter import (
    ActiveProfileChanged,
    ProfileContentMismatch,
    _build_gui_classes,
)
from dashpva.utils.config.revision import mapping_revision
from dashpva.utils.rsm_geometry import DETECTOR_SETUP_FIELDS
from dashpva.utils.rsm_parameter_config import (
    RSMParameterEditSession,
    default_parameter_mapping,
    profile_from_raw,
)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class _FakeSource:
    """Enough of the SnapshotConfigSource protocol to build an edit session."""

    def __init__(self, raw):
        self.raw = raw

    def load_snapshot(self):
        return self.raw, mapping_revision(self.raw)

    def replace_if_revision(self, full_config, revision):
        raise NotImplementedError("these tests never save")


def _raw(parameters=None):
    return {
        "IOC_PREFIX": "sim:",
        "IOC_RSM_PARAMETER": parameters or default_parameter_mapping(),
    }


def _make_window(parameters=None):
    raw = _raw(parameters)
    session = RSMParameterEditSession(_FakeSource(raw))
    profile = profile_from_raw(raw)
    _, _, SimulatorWindow = _build_gui_classes()
    return SimulatorWindow(session, profile, lambda snapshot: None, {}, threading.Lock())


@pytest.fixture()
def window(qapp):
    win = _make_window()
    yield win
    win._stop_worker()


# --- Labels: explicit, not generated from key.replace('_',' ').title() -----


def test_ub_matrix_has_explicit_label_and_dedicated_widget(window):
    assert window._change_target("UB Matrix (PV or value)") is window.ub_matrix_edit
    # Must have been pulled out of the generic calibration loop, not just
    # relabeled in place -- otherwise "Ub Matrix" would still be generated.
    assert "UB_MATRIX" not in window.calibration_values
    labels = {label.text().rstrip(":") for label in window.findChildren(QLabel)}
    assert "UB Matrix (PV or value)" in labels


def test_distance_has_explicit_label_and_dedicated_widget(window):
    target = window._change_target("Distance (PV or value)")
    assert target is window.detector_distance_edit
    assert "DISTANCE" not in window.detector_setup_values
    labels = {label.text().rstrip(":") for label in window.findChildren(QLabel)}
    assert "Distance (PV or value)" in labels


# --- UB matrix / distance: literal vs PV rendering, tooltip -----------------


def test_ub_matrix_widget_shows_literal_by_default(window):
    assert window.ub_matrix_edit.text() == json.dumps(
        list(window.profile.ub_matrix), sort_keys=True
    )
    assert "flat row-major JSON array" in window.ub_matrix_edit.toolTip()
    assert window.ub_matrix_edit.alignment() & Qt.AlignRight


def test_ub_matrix_widget_shows_pv_and_fallback_tooltip_when_configured(qapp):
    parameters = default_parameter_mapping()
    parameters["UB_MATRIX_SOURCE_PV"] = "28idbSoft:spec:UB"
    win = _make_window(parameters)
    try:
        assert win.ub_matrix_edit.text() == "28idbSoft:spec:UB"
        assert "Fallback" in win.ub_matrix_edit.toolTip()
    finally:
        win._stop_worker()


def test_distance_widget_shows_literal_by_default(window):
    assert window.detector_distance_edit.text() == json.dumps(
        window.profile.detector_setup["DISTANCE"]
    )
    assert "single finite positive scalar in mm" in window.detector_distance_edit.toolTip()
    assert window.detector_distance_edit.alignment() & Qt.AlignRight


def test_distance_widget_shows_pv_and_fallback_tooltip_when_configured(qapp):
    parameters = default_parameter_mapping()
    parameters["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"] = "6idb1:distance.RBV"
    win = _make_window(parameters)
    try:
        assert win.detector_distance_edit.text() == "6idb1:distance.RBV"
        assert "Fallback" in win.detector_distance_edit.toolTip()
    finally:
        win._stop_worker()


# --- Detector setup group: one row per DETECTOR_SETUP_FIELDS entry ---------


def test_detector_setup_group_has_one_row_per_declared_field(window):
    # DISTANCE gets its own dedicated PV-or-value widget instead of a plain
    # per-field row, so it's excluded from the generic field dict.
    assert set(window.detector_setup_values) == set(DETECTOR_SETUP_FIELDS) - {
        "DISTANCE"
    }
    assert len(window.detector_setup_values) == len(DETECTOR_SETUP_FIELDS) - 1
    labels = {label.text().rstrip(":") for label in window.findChildren(QLabel)}
    assert {"ROI", "Detector rotation", "Tilt azimuth", "Frame axis order"} <= labels


def test_required_detector_fields_render_prefilled(window):
    assert window.detector_setup_values["PIXEL_DIRECTION_1"].text() == "z-"
    assert window.detector_setup_values["PIXEL_DIRECTION_2"].text() == "x-"
    assert window.detector_setup_values["CENTER_CHANNEL_PIXEL"].text() == "300.0, 300.0"
    assert window.detector_setup_values["SIZE"].text() == "28.38, 28.38"
    assert window.detector_setup_values["UNITS"].text() == "mm"


def test_optional_detector_fields_render_blank_when_absent(window):
    for key in ("ROI", "BINNING", "DETROT", "TILT", "TILTAZIMUTH", "PIXEL_SIZE"):
        assert window.detector_setup_values[key].text() == "", key


def test_optional_detector_fields_have_placeholder_hints(window):
    assert window.detector_setup_values["ROI"].placeholderText()
    assert window.detector_setup_values["BINNING"].placeholderText()


def test_detector_tooltips_use_direction_order_not_array_row_column(window):
    center = window.detector_setup_values["CENTER_CHANNEL_PIXEL"].toolTip()
    roi = window.detector_setup_values["ROI"].toolTip()
    axis_order = window.detector_setup_values["FRAME_AXIS_ORDER"].toolTip()

    assert "direction 1, direction 2" in center
    assert "[start1, stop1, start2, stop2)" in roi
    assert "array rows/columns" in axis_order


def test_detector_inputs_are_uniformly_right_aligned(window):
    assert all(
        edit.alignment() & Qt.AlignRight
        for edit in window.detector_setup_values.values()
    )
    assert window.prefix_edit.alignment() & Qt.AlignRight
    assert window.energy_edit.alignment() & Qt.AlignRight


def test_switching_distance_from_pv_to_literal_clears_the_source_key():
    # Regression: DISTANCE_SOURCE_PV was being swept into
    # _detector_setup_extras as an "unknown key" on load (it isn't in
    # DETECTOR_SETUP_FIELDS, since it's handled by the dedicated distance
    # widget, not the generic per-field loop), so switching the field back to
    # a literal left the stale PV name in the saved DETECTOR_SETUP dict.
    parameters = default_parameter_mapping()
    parameters["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"] = "6idb1:distance.RBV"
    win = _make_window(parameters)
    try:
        assert win._detector_setup_extras == {}

        win.detector_distance_edit.setText("500.0")
        edits = win._detector_setup_edits()

        assert "DISTANCE_SOURCE_PV" not in edits
        assert edits["DISTANCE"] == 500.0
    finally:
        win._stop_worker()


def test_switching_distance_from_fresh_literal_to_pv_keeps_the_typed_literal(window):
    # Regression: typing a literal (no Apply yet), then switching to a PV name
    # in the same session, used to silently discard the just-typed literal and
    # revert the fallback to whatever self.profile still held (the last
    # applied value) -- self.profile only updates on Apply/Reload, never on a
    # keystroke, so the fallback source used to be stale by definition.
    window.detector_distance_edit.setText("600")
    window.detector_distance_edit.setText("new:distance:pv")

    parameters = window._parameters()

    assert parameters["DETECTOR_SETUP"]["DISTANCE"] == 600.0
    assert parameters["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"] == "new:distance:pv"


def test_switching_ub_matrix_from_fresh_literal_to_pv_keeps_the_typed_literal(window):
    window.ub_matrix_edit.setText(json.dumps([2, 0, 0, 0, 2, 0, 0, 0, 2]))
    window.ub_matrix_edit.setText("some:ub:pv")

    parameters = window._parameters()

    assert parameters["UB_MATRIX"] == [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
    assert parameters["UB_MATRIX_SOURCE_PV"] == "some:ub:pv"


def test_detector_setup_round_trips_original_int_and_float_types(qapp):
    parameters = default_parameter_mapping()
    parameters["DETECTOR_SETUP"]["DETECTOR_SHAPE"] = [512, 512]
    win = _make_window(parameters)
    try:
        assert win.detector_setup_values["DETECTOR_SHAPE"].text() == "512, 512"
        edits = win._detector_setup_edits()
        assert edits["DETECTOR_SHAPE"] == [512, 512]
        assert all(isinstance(v, int) for v in edits["DETECTOR_SHAPE"])
    finally:
        win._stop_worker()


# --- Change-review collapse: one logical row, keep/drop moves both halves --


def test_change_review_collapses_ub_matrix_pair_into_one_row(window):
    window.ub_matrix_edit.setText("28idbSoft:spec:UB")

    rows = window.unsaved_changes_rows()

    ub_rows = [row for row in rows if row[1] == "UB Matrix (PV or value)"]
    assert len(ub_rows) == 1
    assert ub_rows[0][0] == "change"
    assert ub_rows[0][3] == "28idbSoft:spec:UB"
    # Not also reported as two independent flattened-key rows.
    assert not any(row[1] in ("UB_MATRIX", "UB_MATRIX_SOURCE_PV") for row in rows)


def test_change_review_collapses_distance_pair_into_one_row(window):
    window.detector_distance_edit.setText("6idb1:distance.RBV")

    rows = window.unsaved_changes_rows()

    distance_rows = [row for row in rows if row[1] == "Distance (PV or value)"]
    assert len(distance_rows) == 1
    assert distance_rows[0][3] == "6idb1:distance.RBV"
    assert not any(
        "DISTANCE" in row[1] and row[1] != "Distance (PV or value)" for row in rows
    )


def test_unchanged_ub_and_distance_produce_no_review_rows(window):
    rows = window.unsaved_changes_rows()
    assert not any(
        row[1] in ("UB Matrix (PV or value)", "Distance (PV or value)") for row in rows
    )


def test_apply_change_decisions_keep_writes_both_ub_subkeys(window):
    window.ub_matrix_edit.setText("28idbSoft:spec:UB")
    ub_row = next(
        row
        for row in window.unsaved_changes_rows()
        if row[1] == "UB Matrix (PV or value)"
    )

    window.apply_change_decisions(kept=[ub_row], dropped=[])

    parameters = window._parameters()
    assert parameters["UB_MATRIX_SOURCE_PV"] == "28idbSoft:spec:UB"
    assert parameters["UB_MATRIX"] == list(window.profile.ub_matrix)


def test_apply_change_decisions_drop_reverts_both_ub_subkeys(window):
    original_text = window.ub_matrix_edit.text()
    window.ub_matrix_edit.setText("28idbSoft:spec:UB")
    ub_row = next(
        row
        for row in window.unsaved_changes_rows()
        if row[1] == "UB Matrix (PV or value)"
    )

    window.apply_change_decisions(kept=[], dropped=[ub_row])

    assert window.ub_matrix_edit.text() == original_text
    parameters = window._parameters()
    assert parameters["UB_MATRIX_SOURCE_PV"] == ""


def test_apply_change_decisions_keep_writes_both_distance_subkeys(window):
    window.detector_distance_edit.setText("6idb1:distance.RBV")
    distance_row = next(
        row
        for row in window.unsaved_changes_rows()
        if row[1] == "Distance (PV or value)"
    )

    window.apply_change_decisions(kept=[distance_row], dropped=[])

    parameters = window._parameters()
    assert parameters["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"] == "6idb1:distance.RBV"
    assert parameters["DETECTOR_SETUP"]["DISTANCE"] == pytest.approx(
        window.profile.detector_setup["DISTANCE"]
    )


def test_detector_field_review_row_is_editable_and_can_be_dropped(window):
    original = window.detector_setup_values["SIZE"].text()
    window.detector_setup_values["SIZE"].setText("30.0, 31.0")
    row = next(
        item for item in window.unsaved_changes_rows() if item[1] == "DETECTOR_SETUP.SIZE"
    )

    assert window.is_change_editable(row[1])
    window.apply_change_decisions(kept=[], dropped=[row])

    assert window.detector_setup_values["SIZE"].text() == original
    assert window._parameters()["DETECTOR_SETUP"]["SIZE"] == [28.38, 28.38]


# --- Resolved-profile-identity 3-way activation-mismatch routing -----------
#
# _activate_snapshot/_mark_out_of_sync/_mark_profile_changed were previously
# only exercised via the pure _classify_activation_mismatch/
# _describe_config_differences helpers in test_rsm_parameter_ioc.py, never
# through a real SimulatorWindow -- so a broken isinstance dispatch (e.g.
# ActiveProfileChanged falling through to the generic "Retry" branch when the
# docstring says retrying can never succeed) would pass every existing test.
# QMessageBox.exec_ is patched directly on the class (a single shared PyQt5
# object regardless of which module imported it -- see
# test_base_window_close_gate.py for the same established pattern) so the
# real, blocking modal dialogs these paths show never hang the test run.


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: 0)


def test_activate_snapshot_raises_active_profile_changed_when_profile_moved(window, monkeypatch):
    window._startup_locator = 5
    monkeypatch.setattr(iocmod, "_active_profile_identity", lambda: 7)
    monkeypatch.setattr(iocmod.app_settings, "reload", lambda: None)
    monkeypatch.setattr(iocmod.app_settings, "RAW_CONFIG", {"different": True})

    with pytest.raises(ActiveProfileChanged):
        window._activate_snapshot({"snapshot": True})


def test_activate_snapshot_raises_profile_content_mismatch_when_identity_unchanged(
    window, monkeypatch
):
    window._startup_locator = 5
    monkeypatch.setattr(iocmod, "_active_profile_identity", lambda: 5)
    monkeypatch.setattr(iocmod.app_settings, "reload", lambda: None)
    monkeypatch.setattr(iocmod.app_settings, "RAW_CONFIG", {"IOC_RSM_PARAMETER": {"X": 2}})

    with pytest.raises(ProfileContentMismatch):
        window._activate_snapshot({"IOC_RSM_PARAMETER": {"X": 1}})


def test_mark_out_of_sync_on_active_profile_changed_clears_retry_and_pending(window):
    window.retry_button = MagicMock()
    window.pending_snapshot = {"stale": True}

    window._mark_out_of_sync({"snapshot": True}, ActiveProfileChanged(5, 7))

    # No retry offered -- the docstring is explicit that retrying a moved
    # profile can never succeed, so re-enabling it here would mislead the user.
    window.retry_button.setVisible.assert_called_once_with(False)
    assert window.pending_snapshot is None
    assert "now the active profile" in window.profile_notice.text()


def test_mark_out_of_sync_on_profile_content_mismatch_keeps_retry_and_pending(window):
    window.retry_button = MagicMock()
    snapshot = {"snapshot": True}

    window._mark_out_of_sync(
        snapshot, ProfileContentMismatch(["Sample axis 1 - DIRECTION: saved 'x+', read back 'y+'"], 1)
    )

    # A concurrent-writer content mismatch is retriable -- unlike a moved
    # profile, the target didn't move, so Retry IOC sync can still succeed.
    window.retry_button.setVisible.assert_called_once_with(True)
    assert window.pending_snapshot == snapshot
    assert "1 setting" in window.profile_notice.text()


def test_mark_out_of_sync_on_generic_restart_failure_keeps_retry_and_pending(window):
    window.retry_button = MagicMock()
    snapshot = {"snapshot": True}

    window._mark_out_of_sync(snapshot, RuntimeError("IOC did not respond within 15 seconds"))

    window.retry_button.setVisible.assert_called_once_with(True)
    assert window.pending_snapshot == snapshot
    assert "did not restart" in window.profile_notice.text()
