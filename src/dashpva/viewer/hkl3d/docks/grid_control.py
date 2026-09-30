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

"""Live gridded-volume controls for the HKL 3D viewer.

Flow: Connect, watch the observed H/K/L range fill the bounds automatically,
Start Grid, Pause/Resume with the same button, New Grid… to discard and unlock.

Bounds are frozen at Start -- Gridder3D latches its range, and resizing a grid
mid-run would change what a voxel means. Automatic bounds only know positions
already observed, so later scan motion can leave the grid; the out-of-grid
percentage is reported while accumulating.
"""

from PyQt5.QtCore import QSettings, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import dashpva.settings as app_settings
from dashpva.utils.rsm_gridder import (
    RSMMergeError,
    ensure_memory_available,
    estimate_grid_memory,
)
from dashpva.viewer.core.docks.base_dock import BaseDock

_AXES = ("H", "K", "L")
_BOUND_KEYS = ("HMIN", "HMAX", "KMIN", "KMAX", "LMIN", "LMAX")
AUTO, MANUAL = "auto", "manual"
# Observed data within this fraction of a manual edge is flagged as close.
_EDGE_FRACTION = 0.05


def _settings() -> QSettings:
    return QSettings("DashPVA", "Viewer")


class GridControlDock(BaseDock):
    """Start/Pause/Resume, New Grid…, Save Volume…, and collapsible Grid Settings."""

    start_requested = pyqtSignal(dict)
    stop_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    save_requested = pyqtSignal()

    _EDITABLE_STATES = {"idle", "unreachable"}
    _LOCKED_STATES = {"running", "stopped", "saving"}

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Live Grid", main_window=main_window,
                         segment_name="hkl", dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._state = "idle"
        self._command_pending = False
        self._remote_busy = False
        self._has_accumulator = False
        self._frames_accepted = 0
        self._observed_bounds: list = []
        self._locked_payload: dict = {}
        self._local_error = ""
        self._mode = AUTO
        self._manual_dirty = False
        self._programmatic = False
        self._setup_ui()
        self._restore_preferences()
        self._apply_running_state()

    # -- construction ------------------------------------------------------

    def _setup_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        self.observed_label = QLabel()
        self.observed_label.setObjectName("gridObserved")
        self.observed_label.setTextFormat(Qt.RichText)
        layout.addWidget(self.observed_label)

        self.state_label = QLabel("Idle")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)

        row = QHBoxLayout()
        self.btn_primary = QPushButton("Start Grid")
        self.btn_new = QPushButton("New Grid…")
        self.btn_new.setToolTip("Discard the accumulated volume and unlock the grid settings.")
        self.btn_save = QPushButton("Save Volume…")
        self.btn_save.setToolTip("Available while paused with accumulated data.")
        for button in (self.btn_primary, self.btn_new, self.btn_save):
            row.addWidget(button)
        layout.addLayout(row)
        self.btn_primary.clicked.connect(self._on_primary)
        self.btn_new.clicked.connect(self._on_new_grid)
        self.btn_save.clicked.connect(self.save_requested)

        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)
        self.notice_label = QLabel("")
        self.notice_label.setWordWrap(True)
        layout.addWidget(self.notice_label)

        self.settings_toggle = QToolButton()
        self.settings_toggle.setText("Grid Settings")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.settings_toggle.setArrowType(Qt.RightArrow)
        self.settings_toggle.toggled.connect(self._on_settings_toggled)
        layout.addWidget(self.settings_toggle)
        self.settings_body = self._build_settings()
        self.settings_body.setVisible(False)
        layout.addWidget(self.settings_body)

        layout.addStretch()
        self.set_scrollable_widget(container)
        self._show_observed([])

    def _build_settings(self) -> QWidget:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(12, 0, 0, 0)

        self.rb_auto = QRadioButton("Automatic (Observed)")
        self.rb_manual = QRadioButton("Manual")
        self.rb_auto.setToolTip(
            "Bounds follow the observed H/K/L range (+5%) until Start Grid. "
            "Only positions already seen are known — later scan motion may leave the grid."
        )
        self.rb_manual.setToolTip("Enter the planned scan bounds before any data arrives.")
        self._mode_group = QButtonGroup(body)
        self._mode_group.addButton(self.rb_auto)
        self._mode_group.addButton(self.rb_manual)
        self.rb_auto.setChecked(True)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Bounds:"))
        mode_row.addWidget(self.rb_auto)
        mode_row.addWidget(self.rb_manual)
        mode_row.addStretch()
        grid.addLayout(mode_row, 0, 0, 1, 4)
        self.rb_manual.toggled.connect(self._on_mode_toggled)

        grid.addWidget(QLabel("min"), 1, 1)
        grid.addWidget(QLabel("max"), 1, 2)
        grid.addWidget(QLabel("bins"), 1, 3)
        self.min_boxes, self.max_boxes, self.bin_boxes = {}, {}, {}
        for row, axis in enumerate(_AXES, start=2):
            grid.addWidget(QLabel(axis), row, 0)
            low, high = QDoubleSpinBox(), QDoubleSpinBox()
            for box in (low, high):
                box.setDecimals(5)
                box.setRange(-1000.0, 1000.0)
                box.setSingleStep(0.001)
                box.setFixedWidth(105)
                box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                box.valueChanged.connect(self._on_bound_edited)
            bins = QSpinBox()
            # Gridder3D.axis() returns a bare float for n == 1, and one bin
            # carries no spatial information anyway.
            bins.setRange(2, 1024)
            bins.setValue(128)
            bins.setFixedWidth(70)
            bins.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            bins.valueChanged.connect(self._on_bins_edited)
            grid.addWidget(low, row, 1)
            grid.addWidget(high, row, 2)
            grid.addWidget(bins, row, 3)
            self.min_boxes[axis], self.max_boxes[axis], self.bin_boxes[axis] = low, high, bins

        self.memory_label = QLabel()
        self.memory_label.setWordWrap(True)
        grid.addWidget(self.memory_label, len(_AXES) + 2, 0, 1, 4)
        self.btn_use_observed = QPushButton("Use Observed Bounds")
        self.btn_use_observed.setToolTip("Copy the observed H/K/L range (+5%) into min/max.")
        self.btn_use_observed.clicked.connect(self.use_observed_bounds)
        grid.addWidget(self.btn_use_observed, len(_AXES) + 3, 0, 1, 4)
        return body

    # -- preferences (per profile) ----------------------------------------

    @staticmethod
    def _pref_key(name: str) -> str:
        return f"hkl3d_grid/{app_settings.LOCATOR or 'default'}/{name}"

    def _restore_preferences(self) -> None:
        s = _settings()
        self._programmatic = True
        try:
            bins = s.value(self._pref_key("bins"), [], type=list)
            if len(bins) == 3:
                for axis, value in zip(_AXES, bins):
                    self.bin_boxes[axis].setValue(int(value))
            bounds = s.value(self._pref_key("manual_bounds"), [], type=list)
            if len(bounds) == 6:
                self.set_bounds(self._bounds_dict(bounds))
            if s.value(self._pref_key("bounds_mode"), AUTO, type=str) == MANUAL:
                self._mode = MANUAL
                self.rb_manual.setChecked(True)
        finally:
            self._programmatic = False
        self._update_memory_estimate()

    def _save_preferences(self) -> None:
        s = _settings()
        s.setValue(self._pref_key("bounds_mode"), self._mode)
        s.setValue(self._pref_key("bins"), [self.bin_boxes[a].value() for a in _AXES])
        if self._mode == MANUAL:
            s.setValue(self._pref_key("manual_bounds"), self._field_bounds())

    # -- helpers -----------------------------------------------------------

    def _set_level(self, widget, level: str) -> None:
        widget.setProperty("messageLevel", level)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    @staticmethod
    def _bounds_dict(values) -> dict:
        return {key: float(value) for key, value in zip(_BOUND_KEYS, values)}

    def _field_bounds(self) -> list:
        values = []
        for axis in _AXES:
            values.extend((self.min_boxes[axis].value(), self.max_boxes[axis].value()))
        return values

    def set_bounds(self, bounds: dict) -> None:
        was = self._programmatic
        self._programmatic = True
        try:
            for axis in _AXES:
                if f"{axis}MIN" in bounds:
                    self.min_boxes[axis].setValue(float(bounds[f"{axis}MIN"]))
                if f"{axis}MAX" in bounds:
                    self.max_boxes[axis].setValue(float(bounds[f"{axis}MAX"]))
        finally:
            self._programmatic = was

    def bounds_payload(self) -> dict:
        payload = self._bounds_dict(self._field_bounds())
        payload["NX"] = self.bin_boxes["H"].value()
        payload["NY"] = self.bin_boxes["K"].value()
        payload["NZ"] = self.bin_boxes["L"].value()
        return payload

    def _confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        return answer == QMessageBox.Yes

    # -- user actions ------------------------------------------------------

    def _on_settings_toggled(self, expanded: bool) -> None:
        self.settings_body.setVisible(expanded)
        self.settings_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _on_mode_toggled(self, manual: bool) -> None:
        if self._programmatic:
            return
        if manual:
            # Start from the values on screen (the automatic bounds), not blanks.
            self._mode = MANUAL
            self._manual_dirty = False
        else:
            if self._manual_dirty and not self._confirm(
                "Use automatic bounds?",
                "Discard the manually entered bounds and follow the observed range?",
            ):
                self._programmatic = True
                self.rb_manual.setChecked(True)
                self._programmatic = False
                return
            self._mode = AUTO
            self._manual_dirty = False
            self._follow_observed()
        self._save_preferences()
        self._apply_running_state()
        self._refresh_validation()

    def _on_bound_edited(self, _value) -> None:
        if self._programmatic:
            return
        if self._mode == AUTO:
            # Editing a bound means the user wants control: stop following.
            self._programmatic = True
            self.rb_manual.setChecked(True)
            self._programmatic = False
            self._mode = MANUAL
        self._manual_dirty = True
        self._save_preferences()
        self._apply_running_state()
        self._refresh_validation()

    def _on_bins_edited(self, _value) -> None:
        self._update_memory_estimate()
        if not self._programmatic:
            self._save_preferences()
        self._refresh_validation()

    def use_observed_bounds(self) -> None:
        if len(self._observed_bounds) != 6:
            return
        self.set_bounds(self._bounds_dict(self._observed_bounds))
        self._manual_dirty = True
        self._save_preferences()
        self._refresh_validation()

    def _on_primary(self) -> None:
        if self._state == "running":
            self.stop_requested.emit()
        elif self._state == "stopped":
            # Resume must send exactly the latched configuration.
            self.start_requested.emit(dict(self._locked_payload))
        else:
            problem = self.start_problem()
            if problem:
                self.show_error(problem)
                return
            self.show_error("")
            self._save_preferences()
            self.start_requested.emit(self.bounds_payload())

    def _on_new_grid(self) -> None:
        if (self._has_accumulator or self._frames_accepted) and not self._confirm(
            "Start a new grid?",
            f"Discard the accumulated volume ({self._frames_accepted:,} frames) and "
            "unlock the grid settings?",
        ):
            return
        self.clear_requested.emit()

    # -- validation --------------------------------------------------------

    def start_problem(self) -> str:
        """Why Start Grid cannot run now, or '' when it can."""
        if self._mode == AUTO and len(self._observed_bounds) != 6:
            return "Waiting for the first valid HKL frame to determine bounds."
        for axis in _AXES:
            if self.max_boxes[axis].value() <= self.min_boxes[axis].value():
                return f"{axis} max must be greater than {axis} min."
        try:
            ensure_memory_available(estimate_grid_memory(
                *(self.bin_boxes[a].value() for a in _AXES)
            ))
        except RSMMergeError as exc:
            return str(exc)
        return ""

    def _refresh_validation(self) -> None:
        idle = self._state in self._EDITABLE_STATES
        problem = self.start_problem() if idle else ""
        self.hint_label.setText(problem)
        self.hint_label.setVisible(bool(problem))
        self._set_level(self.hint_label, "warning" if problem else "info")
        self.btn_primary.setEnabled(self._primary_enabled(problem))

    def _primary_enabled(self, problem: str) -> bool:
        if self._command_pending or self._remote_busy:
            return False
        if self._state == "running":
            return True
        if self._state == "stopped":
            return bool(self._locked_payload)
        return not problem

    def _update_memory_estimate(self) -> None:
        estimate = estimate_grid_memory(*(self.bin_boxes[a].value() for a in _AXES))
        voxels = 1
        for axis in _AXES:
            voxels *= self.bin_boxes[axis].value()
        self.memory_label.setText(
            f"Estimated memory: {estimate.peak_bytes / 1024 ** 2:,.0f} MB "
            f"({voxels:,} voxels)."
        )

    # -- state from the consumer ------------------------------------------

    def _follow_observed(self) -> None:
        if (
            self._mode == AUTO
            and self._state in self._EDITABLE_STATES
            and len(self._observed_bounds) == 6
        ):
            self.set_bounds(self._bounds_dict(self._observed_bounds))

    def _observed_status(self) -> tuple:
        """(level, message) comparing the observed range with manual bounds."""
        if self._mode != MANUAL or len(self._observed_bounds) != 6:
            return "info", ""
        outside, close = [], []
        for index, axis in enumerate(_AXES):
            lo, hi = self._observed_bounds[2 * index], self._observed_bounds[2 * index + 1]
            box_lo, box_hi = self.min_boxes[axis].value(), self.max_boxes[axis].value()
            margin = (box_hi - box_lo) * _EDGE_FRACTION
            if lo < box_lo or hi > box_hi:
                outside.append(axis)
            elif lo < box_lo + margin or hi > box_hi - margin:
                close.append(axis)
        if outside:
            return "error", f"Observed data extends outside the grid on {', '.join(outside)}."
        if close:
            return "warning", f"Observed data is close to the grid edge on {', '.join(close)}."
        return "success", "Observed data is inside the grid."

    def _show_observed(self, observed) -> None:
        self._observed_bounds = [float(v) for v in observed] if len(observed) == 6 else []
        if not self._observed_bounds:
            self.observed_label.setText("<b>Observed range:</b> waiting for frames…")
            return
        b = self._observed_bounds
        rows = "".join(
            f"<tr><td>{axis}&nbsp;&nbsp;</td><td align='right'>{b[2 * i]:.5f}</td>"
            f"<td>&nbsp;to&nbsp;</td><td align='right'>{b[2 * i + 1]:.5f}</td></tr>"
            for i, axis in enumerate(_AXES)
        )
        self.observed_label.setText(f"<b>Observed range</b><table>{rows}</table>")

    def update_status(self, state: dict) -> None:
        """Refresh from a remote status dict. Never overwrites manual fields."""
        if "state" in state:
            self._state = str(state["state"])
        locked = self._state in self._LOCKED_STATES
        shape = state.get("grid_shape", [])
        bounds = state.get("grid_bounds", [])
        self._has_accumulator = len(shape) > 0
        self._frames_accepted = int(state.get("frames_accepted", 0))
        if locked and len(shape) == 3 and len(bounds) == 6:
            self._locked_payload = self._bounds_dict(bounds)
            self._locked_payload.update(NX=int(shape[0]), NY=int(shape[1]), NZ=int(shape[2]))
            self._programmatic = True
            try:
                self.set_bounds(self._locked_payload)
                for axis, value in zip(_AXES, shape):
                    self.bin_boxes[axis].setValue(int(value))
            finally:
                self._programmatic = False
        elif not locked:
            self._locked_payload = {}
        self._show_observed(state.get("observed_bounds", []))
        self._follow_observed()
        self._remote_busy = (
            self._state == "saving" or state.get("save_state") == "running"
        )
        self._apply_running_state()
        self._show_state_line(state)
        self._show_notice(state)

    def _show_state_line(self, state: dict) -> None:
        frames = self._frames_accepted
        voxels = int(state.get("voxels_filled", 0))
        text = {
            "running": f"Accumulating — {frames:,} frames, {voxels:,} voxels",
            "stopped": f"Paused — {frames:,} frames, {voxels:,} voxels",
            "saving": "Saving volume…",
            "unreachable": "Grid consumer unreachable",
        }.get(self._state, "Idle — press Start Grid to accumulate")
        self.state_label.setText(text)

    def _show_notice(self, state: dict) -> None:
        if state.get("last_error"):
            message, level = str(state["last_error"]), "error"
        elif self._local_error:
            message, level = self._local_error, "error"
        elif self._frames_accepted == 0 and int(state.get("frames_rejected_binding", 0)) > 0:
            reason = str(state.get("last_binding_rejection") or "metadata mismatch")
            message = (
                "No frames reached the grid — metadata binding rejected them "
                f"({reason.replace('_', ' ')})."
            )
            level = "error"
        elif self._sparse_message(state):
            message, level = self._sparse_message(state), "warning"
        elif state.get("incomplete"):
            message, level = self._incomplete_message(state), "warning"
        else:
            level, message = self._observed_status()
        self.notice_label.setText(message)
        self.notice_label.setVisible(bool(message))
        self._set_level(self.notice_label, level)

    @staticmethod
    def _sparse_message(state: dict) -> str:
        """Frames covering a tiny part of the box render as a speck: say so."""
        shape = state.get("preview_shape", [])
        filled = int(state.get("voxels_filled", 0))
        if len(shape) != 3 or filled == 0:
            return ""
        total = int(shape[0]) * int(shape[1]) * int(shape[2])
        if filled / total >= app_settings.RSM_GRID_SPARSE_FILL_FRACTION:
            return ""
        return (
            f"Data fills only {filled:,} of {total:,} voxels — the frames cover a "
            "far smaller H/K/L region than these bounds. New Grid… with Automatic "
            "bounds to resolve it."
        )

    @staticmethod
    def _incomplete_message(state: dict) -> str:
        parts = []
        skipped = int(state.get("frames_missing_upstream", 0))
        if skipped:
            parts.append(
                f"{skipped:,} frames skipped upstream (consumer slower than the detector)"
            )
        outside = int(state.get("points_out_of_range", 0))
        binned = int(state.get("points_binned", 0))
        if outside:
            parts.append(
                f"{100.0 * outside / max(1, outside + binned):.1f}% of points are "
                "outside the grid"
            )
        rejected = int(state.get("frames_rejected_binding", 0)) + int(
            state.get("frames_rejected_processing", 0)
        )
        if rejected:
            parts.append(f"{rejected:,} frames rejected")
        if not parts:
            return "Preview is incomplete — it is a preview, not a complete record."
        return "Preview incomplete: " + "; ".join(parts) + "."

    def _apply_running_state(self) -> None:
        """Settings follow the remote state; a pending command only blocks buttons."""
        editable = self._state in self._EDITABLE_STATES
        blocked = self._command_pending or self._remote_busy
        for boxes in (self.min_boxes, self.max_boxes, self.bin_boxes):
            for box in boxes.values():
                box.setEnabled(editable)
        self.rb_auto.setEnabled(editable)
        self.rb_manual.setEnabled(editable)
        self.btn_use_observed.setVisible(self._mode == MANUAL)
        self.btn_use_observed.setEnabled(editable and len(self._observed_bounds) == 6)
        self.settings_toggle.setText("Grid Settings" if editable else "Grid Settings (locked)")
        self.btn_primary.setText(
            {"running": "Pause Grid", "stopped": "Resume Grid"}.get(self._state, "Start Grid")
        )
        self.btn_new.setEnabled(not blocked)
        self.btn_save.setEnabled(
            self._state == "stopped" and self._has_accumulator and not blocked
        )
        self._refresh_validation()

    def set_command_pending(self, pending: bool) -> None:
        self._command_pending = bool(pending)
        self._apply_running_state()

    def show_error(self, message: str) -> None:
        """Sticky viewer-side error (failed command/unreachable); '' clears it."""
        self._local_error = message
        if message:
            self.notice_label.setText(message)
            self.notice_label.setVisible(True)
            self._set_level(self.notice_label, "error")

    def mark_unreachable(self, message: str) -> None:
        """Status could not be read: never leave the settings locked on stale state."""
        self._state = "unreachable"
        self._remote_busy = False
        self._locked_payload = {}
        self.state_label.setText("Grid consumer unreachable")
        self._apply_running_state()
        self.show_error(message)
