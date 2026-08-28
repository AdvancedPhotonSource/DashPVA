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

Bounds must be fixed before the first frame -- Gridder3D latches its range on
first use, and rebinning mid-scan would change what a voxel means. So this dock
gates Start behind a concrete box, offers an estimate to fill it in, and locks
the fields for the duration of a run.

The estimate is explicitly labelled as *observed* bounds: it can only describe
frames already seen, so later scan motion may fall outside. That is why the
out-of-range counter is on screen next to it rather than buried in a log.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock

_AXES = ("H", "K", "L")


class GridControlDock(BaseDock):
    """Bounds, resolution, and start/stop/clear/save for the live grid."""

    start_requested = pyqtSignal(dict)
    stop_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    save_requested = pyqtSignal()
    estimate_requested = pyqtSignal()

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Live Grid", main_window=main_window,
                         segment_name="hkl", dock_area=Qt.RightDockWidgetArea,
        show=show)
        self._state = "idle"
        self._running = False
        self._busy = False
        self._remote_busy = False
        self._has_accumulator = False
        self._setup_ui()
        self._apply_running_state()

    # -- construction ------------------------------------------------------

    def _setup_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._build_bounds_group())
        layout.addWidget(self._build_buttons())
        layout.addWidget(self._build_status_group())
        layout.addStretch()
        self.setWidget(container)

    def _build_bounds_group(self) -> QGroupBox:
        group = QGroupBox("Grid bounds (HKL) and resolution")
        grid = QGridLayout(group)
        grid.addWidget(QLabel("min"), 0, 1)
        grid.addWidget(QLabel("max"), 0, 2)
        grid.addWidget(QLabel("bins"), 0, 3)

        self.min_boxes, self.max_boxes, self.bin_boxes = {}, {}, {}
        for row, axis in enumerate(_AXES, start=1):
            grid.addWidget(QLabel(axis), row, 0)
            low, high = QDoubleSpinBox(), QDoubleSpinBox()
            for box in (low, high):
                box.setDecimals(5)
                box.setRange(-1e6, 1e6)
            low.setValue(-1.0)
            high.setValue(1.0)
            bins = QSpinBox()
            # Gridder3D.axis() returns a bare float for n == 1, and one bin
            # carries no spatial information anyway.
            bins.setRange(2, 1024)
            bins.setValue(128)
            grid.addWidget(low, row, 1)
            grid.addWidget(high, row, 2)
            grid.addWidget(bins, row, 3)
            self.min_boxes[axis], self.max_boxes[axis], self.bin_boxes[axis] = low, high, bins

        self.memory_label = QLabel()
        self.memory_label.setWordWrap(True)
        grid.addWidget(self.memory_label, len(_AXES) + 1, 0, 1, 4)
        for box in self.bin_boxes.values():
            box.valueChanged.connect(self._update_memory_estimate)
        self._update_memory_estimate()
        return group

    def _build_buttons(self) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        self.btn_estimate = QPushButton("Estimate bounds")
        self.btn_estimate.setToolTip(
            "Fill the box from frames already received. Later scan motion may "
            "still fall outside it — watch the out-of-range count."
        )
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_clear = QPushButton("Clear")
        self.btn_save = QPushButton("Save volume…")
        self.btn_save.setToolTip("Only available once stopped.")
        for button in (self.btn_estimate, self.btn_start, self.btn_stop,
                       self.btn_clear, self.btn_save):
            row.addWidget(button)
        self.btn_estimate.clicked.connect(self.estimate_requested)
        self.btn_start.clicked.connect(self._emit_start)
        self.btn_stop.clicked.connect(self.stop_requested)
        self.btn_clear.clicked.connect(self.clear_requested)
        self.btn_save.clicked.connect(self.save_requested)
        return holder

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("Accumulation")
        form = QFormLayout(group)
        self.state_label = QLabel("idle")
        self.frames_label = QLabel("0")
        self.binned_label = QLabel("0")
        self.out_of_range_label = QLabel("0")
        self.filled_label = QLabel("0")
        self.notice_label = QLabel("")
        self.notice_label.setWordWrap(True)
        form.addRow("State", self.state_label)
        form.addRow("Frames accepted", self.frames_label)
        form.addRow("Points binned", self.binned_label)
        form.addRow("Points out of grid", self.out_of_range_label)
        form.addRow("Voxels filled", self.filled_label)
        form.addRow(self.notice_label)
        return group

    # -- behaviour ---------------------------------------------------------

    def _update_memory_estimate(self) -> None:
        """Gridder3D holds two float64 arrays and offers no dtype knob."""
        voxels = 1
        for axis in _AXES:
            voxels *= self.bin_boxes[axis].value()
        megabytes = voxels * 8 * 2 / (1024 ** 2)
        self.memory_label.setText(
            f"{voxels:,} voxels — about {megabytes:,.0f} MB resident "
            f"(two float64 accumulators)."
        )
        self._set_level(self.memory_label, "warning" if megabytes > 2048 else "info")

    def _set_level(self, widget, level: str) -> None:
        widget.setProperty("messageLevel", level)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def bounds_payload(self) -> dict:
        payload = {}
        for axis, prefix in zip(_AXES, ("H", "K", "L")):
            payload[f"{prefix}MIN"] = self.min_boxes[axis].value()
            payload[f"{prefix}MAX"] = self.max_boxes[axis].value()
        payload["NX"] = self.bin_boxes["H"].value()
        payload["NY"] = self.bin_boxes["K"].value()
        payload["NZ"] = self.bin_boxes["L"].value()
        return payload

    def _emit_start(self) -> None:
        for axis in _AXES:
            if self.max_boxes[axis].value() <= self.min_boxes[axis].value():
                self.notice_label.setText(
                    f"{axis} max must exceed {axis} min before starting."
                )
                self._set_level(self.notice_label, "error")
                return
        self.notice_label.setText("")
        self._set_level(self.notice_label, "info")
        self.start_requested.emit(self.bounds_payload())

    def set_bounds(self, bounds: dict) -> None:
        """Fill the fields from an estimate."""
        for axis, prefix in zip(_AXES, ("H", "K", "L")):
            if f"{prefix}MIN" in bounds:
                self.min_boxes[axis].setValue(float(bounds[f"{prefix}MIN"]))
            if f"{prefix}MAX" in bounds:
                self.max_boxes[axis].setValue(float(bounds[f"{prefix}MAX"]))

    def _apply_running_state(self) -> None:
        """Bounds are locked while running: they cannot change mid-run."""
        busy = self._busy or self._remote_busy
        for boxes in (self.min_boxes, self.max_boxes, self.bin_boxes):
            for box in boxes.values():
                box.setEnabled(self._state == "idle" and not busy)
        self.btn_estimate.setEnabled(self._state == "idle" and not busy)
        self.btn_start.setEnabled(
            self._state in {"idle", "stopped"} and not busy
        )
        self.btn_stop.setEnabled(self._running and not busy)
        self.btn_clear.setEnabled(not busy)
        self.btn_save.setEnabled(
            self._state == "stopped" and self._has_accumulator and not busy
        )

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._apply_running_state()

    def update_status(self, state: dict) -> None:
        """Refresh from a session state dict."""
        if "state" in state:
            self._state = str(state["state"])
        if "grid_shape" in state:
            self._has_accumulator = bool(state["grid_shape"])
            shape = state["grid_shape"]
            if len(shape) == 3:
                for axis, value in zip(_AXES, shape):
                    self.bin_boxes[axis].setValue(int(value))
        bounds = state.get("grid_bounds", [])
        if len(bounds) == 6:
            self.set_bounds(
                {
                    "HMIN": bounds[0],
                    "HMAX": bounds[1],
                    "KMIN": bounds[2],
                    "KMAX": bounds[3],
                    "LMIN": bounds[4],
                    "LMAX": bounds[5],
                }
            )
        self._running = self._state == "running"
        self._remote_busy = (
            self._state == "saving" or state.get("save_state") == "running"
        )
        self._apply_running_state()
        self.state_label.setText(self._state)
        self.frames_label.setText(f"{int(state.get('frames_accepted', 0)):,}")
        self.binned_label.setText(f"{int(state.get('points_binned', 0)):,}")

        out_of_range = int(state.get("points_out_of_range", 0))
        self.out_of_range_label.setText(f"{out_of_range:,}")
        self._set_level(self.out_of_range_label, "warning" if out_of_range else "info")
        self.filled_label.setText(f"{int(state.get('voxels_filled', 0)):,}")

        if state.get("last_error"):
            self.notice_label.setText(str(state["last_error"]))
            self._set_level(self.notice_label, "error")
        elif state.get("incomplete"):
            self.notice_label.setText(
                "Preview is incomplete — frames were dropped or fell outside "
                "the grid. It is a preview, not a complete record."
            )
            self._set_level(self.notice_label, "warning")
        else:
            self.notice_label.setText("")
            self._set_level(self.notice_label, "info")
