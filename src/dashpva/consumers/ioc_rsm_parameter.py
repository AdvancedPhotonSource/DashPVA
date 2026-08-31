#!/usr/bin/env python3
# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Profile-driven RSM-parameter IOC and staged geometry editor.

The GUI and pvAccess IOC run in separate processes. The child reads the active
profile from the database and is restarted only after a successful
compare-and-swap save -- so by the time it starts, the database already holds
exactly the snapshot that was approved. Configuration is never handed over as
a side-channel file: per AGENTS.md the database is the config source, with
TOML reserved for import/export.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping

import dashpva.settings as app_settings
from dashpva.utils.rsm_parameter_config import (
    RSMParameterEditSession,
    RSMParameterProfile,
    SnapshotActivationError,
    adoption_diff,
    apply_and_activate,
    merge_live_records,
    profile_from_raw,
    requires_adoption_confirmation,
    update_raw_profile,
    validate_parameter_profile,
)


def _flatten_config(
    value: object, prefix: str = "", *, index_lists: bool = False
) -> dict[str, Any]:
    """Flatten configuration fields for review without losing axis rows."""
    if index_lists and isinstance(value, (list, tuple)) and any(
        isinstance(item, Mapping) for item in value
    ):
        flattened: dict[str, Any] = {}
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(
                _flatten_config(item, path, index_lists=index_lists)
            )
        return flattened
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten_config(item, path, index_lists=index_lists))
    return flattened


def _restore_axis_row(
    rows: list[tuple[int | None, Mapping[str, Any]]],
    origin: int,
    values: Mapping[str, Any],
) -> list[tuple[int | None, dict[str, Any]]]:
    """Restore a removed loaded axis without changing any surviving identity."""
    restored = [(item_origin, dict(item)) for item_origin, item in rows]
    if any(item_origin == origin for item_origin, _item in restored):
        return restored
    insertion = len(restored)
    for row, (candidate, _item) in enumerate(restored):
        if candidate is not None and candidate >= 0 and candidate > origin:
            insertion = row
            break
    restored.insert(insertion, (origin, dict(values)))
    return restored


def _reorder_loaded_axis_rows(
    rows: list[tuple[int | None, Mapping[str, Any]]],
    order: list[int],
) -> list[tuple[int | None, dict[str, Any]]]:
    """Reorder loaded axes in place while keeping new-axis slots unchanged."""
    copied = [(origin, dict(values)) for origin, values in rows]
    loaded = [
        origin for origin, _values in copied if origin is not None and origin >= 0
    ]
    if len(order) != len(loaded) or set(order) != set(loaded):
        raise ValueError("axis order must contain every loaded identity exactly once")
    by_origin = {origin: values for origin, values in copied if origin is not None}
    ordered = iter(
        (origin, by_origin[origin]) for origin in order if origin in by_origin
    )
    return [
        next(ordered) if origin is not None and origin >= 0 else (origin, values)
        for origin, values in copied
    ]


def _review_pending_change(build_pending):
    """Keep an invalid staged RSM form visible to the close gate."""
    try:
        return build_pending(), []
    except ValueError as exc:
        return None, [
            (
                "change",
                "RSM_CONFIGURATION_INVALID",
                "valid loaded configuration",
                str(exc),
            )
        ]


def _has_pending_rsm_change(build_pending, raw: Mapping[str, Any]) -> bool:
    """Treat invalid staged values as dirty until saved or discarded."""
    try:
        return build_pending()["replacement"] != raw
    except ValueError:
        return True


def _ai(name: str) -> str:
    return (
        f'record(ai, "{name}") {{\n'
        '  field(DTYP, "Soft Channel")\n'
        '  field(PREC, "6")\n'
        '}\n'
    )


def _longout(name: str) -> str:
    return f'record(longout, "{name}") {{\n  field(DTYP, "Soft Channel")\n}}\n'


def _stringout(name: str) -> str:
    return f'record(stringout, "{name}") {{\n  field(DTYP, "Soft Channel")\n}}\n'


def _waveform(name: str, count: int) -> str:
    return (
        f'record(waveform, "{name}") {{\n'
        '  field(DTYP, "Soft Channel")\n'
        '  field(FTVL, "DOUBLE")\n'
        f'  field(NELM, "{count}")\n'
        '}\n'
    )


def build_ioc_database(profile: RSMParameterProfile) -> str:
    """Build an EPICS database for every configured circle without a max."""
    lines: list[str] = []
    prefix = profile.prefix
    for axis in profile.axes:
        base = f"{prefix}{axis.record_name}"
        lines.extend(
            (
                _ai(f"{base}:Position"),
                _ai(f"{base}:AxisNumber"),
                _stringout(f"{base}:DirectionAxis"),
                _stringout(f"{base}:SpecMotorName"),
            )
        )

    lines.extend(
        (
            _ai(f"{prefix}spec:Energy:Value"),
            _stringout(f"{prefix}spec:Energy:Units"),
            _waveform(f"{prefix}spec:UB_matrix:Value", 9),
        )
    )
    for group in (
        "PrimaryBeamDirection",
        "InplaneReferenceDirection",
        "SampleSurfaceNormalDirection",
    ):
        for index in range(1, 4):
            lines.append(_ai(f"{prefix}{group}:AxisNumber{index}"))

    lines.extend(
        (
            _stringout(f"{prefix}DetectorSetup:PixelDirection1"),
            _stringout(f"{prefix}DetectorSetup:PixelDirection2"),
            _waveform(f"{prefix}DetectorSetup:CenterChannelPixel", 2),
            _waveform(f"{prefix}DetectorSetup:Size", 2),
            _ai(f"{prefix}DetectorSetup:Distance"),
            _stringout(f"{prefix}DetectorSetup:Units"),
            _longout(f"{prefix}ScanOn:Value"),
            _stringout(f"{prefix}FilePath:Value"),
            _stringout(f"{prefix}FileName:Value"),
        )
    )
    return "".join(lines)


def static_ioc_values(profile: RSMParameterProfile) -> dict[str, Any]:
    """Return all frame-invariant IOC values for a validated profile."""
    prefix = profile.prefix
    values: dict[str, Any] = {}
    for axes in (profile.sample_axes, profile.detector_axes):
        for axis_number, axis in enumerate(axes, start=1):
            base = f"{prefix}{axis.record_name}"
            values[f"{base}:AxisNumber"] = float(axis_number)
            values[f"{base}:DirectionAxis"] = axis.direction
            values[f"{base}:SpecMotorName"] = axis.published_spec_motor_name

    values[f"{prefix}spec:Energy:Units"] = profile.energy_units
    values[f"{prefix}spec:UB_matrix:Value"] = list(profile.ub_matrix)
    for group, vector in (
        ("PrimaryBeamDirection", profile.primary_beam_direction),
        ("InplaneReferenceDirection", profile.inplane_reference_direction),
        ("SampleSurfaceNormalDirection", profile.sample_surface_normal_direction),
    ):
        for index, value in enumerate(vector, start=1):
            values[f"{prefix}{group}:AxisNumber{index}"] = float(value)

    detector = profile.detector_setup
    values[f"{prefix}DetectorSetup:PixelDirection1"] = detector["PIXEL_DIRECTION_1"]
    values[f"{prefix}DetectorSetup:PixelDirection2"] = detector["PIXEL_DIRECTION_2"]
    values[f"{prefix}DetectorSetup:CenterChannelPixel"] = list(
        detector["CENTER_CHANNEL_PIXEL"]
    )
    values[f"{prefix}DetectorSetup:Size"] = list(detector["SIZE"])
    values[f"{prefix}DetectorSetup:Distance"] = float(detector["DISTANCE"])
    values[f"{prefix}DetectorSetup:Units"] = detector["UNITS"]
    values[f"{prefix}ScanOn:Value"] = 0
    values[f"{prefix}FilePath:Value"] = ""
    values[f"{prefix}FileName:Value"] = ""
    return values


def all_pv_names(profile: RSMParameterProfile) -> list[tuple[str, str]]:
    """Return the dynamic display list for the IOC's records."""
    prefix = profile.prefix
    records: list[tuple[str, str]] = []
    for axis in profile.axes:
        base = f"{prefix}{axis.record_name}"
        records.extend(
            (
                (f"{base}:Position", f"{axis.label} position"),
                (f"{base}:AxisNumber", f"{axis.label} axis number"),
                (f"{base}:DirectionAxis", f"{axis.label} direction"),
                (f"{base}:SpecMotorName", f"{axis.label} label"),
            )
        )
    records.extend(
        (
            (f"{prefix}spec:Energy:Value", "Energy value"),
            (f"{prefix}spec:Energy:Units", "Energy units"),
            (f"{prefix}spec:UB_matrix:Value", "UB matrix"),
        )
    )
    for group, description in (
        ("PrimaryBeamDirection", "Primary beam"),
        ("InplaneReferenceDirection", "In-plane reference"),
        ("SampleSurfaceNormalDirection", "Sample surface normal"),
    ):
        for index in range(1, 4):
            records.append(
                (f"{prefix}{group}:AxisNumber{index}", f"{description} {index}")
            )
    records.extend(
        (
            (f"{prefix}DetectorSetup:PixelDirection1", "Detector pixel direction 1"),
            (f"{prefix}DetectorSetup:PixelDirection2", "Detector pixel direction 2"),
            (f"{prefix}DetectorSetup:CenterChannelPixel", "Detector center"),
            (f"{prefix}DetectorSetup:Size", "Detector size"),
            (f"{prefix}DetectorSetup:Distance", "Detector distance"),
            (f"{prefix}DetectorSetup:Units", "Detector units"),
            (f"{prefix}ScanOn:Value", "Scan on flag"),
            (f"{prefix}FilePath:Value", "File path"),
            (f"{prefix}FileName:Value", "File name"),
        )
    )
    return records


def _run_ioc(raw_config: Mapping[str, Any]) -> None:
    """Run the GUI-free pvAccess IOC in its dedicated process."""
    import ctypes.util

    import numpy as np
    import pvaccess as pva
    from epics import PV as EpicsPV

    profile = profile_from_raw(raw_config)
    current_values: dict[str, Any] = {}

    def ioc_put(ca_ioc, record: str, value: Any) -> None:
        if isinstance(value, (list, tuple, np.ndarray)):
            converted = [
                float(item) if isinstance(item, (int, float, np.floating)) else str(item)
                for item in value
            ]
        elif isinstance(value, bool):
            converted = int(value)
        else:
            converted = value
        current_values[record] = converted
        try:
            ca_ioc.putField(record, converted)
        except Exception as exc:
            print(f"IOC put [{record}]: {exc}", flush=True)

    if not os.environ.get("EPICS_DB_INCLUDE_PATH"):
        library = ctypes.util.find_library("pvData")
        if library:
            library = os.path.realpath(library)
            dbd = os.path.realpath(os.path.join(os.path.dirname(library), "../../dbd"))
        elif os.environ.get("EPICS_BASE"):
            dbd = os.path.join(os.environ["EPICS_BASE"], "dbd")
        else:
            dbd = os.path.join(os.path.dirname(pva.__file__), "dbd")
            if not os.path.isdir(dbd):
                raise RuntimeError(
                    "Cannot find dbd directory. Set EPICS_DB_INCLUDE_PATH."
                )
        os.environ["EPICS_DB_INCLUDE_PATH"] = dbd

    base_dbd = os.path.join(os.environ["EPICS_DB_INCLUDE_PATH"], "base.dbd")
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".db") as stream:
        stream.write(build_ioc_database(profile))
        database_path = stream.name
    try:
        ca_ioc = pva.CaIoc()
        ca_ioc.loadDatabase(base_dbd, "", "")
        ca_ioc.registerRecordDeviceDriver()
        ca_ioc.loadRecords(database_path, "")
        ca_ioc.start()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(database_path)

    static_values = static_ioc_values(profile)
    for record, value in static_values.items():
        ioc_put(ca_ioc, record, value)

    stop_event = threading.Event()
    pv_monitors: dict[str, Any] = {}
    pv_monitor_lock = threading.Lock()
    unavailable_sources: set[str] = set()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    def source_value(source: str) -> float:
        source = source.strip()
        try:
            return float(source)
        except ValueError:
            pass
        with pv_monitor_lock:
            monitor = pv_monitors.get(source)
            if monitor is None:
                monitor = EpicsPV(source, auto_monitor=True)
                pv_monitors[source] = monitor
            value = monitor.value if monitor.connected else None
        try:
            if value is None:
                raise ValueError("no connection or value")
            result = float(value)
            if not np.isfinite(result):
                raise ValueError("non-finite value")
        except (TypeError, ValueError) as exc:
            if source not in unavailable_sources:
                unavailable_sources.add(source)
                print(f"[IOC] source unavailable: {source!r} ({exc})", flush=True)
            return float("nan")
        if source in unavailable_sources:
            unavailable_sources.remove(source)
            print(f"[IOC] source recovered: {source!r}", flush=True)
        return result

    print(
        f"IOC ready (prefix={profile.prefix!r}, axes={len(profile.axes)})",
        flush=True,
    )
    print(json.dumps({"type": "values", "data": dict(current_values)}), flush=True)
    loop_count = 0
    while not stop_event.is_set():
        started = time.monotonic()
        try:
            for axis in profile.axes:
                ioc_put(
                    ca_ioc,
                    f"{profile.prefix}{axis.record_name}:Position",
                    source_value(axis.source_pv),
                )
            ioc_put(
                ca_ioc,
                f"{profile.prefix}spec:Energy:Value",
                source_value(profile.energy_source_pv),
            )
            loop_count += 1
            if loop_count % app_settings.RSM_IOC_SNAPSHOT_EVERY == 0:
                for record in static_values:
                    try:
                        current_values[record] = ca_ioc.getField(record)
                    except Exception as exc:
                        print(f"IOC read [{record}]: {exc}", flush=True)
                print(
                    json.dumps({"type": "values", "data": dict(current_values)}),
                    flush=True,
                )
        except Exception as exc:
            print(f"IOC update error: {exc}", flush=True)
        stop_event.wait(
            max(
                0.0,
                app_settings.RSM_IOC_POLL_INTERVAL_SECONDS
                - (time.monotonic() - started),
            )
        )
    print("IOC subprocess exiting.", flush=True)


def _run_gui(
    session: RSMParameterEditSession,
    initial_profile: RSMParameterProfile,
    restart_ioc,
    pv_values: dict[str, Any],
    pv_lock: threading.Lock,
) -> None:
    from PyQt5.QtCore import QSettings, Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    from dashpva.gui import configure_app
    from dashpva.viewer.core.base_window import BaseWindow

    def format_value(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, list):
            return "[" + ", ".join(
                f"{item:.6g}" if isinstance(item, float) else str(item)
                for item in value
            ) + "]"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    class PollWorker(QThread):
        results_ready = pyqtSignal(list)

        def __init__(self, records: list[tuple[str, str]]):
            super().__init__()
            self.records = records
            self.running = True

        def run(self):
            previous: list[str] | None = None
            while self.running:
                with pv_lock:
                    snapshot = dict(pv_values)
                values = [
                    format_value(snapshot.get(name)) for name, _ in self.records
                ]
                if values != previous:
                    previous = values
                    self.results_ready.emit(values)
                self.msleep(50)

        def stop(self):
            self.running = False

    class AxisTable(QWidget):
        headers = (
            "Label",
            "SPEC motor name",
            "Record name",
            "Source PV / static",
            "Direction",
            "Units",
        )
        axis_keys = (
            "LABEL",
            "SPEC_MOTOR_NAME",
            "RECORD_NAME",
            "SOURCE_PV",
            "DIRECTION",
            "ANGLE_UNITS",
        )

        def __init__(self, role: str):
            super().__init__()
            self.role = role
            self._next_new_origin = -1
            layout = QVBoxLayout(self)
            self.table = QTableWidget(0, len(self.headers))
            self.table.setHorizontalHeaderLabels(self.headers)
            self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            layout.addWidget(self.table)
            controls = QHBoxLayout()
            for label, callback in (
                ("Add", self.add_axis),
                ("Remove", self.remove_axis),
                ("Move up", lambda: self.move_axis(-1)),
                ("Move down", lambda: self.move_axis(1)),
            ):
                button = QPushButton(label)
                button.clicked.connect(callback)
                controls.addWidget(button)
            controls.addStretch()
            layout.addLayout(controls)

        def load_axes(self, axes) -> None:
            self.table.setRowCount(0)
            self._next_new_origin = -1
            for origin, axis in enumerate(axes):
                self._append(axis.as_mapping(), origin)

        def _append(
            self,
            values: Mapping[str, Any],
            origin: int | None = None,
            row: int | None = None,
        ) -> None:
            if origin is None:
                origin = self._next_new_origin
                self._next_new_origin -= 1
            row = self.table.rowCount() if row is None else row
            self.table.insertRow(row)
            for column, key in enumerate(self.axis_keys):
                item = QTableWidgetItem(str(values.get(key, "")))
                item.setData(Qt.UserRole, origin)
                self.table.setItem(row, column, item)
            self.table.setCurrentCell(row, 0)

        def add_axis(self) -> None:
            ordinal = self.table.rowCount() + 1
            stem = f"{self.role.title()}Axis{ordinal}"
            self._append(
                {
                    "LABEL": stem,
                    "SPEC_MOTOR_NAME": "",
                    "RECORD_NAME": stem,
                    "SOURCE_PV": "0",
                    "DIRECTION": "z-",
                    "ANGLE_UNITS": "deg",
                }
            )

        def remove_axis(self) -> None:
            row = self.table.currentRow()
            if row >= 0:
                self.table.removeRow(row)

        def move_axis(self, offset: int) -> None:
            source_row = self.table.currentRow()
            target_row = source_row + offset
            if source_row < 0 or not 0 <= target_row < self.table.rowCount():
                return
            row_values = [
                self.table.takeItem(source_row, column)
                for column in range(self.table.columnCount())
            ]
            self.table.removeRow(source_row)
            self.table.insertRow(target_row)
            for column, item in enumerate(row_values):
                self.table.setItem(target_row, column, item)
            self.table.setCurrentCell(target_row, 0)

        def values(self) -> list[dict[str, str]]:
            axes = []
            for row in range(self.table.rowCount()):
                axes.append(
                    {
                        key: (self.table.item(row, column).text().strip()
                              if self.table.item(row, column) is not None else "")
                        for column, key in enumerate(self.axis_keys)
                    }
                )
            return axes

        def origins(self) -> tuple[int | None, ...]:
            return tuple(
                self.table.item(row, 0).data(Qt.UserRole)
                if self.table.item(row, 0) is not None
                else None
                for row in range(self.table.rowCount())
            )

        def row_for_origin(self, origin: int) -> int | None:
            for row, candidate in enumerate(self.origins()):
                if candidate == origin:
                    return row
            return None

        def restore_axis(self, origin: int, values: Mapping[str, Any]) -> None:
            rows = list(zip(self.origins(), self.values()))
            restored = _restore_axis_row(rows, origin, values)
            if len(restored) == len(rows):
                return
            self.table.setRowCount(0)
            for item_origin, item_values in restored:
                self._append(item_values, item_origin)

        def remove_origin(self, origin: int) -> None:
            row = self.row_for_origin(origin)
            if row is not None:
                self.table.removeRow(row)

        def replace_axis(self, origin: int, values: Mapping[str, Any]) -> None:
            row = self.row_for_origin(origin)
            if row is None:
                self.restore_axis(origin, values)
                return
            for column, key in enumerate(self.axis_keys):
                item = QTableWidgetItem(str(values.get(key, "")))
                item.setData(Qt.UserRole, origin)
                self.table.setItem(row, column, item)

        def reorder_loaded(self, order: list[int]) -> None:
            rows = list(zip(self.origins(), self.values()))
            new_rows = _reorder_loaded_axis_rows(rows, order)
            self.table.setRowCount(0)
            for origin, values in new_rows:
                self._append(values, origin)

    class SimulatorWindow(BaseWindow):
        def __init__(self, profile: RSMParameterProfile):
            super().__init__(
                viewer_name="RSM Parameter IOC", visible_actions=["Documentation"]
            )
            self.profile = profile
            self.worker: PollWorker | None = None
            self.setWindowTitle("RSM Parameter IOC")
            self._build_ui()
            self._load_profile(profile)
            self._reset_record_monitor(profile)
            settings = QSettings("DashPVA", "RSMParameterIOC")
            geometry = settings.value("window_geom")
            if geometry:
                self.restoreGeometry(geometry)

        def _build_ui(self) -> None:
            central = QWidget()
            central_layout = QVBoxLayout(central)
            central_layout.setContentsMargins(0, 0, 0, 0)
            self.setCentralWidget(central)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            central_layout.addWidget(scroll)
            content = QWidget()
            root = QVBoxLayout(content)
            scroll.setWidget(content)

            self.profile_notice = QLabel()
            self.profile_notice.setWordWrap(True)
            self.profile_notice.setProperty("messageLevel", "warning")
            root.addWidget(self.profile_notice)

            general = QGroupBox("Profile-backed IOC settings")
            form = QFormLayout(general)
            self.prefix_edit = QLineEdit()
            form.addRow("IOC prefix", self.prefix_edit)
            self.energy_edit = QLineEdit()
            form.addRow("Energy source PV / static", self.energy_edit)
            self.energy_units = QComboBox()
            self.energy_units.setEditable(True)
            self.energy_units.addItem("keV")
            self.energy_units.setToolTip("RSM energy is configured in keV.")
            form.addRow("Energy units", self.energy_units)
            root.addWidget(general)

            sample_group = QGroupBox("Ordered sample circles")
            sample_layout = QVBoxLayout(sample_group)
            self.sample_table = AxisTable("sample")
            sample_layout.addWidget(self.sample_table)
            root.addWidget(sample_group)

            detector_group = QGroupBox("Ordered detector circles")
            detector_layout = QVBoxLayout(detector_group)
            self.detector_table = AxisTable("detector")
            detector_layout.addWidget(self.detector_table)
            root.addWidget(detector_group)

            root.addWidget(self._build_calibration_group())
            root.addWidget(self._build_advanced_group())
            root.addWidget(self._build_records_group())

            controls = QHBoxLayout()
            reload_button = QPushButton("Reload profile")
            reload_button.clicked.connect(self._reload)
            apply_button = QPushButton("Apply && Save")
            apply_button.clicked.connect(self._apply)
            self.retry_button = QPushButton("Retry IOC sync")
            self.retry_button.clicked.connect(self._retry_sync)
            self.retry_button.setVisible(False)
            controls.addWidget(reload_button)
            controls.addStretch()
            controls.addWidget(self.retry_button)
            controls.addWidget(apply_button)
            # Outside the scroll area so it stays visible without scrolling.
            central_layout.addLayout(controls)

            self.resize(1000, 1000)

        def _build_calibration_group(self) -> QGroupBox:
            group = QGroupBox(
                "Static geometry — JSON; DETECTOR_SETUP accepts PIXEL_SIZE, "
                "DETECTOR_SHAPE, ROI, BINNING, DETROT/TILT/TILTAZIMUTH and per-field units"
            )
            form = QFormLayout(group)
            self.calibration_values: dict[str, QLineEdit] = {}
            for key in (
                "UB_MATRIX",
                "PRIMARY_BEAM_DIRECTION",
                "INPLANE_REFERENCE_DIRECTION",
                "SAMPLE_SURFACE_NORMAL_DIRECTION",
                "DETECTOR_SETUP",
            ):
                edit = QLineEdit()
                self.calibration_values[key] = edit
                form.addRow(key.replace("_", " ").title(), edit)
            return group

        def _build_advanced_group(self) -> QGroupBox:
            group = QGroupBox("Advanced")
            form = QFormLayout(group)
            self.sample_orientation = QComboBox()
            self.sample_orientation.setEditable(True)
            self.sample_orientation.addItems(
                ("det", "sam", "x+", "x-", "y+", "y-", "z+", "z-")
            )
            form.addRow("Sample orientation", self.sample_orientation)
            return group

        def _calibration_edits(self) -> dict[str, Any]:
            """Parse the calibration fields, raising on the first bad entry.

            Validation proper happens in normalize_parameters; this only turns
            the text back into JSON so a typo is reported as a typo instead of
            surfacing later as an unrelated geometry error.
            """
            parsed: dict[str, Any] = {}
            for key, edit in self.calibration_values.items():
                text = edit.text().strip()
                if not text:
                    continue
                try:
                    parsed[key] = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{key} is not valid JSON: {exc}") from exc
            return parsed

        def _build_records_group(self) -> QGroupBox:
            group = QGroupBox("Live IOC records")
            layout = QVBoxLayout(group)
            self.records_table = QTableWidget(0, 2)
            self.records_table.setHorizontalHeaderLabels(("PV name", "Value"))
            self.records_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.records_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.records_table.setMinimumHeight(320)
            layout.addWidget(self.records_table)
            return group

        def _load_profile(self, profile: RSMParameterProfile) -> None:
            self.profile = profile
            raw_parameters = session.raw.get("IOC_RSM_PARAMETER")
            self._raw_baseline = copy.deepcopy(
                raw_parameters if isinstance(raw_parameters, Mapping) else {}
            )
            self._normalized_baseline = profile.parameter_mapping()
            self.prefix_edit.setText(profile.prefix)
            self.energy_edit.setText(profile.energy_source_pv)
            self.energy_units.setCurrentText(profile.energy_units)
            self.sample_orientation.setCurrentText(profile.sample_orientation)
            self.sample_table.load_axes(profile.sample_axes)
            self.detector_table.load_axes(profile.detector_axes)
            mapping = profile.parameter_mapping()
            for key, edit in self.calibration_values.items():
                edit.setText(json.dumps(mapping[key], sort_keys=True))
            if app_settings.CONFIG_ERROR:
                self.profile_notice.setText(
                    "The active profile failed to resolve and is being shown/edited as a "
                    f"fallback so it can be repaired here: {app_settings.CONFIG_ERROR}"
                )
                self.profile_notice.setProperty("messageLevel", "error")
            elif session.has_canonical_parameters and not requires_adoption_confirmation(session.raw):
                self.profile_notice.setText("Edits are staged until Apply & Save succeeds.")
                self.profile_notice.setProperty("messageLevel", "info")
            else:
                self.profile_notice.setText(
                    "This profile is not fully canonical. First Apply & Save will show "
                    "the effective HKL and static-geometry changes for confirmation."
                )
                self.profile_notice.setProperty("messageLevel", "warning")
            self.profile_notice.style().unpolish(self.profile_notice)
            self.profile_notice.style().polish(self.profile_notice)

        def _parameters(self) -> dict[str, Any]:
            current = self.profile.parameter_mapping()
            current.update(
                {
                    "SAMPLE_AXES": self.sample_table.values(),
                    "DETECTOR_AXES": self.detector_table.values(),
                    "ENERGY_SOURCE_PV": self.energy_edit.text().strip(),
                    "ENERGY_UNITS": self.energy_units.currentText().strip(),
                    "SAMPLE_ORIENTATION": self.sample_orientation.currentText().strip(),
                }
            )
            current.update(self._calibration_edits())
            return current

        def _axis_origins(self) -> dict[str, tuple[int | None, ...]]:
            return {
                "SAMPLE_AXES": self.sample_table.origins(),
                "DETECTOR_AXES": self.detector_table.origins(),
            }

        def _pending_change(self) -> dict[str, Any]:
            """Build the exact candidate used by Apply, close, and review."""
            form_parameters = self._parameters()
            parameters = copy.deepcopy(form_parameters)
            validate_parameter_profile(self.prefix_edit.text(), parameters)
            with pv_lock:
                live = dict(pv_values)
            origins = self._axis_origins()
            adopted, conflicts = merge_live_records(
                parameters,
                self.profile,
                live,
                self._raw_baseline,
                self._normalized_baseline,
                axis_origins=origins,
            )
            candidate = validate_parameter_profile(
                self.prefix_edit.text(), parameters
            )
            replacement = update_raw_profile(
                session.raw,
                self.prefix_edit.text(),
                parameters,
                axis_origins=origins,
            )
            return {
                "form_parameters": form_parameters,
                "parameters": parameters,
                "candidate": candidate,
                "origins": origins,
                "adopted": adopted,
                "conflicts": conflicts,
                "replacement": replacement,
            }

        def _confirm_adoption(self, parameters: Mapping[str, Any]) -> bool:
            if not requires_adoption_confirmation(session.raw):
                return True
            try:
                details = adoption_diff(session.raw, self.prefix_edit.text(), parameters)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid RSM configuration", str(exc))
                return False
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Confirm canonical RSM adoption")
            box.setText(
                "Applying will make IOC_RSM_PARAMETER authoritative for HKL channels "
                "and static geometry. Verify the diff before continuing."
            )
            box.setDetailedText(details)
            box.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Cancel)
            return box.exec_() == QMessageBox.Save

        def _confirm_live_adoption(self, adopted: list[str]) -> bool:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Records changed on the IOC")
            box.setText(
                "These IOC records changed since this profile was loaded. Save "
                "to adopt the live values, or Cancel to leave the profile untouched."
            )
            box.setDetailedText("\n".join(adopted))
            box.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Cancel)
            return box.exec_() == QMessageBox.Save

        def _apply(self) -> bool:
            try:
                pending = self._pending_change()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid RSM configuration", str(exc))
                return False
            if pending["conflicts"]:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setWindowTitle("Conflicting IOC changes")
                box.setText(
                    "The same settings changed both in this editor and on the IOC. "
                    "Nothing was saved. Reconcile the values and try again."
                )
                box.setDetailedText("\n".join(pending["conflicts"]))
                box.exec_()
                return False
            if pending["adopted"] and not self._confirm_live_adoption(
                pending["adopted"]
            ):
                return False
            parameters = pending["parameters"]
            if not self._confirm_adoption(parameters):
                return False
            if self.sample_orientation.currentText().strip().lower() == "sam":
                answer = QMessageBox.warning(
                    self,
                    "Check sample orientation",
                    "SAMPLE_ORIENTATION='sam' is physically correct only when the "
                    "innermost sample circle is the azimuth motor. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    return False
            try:
                result, saved_snapshot = apply_and_activate(
                    session,
                    self.prefix_edit.text(),
                    parameters,
                    self._activate_snapshot,
                    axis_origins=pending["origins"],
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid RSM configuration", str(exc))
                return False
            except SnapshotActivationError as exc:
                saved_profile = profile_from_raw(exc.snapshot)
                self._load_profile(saved_profile)
                self._reset_record_monitor(saved_profile)
                self._mark_out_of_sync(exc.snapshot, exc.error)
                return False
            if result.status.value == "conflict":
                QMessageBox.warning(
                    self,
                    "Profile changed",
                    "Another editor changed this profile. Nothing was saved or applied. "
                    "Reload the profile and reapply your edits.",
                )
                return False
            if not result.saved or saved_snapshot is None:
                QMessageBox.critical(
                    self,
                    "Profile save failed",
                    result.error or "The profile could not be saved. The IOC was not restarted.",
                )
                return False
            saved_profile = profile_from_raw(saved_snapshot)
            self._load_profile(saved_profile)
            self._reset_record_monitor(saved_profile)
            self.statusBar().showMessage("Profile saved atomically; IOC restarted from that snapshot")
            return True

        def _reload(self) -> None:
            answer = QMessageBox.question(
                self,
                "Discard staged edits?",
                "Reloading discards edits that have not been applied.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            try:
                profile = session.load()
            except Exception as exc:
                QMessageBox.critical(self, "Profile reload failed", str(exc))
                return
            self._load_profile(profile)
            self._reset_record_monitor(profile)
            try:
                self._activate_snapshot(session.raw)
            except Exception as exc:
                self._mark_out_of_sync(session.raw, exc)
                return
            self.statusBar().showMessage("Profile and IOC reloaded from the same snapshot")

        def _activate_snapshot(self, snapshot: Mapping[str, Any]) -> None:
            app_settings.reload()
            if app_settings.RAW_CONFIG != snapshot:
                raise RuntimeError(
                    "central settings did not reload the saved active-profile snapshot"
                )
            restart_ioc(snapshot)
            self.pending_snapshot = None
            self.retry_button.setVisible(False)

        def _mark_out_of_sync(self, snapshot: Mapping[str, Any], error: Exception) -> None:
            self.pending_snapshot = dict(snapshot)
            self.retry_button.setVisible(True)
            self.profile_notice.setText(
                "OUT OF SYNC: the profile is saved, but the IOC did not activate that "
                f"snapshot ({error}). Use Retry IOC sync; no profile rollback occurred."
            )
            self.profile_notice.setProperty("messageLevel", "error")
            self.profile_notice.style().unpolish(self.profile_notice)
            self.profile_notice.style().polish(self.profile_notice)
            QMessageBox.critical(
                self,
                "IOC out of sync",
                "The profile snapshot is saved, but IOC activation failed. "
                "No rollback was attempted. Use Retry IOC sync.\n\n"
                f"{error}",
            )

        def _retry_sync(self) -> None:
            snapshot = getattr(self, "pending_snapshot", None)
            if snapshot is None:
                return
            try:
                self._activate_snapshot(snapshot)
            except Exception as exc:
                self._mark_out_of_sync(snapshot, exc)
                return
            self._load_profile(profile_from_raw(snapshot))
            self.statusBar().showMessage("IOC synchronized with the saved profile snapshot")

        def _stop_worker(self) -> None:
            if self.worker is not None:
                self.worker.stop()
                self.worker.wait(2000)
                self.worker = None

        def _reset_record_monitor(self, profile: RSMParameterProfile) -> None:
            self._stop_worker()
            records = all_pv_names(profile)
            self.records_table.setRowCount(len(records))
            self.value_items: list[QTableWidgetItem] = []
            for row, (name, _) in enumerate(records):
                self.records_table.setItem(row, 0, QTableWidgetItem(name))
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.records_table.setItem(row, 1, item)
                self.value_items.append(item)
            self.worker = PollWorker(records)
            self.worker.results_ready.connect(self._apply_results)
            self.worker.start()

        def _apply_results(self, values: list[str]) -> None:
            for item, value in zip(self.value_items, values):
                item.setText(value)
            self.statusBar().showMessage(f"Last IOC update: {time.strftime('%H:%M:%S')}")

        def has_unsaved_changes(self) -> bool:
            return _has_pending_rsm_change(self._pending_change, session.raw)

        def save_changes(self) -> bool:
            return self._apply()

        def _change_target(self, key: str):
            simple = {
                "PREFIX": self.prefix_edit,
                "ENERGY_SOURCE_PV": self.energy_edit,
                "ENERGY_UNITS": self.energy_units,
                "SAMPLE_ORIENTATION": self.sample_orientation,
            }
            if key in simple:
                return simple[key]
            if key in self.calibration_values:
                return self.calibration_values[key]
            return None

        def _axis_target(self, key: str):
            parts = key.split(".")
            if len(parts) != 3 or parts[0] not in (
                "SAMPLE_AXES",
                "DETECTOR_AXES",
            ):
                return None
            table = (
                self.sample_table
                if parts[0] == "SAMPLE_AXES"
                else self.detector_table
            )
            if parts[2] not in AxisTable.axis_keys:
                return None
            if parts[1].startswith("@"):
                try:
                    row = table.row_for_origin(int(parts[1][1:]))
                except ValueError:
                    return None
                if row is None:
                    return None
            elif parts[1].isdigit():
                row = int(parts[1])
            else:
                return None
            if row >= table.table.rowCount():
                return None
            return table, row, AxisTable.axis_keys.index(parts[2])

        def is_change_editable(self, key: str) -> bool:
            if key in getattr(self, "_live_only_review_keys", set()):
                return False
            parts = key.split(".")
            if len(parts) == 2 and parts[0] in (
                "SAMPLE_AXES",
                "DETECTOR_AXES",
            ):
                if parts[1] == "ORDER":
                    return True
                if parts[1].startswith("@"):
                    try:
                        int(parts[1][1:])
                    except ValueError:
                        return False
                    return True
            return (
                self._change_target(key) is not None
                or self._axis_target(key) is not None
            )

        def unsaved_changes_rows(self) -> list[tuple[str, str, str, str]]:
            pending, invalid_rows = _review_pending_change(self._pending_change)
            if pending is None:
                return invalid_rows
            baseline_parameters = copy.deepcopy(self._normalized_baseline)
            current_parameters = copy.deepcopy(pending["parameters"])
            form_parameters = copy.deepcopy(pending["form_parameters"])
            for axis_key in ("SAMPLE_AXES", "DETECTOR_AXES"):
                baseline_parameters.pop(axis_key, None)
                current_parameters.pop(axis_key, None)
                form_parameters.pop(axis_key, None)
            baseline = _flatten_config(baseline_parameters, index_lists=True)
            current = _flatten_config(current_parameters, index_lists=True)
            form = _flatten_config(form_parameters, index_lists=True)
            rows: list[tuple[str, str, str, str]] = []
            self._live_only_review_keys: set[str] = set()
            if self.prefix_edit.text() != self.profile.prefix:
                rows.append(
                    (
                        "change",
                        "PREFIX",
                        self.profile.prefix,
                        self.prefix_edit.text(),
                    )
                )
            for key in sorted(set(baseline) & set(current)):
                if baseline[key] != current[key]:
                    rows.append(
                        ("change", key, str(baseline[key]), str(current[key]))
                    )
                    if form.get(key) == baseline[key]:
                        self._live_only_review_keys.add(key)
            for key in sorted(set(current) - set(baseline)):
                rows.append(("add", key, "", str(current[key])))
            for key in sorted(set(baseline) - set(current)):
                rows.append(("remove", key, str(baseline[key]), ""))
            for axis_key, table in (
                ("SAMPLE_AXES", self.sample_table),
                ("DETECTOR_AXES", self.detector_table),
            ):
                baseline_axes = self._normalized_baseline[axis_key]
                current_axes = pending["parameters"][axis_key]
                form_axes = pending["form_parameters"][axis_key]
                origins = table.origins()
                loaded_origins = [
                    origin
                    for origin in origins
                    if origin is not None and origin >= 0
                ]
                if (
                    set(loaded_origins) == set(range(len(baseline_axes)))
                    and loaded_origins != list(range(len(baseline_axes)))
                ):
                    rows.append(
                        (
                            "change",
                            f"{axis_key}.ORDER",
                            json.dumps(list(range(len(baseline_axes)))),
                            json.dumps(loaded_origins),
                        )
                    )
                for row, origin in enumerate(origins):
                    if origin is None:
                        continue
                    if origin < 0:
                        rows.append(
                            (
                                "add",
                                f"{axis_key}.@{origin}",
                                "",
                                json.dumps(current_axes[row], sort_keys=True),
                            )
                        )
                        continue
                    baseline_axis = baseline_axes[origin]
                    for field in AxisTable.axis_keys:
                        old = baseline_axis[field]
                        new = current_axes[row][field]
                        if old == new:
                            continue
                        key = f"{axis_key}.@{origin}.{field}"
                        rows.append(("change", key, str(old), str(new)))
                        if form_axes[row][field] == old:
                            self._live_only_review_keys.add(key)
                present = {origin for origin in origins if origin is not None}
                for origin, axis in enumerate(baseline_axes):
                    if origin not in present:
                        rows.append(
                            (
                                "remove",
                                f"{axis_key}.@{origin}",
                                json.dumps(axis, sort_keys=True),
                                "",
                            )
                        )
            return rows

        def apply_change_decisions(self, kept: list, dropped: list) -> None:
            for _kind, key, old, _new in dropped:
                self._write_change(key, old)
            for kind, key, _old, new in kept:
                if kind != "remove":
                    self._write_change(key, new)

        def _write_change(self, key: str, value: str) -> None:
            parts = key.split(".")
            if len(parts) == 2 and parts[0] in (
                "SAMPLE_AXES",
                "DETECTOR_AXES",
            ):
                table = (
                    self.sample_table
                    if parts[0] == "SAMPLE_AXES"
                    else self.detector_table
                )
                if parts[1] == "ORDER":
                    try:
                        table.reorder_loaded([int(item) for item in json.loads(value)])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        return
                    return
                if parts[1].startswith("@"):
                    try:
                        origin = int(parts[1][1:])
                    except ValueError:
                        return
                    if not value:
                        table.remove_origin(origin)
                        return
                    try:
                        values = json.loads(value)
                    except json.JSONDecodeError:
                        return
                    if isinstance(values, Mapping):
                        table.replace_axis(origin, values)
                    return
            axis = self._axis_target(key)
            if axis is not None:
                table, row, column = axis
                item = table.table.item(row, column)
                origin = item.data(Qt.UserRole) if item is not None else None
                replacement = QTableWidgetItem(value)
                replacement.setData(Qt.UserRole, origin)
                table.table.setItem(row, column, replacement)
                return
            target = self._change_target(key)
            if target is None:
                return
            if isinstance(target, QComboBox):
                target.setCurrentText(value)
            else:
                target.setText(value)

        def unsaved_changes_text(self) -> str:
            return (
                "This editor has staged form or live IOC changes that have not "
                "been saved to the profile."
            )

        def closeEvent(self, event) -> None:
            if not self.confirm_close(event):
                return
            self._stop_worker()
            settings = QSettings("DashPVA", "RSMParameterIOC")
            settings.setValue("window_geom", self.saveGeometry())
            super().closeEvent(event)

    app = QApplication(sys.argv)
    configure_app(app)
    window = SimulatorWindow(initial_profile)
    window.show()
    sys.exit(app.exec_())


def _active_session():
    from dashpva.utils.config.source import ConfigSource

    return RSMParameterEditSession(ConfigSource(app_settings.LOCATOR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile-driven RSM parameter IOC")
    parser.add_argument("--ioc-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prefix", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prefix is not None:
        parser.error(
            "--prefix is no longer an override; set root IOC_PREFIX in the active "
            "Workflow profile"
        )

    if args.ioc_mode:
        _run_ioc(_active_session().raw)
        return

    try:
        session = _active_session()
        profile = profile_from_raw(session.raw)
    except Exception as exc:
        from PyQt5.QtWidgets import QApplication, QMessageBox

        _app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.warning(
            None,
            "RSM profile unavailable",
            f"{exc}\n\nSelect a valid profile in the Workflow dialog and try again.",
        )
        raise SystemExit(2) from exc

    pv_values: dict[str, Any] = {}
    pv_lock = threading.Lock()
    process_holder: list[subprocess.Popen | None] = [None]
    ready = threading.Event()

    def launch_ioc(snapshot: Mapping[str, Any]) -> subprocess.Popen:
        # The child re-reads the profile from the database rather than being
        # handed a serialized copy. Activation only happens after a successful
        # compare-and-swap save, so the database already *is* the snapshot --
        # and per AGENTS.md the DB is the config source, with TOML reserved for
        # import/export. A side-channel JSON file would be a third format.
        del snapshot
        process = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--ioc-mode",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        process_holder[0] = process

        def forward_output() -> None:
            if process.stdout is None:
                return
            for raw_line in process.stdout:
                text = raw_line.decode(errors="replace").strip()
                try:
                    message = json.loads(text)
                    if message.get("type") == "values":
                        with pv_lock:
                            pv_values.clear()
                            pv_values.update(message["data"])
                        ready.set()
                        continue
                except (json.JSONDecodeError, AttributeError):
                    pass
                print(text, flush=True)

        threading.Thread(target=forward_output, daemon=True).start()
        return process

    def stop_ioc() -> None:
        process = process_holder[0]
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        process_holder[0] = None

    def restart_ioc(snapshot: Mapping[str, Any]) -> None:
        stop_ioc()
        with pv_lock:
            pv_values.clear()
        ready.clear()
        launch_ioc(snapshot)
        if not ready.wait(timeout=15):
            raise RuntimeError("IOC did not publish a snapshot within 15 seconds")

    launch_ioc(session.raw)
    if not ready.wait(timeout=15):
        print("Warning: IOC did not respond within 15 seconds.", flush=True)

    try:
        _run_gui(session, profile, restart_ioc, pv_values, pv_lock)
    finally:
        stop_ioc()


if __name__ == "__main__":
    main()
