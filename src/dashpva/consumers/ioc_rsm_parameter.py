#!/usr/bin/env python3
# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""
6-ID-B RSM Data Simulator

Two-process architecture (prevents core dumps from mixing pvaccess + PyQt5):
  Main process  — PyQt5 GUI; sends config changes to IOC via stdin JSON pipe
  Child process — headless CaIoc + polling loop; reads config from stdin

Source PV input behaviour:
  - If the input can be parsed as a float  → used directly as a static value
  - Otherwise                              → treated as a PV name (caget)

Usage:
  python3 6id_sim_rsm_data.py [--prefix PREFIX]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

from dashpva.consumers import ioc_rsm_store

# ─────────────────────────────────────────────────────────────────────────────
# Shared configuration
# ─────────────────────────────────────────────────────────────────────────────
# The IOC prefix is per-profile app config (settings.IOC_PREFIX) — the single
# source of truth, same as every other PV-prefixed tool in the app. The axis
# list and UB/beam/detector defaults live in the active DB profile too (see
# ioc_rsm_store.py) so both follow the profile selected elsewhere in the app.

DEFAULT_PREFIX = ''

POLL_INTERVAL    = 0.01   # IOC publish rate  (100 Hz)
CAGET_INTERVAL   = 0.02   # PV-source refresh (50 Hz)
SNAPSHOT_EVERY   = 5      # emit GUI snapshot every N IOC cycles (~20 Hz)


# ─────────────────────────────────────────────────────────────────────────────
# IOC subprocess  (pvaccess only — NO PyQt5)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ioc(prefix: str, config: ioc_rsm_store.IOCRSMConfig) -> None:
    # Duplicate stdin fd NOW, before pvaccess/EPICS init may close or redirect fd 0
    _cmd_fd = os.dup(0)

    import ctypes.util
    import tempfile

    import numpy as np
    import pvaccess as pva
    from epics import PV as _PV
    from epics import caget as _caget

    # ── DB builder ────────────────────────────────────────────────────────
    def _ai(name):
        return (f'record(ai, "{name}") {{\n'
                f'  field(DTYP, "Soft Channel")\n  field(PREC, "6")\n}}\n')

    def _so(name, val=''):
        return (f'record(stringout, "{name}") {{\n'
                f'  field(DTYP, "Soft Channel")\n  field(VAL, "{val}")\n}}\n')

    def _wf(name, nelm, ftvl='DOUBLE'):
        return (f'record(waveform, "{name}") {{\n'
                f'  field(DTYP, "Soft Channel")\n'
                f'  field(FTVL, "{ftvl}")\n  field(NELM, "{nelm}")\n}}\n')

    def build_db():
        p = prefix
        lines = []
        for ax in config.axes:
            lines.append(_ai(f"{p}{ax.name}:Position"))
            lines.append(_ai(f"{p}{ax.name}:AxisNumber"))
            lines.append(_so(f"{p}{ax.name}:DirectionAxis", ax.direction))
            lines.append(_so(f"{p}{ax.name}:SpecMotorName", ax.name))
        lines.append(_ai(f"{p}Energy:Value"))
        lines.append(_wf(f"{p}UB_matrix:Value", 9))
        for grp in ['PrimaryBeamDirection', 'InplaneReferenceDirection',
                    'SampleSurfaceNormalDirection']:
            for i in [1, 2, 3]:
                lines.append(_ai(f"{p}{grp}:AxisNumber{i}"))
        d = config.detector
        lines.append(_so(f"{p}DetectorSetup:PixelDirection1", d['pixel_dir1']))
        lines.append(_so(f"{p}DetectorSetup:PixelDirection2", d['pixel_dir2']))
        lines.append(_wf(f"{p}DetectorSetup:CenterChannelPixel", 2))
        lines.append(_wf(f"{p}DetectorSetup:Size", 2))
        lines.append(_ai(f"{p}DetectorSetup:Distance"))
        lines.append(_so(f"{p}DetectorSetup:Units", d['units']))
        lines.append(
            f'record(longout, "{p}ScanOn:Value") {{\n  field(DTYP, "Soft Channel")\n}}\n')
        lines.append(_so(f"{p}FilePath:Value"))
        lines.append(_so(f"{p}FileName:Value"))
        return ''.join(lines)

    # ── IOC helpers ───────────────────────────────────────────────────────
    # Tracks every value written to the IOC — used for the GUI snapshot.
    # Also updated by the background caget poll so external CA writes are visible.
    _current_vals = {}

    def ioc_put(caIoc, rec, val):
        # Update _current_vals FIRST so the snapshot always has the value
        # even if caIoc.putField fails.
        if isinstance(val, (list, np.ndarray)):
            converted = [
                float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                for v in val]
            _current_vals[rec] = converted
            try:
                caIoc.putField(rec, converted)
            except Exception as e:
                print(f'IOC put [{rec}]: {e}', flush=True)
        elif isinstance(val, bool):
            _current_vals[rec] = int(val)
            try:
                caIoc.putField(rec, int(val))
            except Exception as e:
                print(f'IOC put [{rec}]: {e}', flush=True)
        else:
            _current_vals[rec] = val
            try:
                caIoc.putField(rec, val)
            except Exception as e:
                print(f'IOC put [{rec}]: {e}', flush=True)

    # ── Local state (updated from stdin pipe) ─────────────────────────────
    _lock = threading.Lock()
    _state = {
        'motor': {ax.name: ax.source_pv for ax in config.axes},
        'energy': config.energy_source_pv,
    }

    # Open from the duplicated fd (line-buffered) so pvaccess closing fd 0
    # does not break our command channel.
    _cmd_pipe = os.fdopen(_cmd_fd, 'r', buffering=1)

    def _stdin_reader():
        for raw in _cmd_pipe:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
                print(f'[IOC] cmd received: {msg}', flush=True)
                with _lock:
                    if msg.get('type') == 'motor':
                        _state['motor'][msg['name']] = msg['value']
                    elif msg.get('type') == 'energy':
                        _state['energy'] = msg['value']
            except Exception as e:
                print(f'IOC stdin error: {e}', flush=True)

    threading.Thread(target=_stdin_reader, daemon=True).start()

    # ── Start IOC ─────────────────────────────────────────────────────────
    if not os.environ.get('EPICS_DB_INCLUDE_PATH'):
        lib = ctypes.util.find_library('pvData')
        if lib:
            lib = os.path.realpath(lib)
            dbd = os.path.realpath(os.path.join(os.path.dirname(lib), '../../dbd'))
        elif os.environ.get('EPICS_BASE'):
            dbd = os.path.join(os.environ['EPICS_BASE'], 'dbd')
        else:
            dbd = os.path.join(os.path.dirname(pva.__file__), 'dbd')
            if not os.path.isdir(dbd):
                raise RuntimeError('Cannot find dbd directory. Please set EPICS_DB_INCLUDE_PATH.')
        os.environ['EPICS_DB_INCLUDE_PATH'] = dbd

    base_dbd = os.path.join(os.environ['EPICS_DB_INCLUDE_PATH'], 'base.dbd')

    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.db')
    tmp.write(build_db())
    tmp.close()

    caIoc = pva.CaIoc()
    caIoc.loadDatabase(base_dbd, '', '')
    caIoc.registerRecordDeviceDriver()
    caIoc.loadRecords(tmp.name, '')
    caIoc.start()
    os.unlink(tmp.name)

    p = prefix
    # Motor per-axis static fields (set in .db but must be in _current_vals too)
    for ax in config.axes:
        ioc_put(caIoc, f"{p}{ax.name}:AxisNumber",    float(ax.axis_number))
        ioc_put(caIoc, f"{p}{ax.name}:DirectionAxis", ax.direction)
        ioc_put(caIoc, f"{p}{ax.name}:SpecMotorName", ax.name)
    ioc_put(caIoc, f"{p}UB_matrix:Value", list(config.ub_matrix))
    for i, v in enumerate(config.primary_beam):
        ioc_put(caIoc, f"{p}PrimaryBeamDirection:AxisNumber{i+1}", float(v))
    for i, v in enumerate(config.inplane_ref):
        ioc_put(caIoc, f"{p}InplaneReferenceDirection:AxisNumber{i+1}", float(v))
    for i, v in enumerate(config.sample_normal):
        ioc_put(caIoc, f"{p}SampleSurfaceNormalDirection:AxisNumber{i+1}", float(v))
    d = config.detector
    ioc_put(caIoc, f"{p}DetectorSetup:PixelDirection1",    d['pixel_dir1'])
    ioc_put(caIoc, f"{p}DetectorSetup:PixelDirection2",    d['pixel_dir2'])
    ioc_put(caIoc, f"{p}DetectorSetup:CenterChannelPixel", [float(v) for v in d['center']])
    ioc_put(caIoc, f"{p}DetectorSetup:Size",               [float(v) for v in d['size']])
    ioc_put(caIoc, f"{p}DetectorSetup:Distance",           float(d['distance']))
    ioc_put(caIoc, f"{p}DetectorSetup:Units",              d['units'])
    ioc_put(caIoc, f"{p}ScanOn:Value",   0)
    ioc_put(caIoc, f"{p}FilePath:Value", '')
    ioc_put(caIoc, f"{p}FileName:Value", '')

    print(f'IOC ready (prefix={prefix})', flush=True)

    # ── PV monitor pool — non-blocking reads at any rate ─────────────────
    # epics.PV with auto_monitor=True: CA delivers updates via callback into
    # memory. _get() is a dict lookup — O(1), safe to call at 100 Hz.
    _stop      = threading.Event()
    _pv_mons   = {}          # pv_name -> epics.PV object
    _pv_mons_lock = threading.Lock()
    _pv_down   = set()       # source PVs currently failing — dedup so we log once
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())

    def _get(src):
        """Return float: static if src parses as float, else latest CA monitor value.
           Empty/blank input means "unused" → 0.0 (no PV lookup, no logging).
           A PV that fails to come up is logged once and treated as 0.0 so one bad
           source does not stall the whole publish loop."""
        src = (src or '').strip()
        if not src:
            return 0.0
        try:
            return float(src)
        except ValueError:
            pass
        with _pv_mons_lock:
            if src not in _pv_mons:
                _pv_mons[src] = _PV(src, auto_monitor=True)
            v = _pv_mons[src].get(use_monitor=True, timeout=0.1)
        try:
            if v is None:
                raise ValueError('no connection / no value')
            result = float(v)
        except (TypeError, ValueError) as e:
            if src not in _pv_down:
                _pv_down.add(src)
                print(f'[IOC] PV not coming up: {src!r} ({e})', flush=True)
            return 0.0
        if src in _pv_down:
            _pv_down.discard(src)
            print(f'[IOC] PV recovered: {src!r}', flush=True)
        return result

    def _status(src):
        """'none' — no source configured; 'down' — source PV not connecting;
           'ok' — static value or a live/connecting PV. Read by the GUI to
           decide between showing 'None', a red '--', or the formatted value."""
        src = (src or '').strip()
        if not src:
            return 'none'
        try:
            float(src)
            return 'ok'
        except ValueError:
            pass
        return 'down' if src in _pv_down else 'ok'

    # ── Background caget poll for static IOC records (2 Hz) ──────────────
    # Uses caget on our own IOC's PV names so external CA writes (caput from
    # scan software, alignment tools, etc.) are reflected in the snapshot.
    _static_pv_names = list(_current_vals.keys())   # all keys populated by startup ioc_puts
    _static_down = set()     # IOC records currently not responding — dedup logging

    def _poll_static():
        while not _stop.is_set():
            for pv_name in _static_pv_names:
                try:
                    v = _caget(pv_name, timeout=0.3)
                    if v is None:
                        if pv_name not in _static_down:
                            _static_down.add(pv_name)
                            print(f'[IOC] PV not coming up: {pv_name!r} (caget timeout)', flush=True)
                        continue
                    if hasattr(v, 'tolist'):          # numpy array → plain list
                        v = v.tolist()
                    elif hasattr(v, 'item'):          # numpy scalar → Python scalar
                        v = v.item()
                    _current_vals[pv_name] = v
                    if pv_name in _static_down:
                        _static_down.discard(pv_name)
                        print(f'[IOC] PV recovered: {pv_name!r}', flush=True)
                except Exception as e:
                    if pv_name not in _static_down:
                        _static_down.add(pv_name)
                        print(f'[IOC] PV not coming up: {pv_name!r} ({e})', flush=True)
            _stop.wait(0.5)   # 2 Hz is plenty for slowly-changing values

    threading.Thread(target=_poll_static, daemon=True).start()

    # ── Publish loop — 100 Hz, reads from cache (no blocking caget) ───────
    _loop_n = 0
    while not _stop.is_set():
        t0 = time.monotonic()
        try:
            with _lock:
                motor_srcs = dict(_state['motor'])
                energy_src = _state['energy']

            motor_vals = {ax.name: _get(motor_srcs[ax.name]) for ax in config.axes}
            energy_val = _get(energy_src)

            for ax in config.axes:
                ioc_put(caIoc, f"{p}{ax.name}:Position", motor_vals[ax.name])
            ioc_put(caIoc, f"{p}Energy:Value", energy_val)

            # Snapshot: _current_vals has every value ever written via ioc_put;
            # overlay the fast-changing motor/energy values on top.
            _loop_n += 1
            if _loop_n % SNAPSHOT_EVERY == 0:
                snap = dict(_current_vals)
                status = {ax.name: _status(motor_srcs[ax.name]) for ax in config.axes}
                status['ENERGY'] = _status(energy_src)
                print(json.dumps({'type': 'values', 'data': snap, 'status': status}), flush=True)

        except Exception as e:
            print(f'Update error: {e}', flush=True)

        elapsed = time.monotonic() - t0
        remaining = POLL_INTERVAL - elapsed
        if remaining > 0:
            _stop.wait(remaining)

    print('IOC subprocess exiting.', flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# GUI  (PyQt5 only — NO pvaccess)
# ─────────────────────────────────────────────────────────────────────────────

def _all_pv_names(prefix: str, config: ioc_rsm_store.IOCRSMConfig) -> list:
    """Return [(pv_name, description, status_key)] for every record the IOC
    publishes. status_key indexes the IOC's per-cycle 'status' snapshot
    (axis name, or 'ENERGY') for the handful of rows backed by a source PV;
    None for everything else (fixed geometry, never source-driven)."""
    p = prefix
    pvs = []
    for ax in config.axes:
        pvs.append((f"{p}{ax.name}:Position",      f"{ax.name} position", ax.name))
        pvs.append((f"{p}{ax.name}:AxisNumber",    f"{ax.name} axis number", None))
        pvs.append((f"{p}{ax.name}:DirectionAxis", f"{ax.name} direction", None))
        pvs.append((f"{p}{ax.name}:SpecMotorName", f"{ax.name} spec name", None))
    pvs.append((f"{p}Energy:Value",         'Energy value (keV)', 'ENERGY'))
    pvs.append((f"{p}UB_matrix:Value",      'UB matrix (9 elements)', None))
    for grp, desc in [
        ('PrimaryBeamDirection',         'Primary beam'),
        ('InplaneReferenceDirection',    'Inplane ref'),
        ('SampleSurfaceNormalDirection', 'Sample normal'),
    ]:
        for i in [1, 2, 3]:
            pvs.append((f"{p}{grp}:AxisNumber{i}", f'{desc} axis {i}', None))
    pvs.append((f"{p}DetectorSetup:PixelDirection1",    'Detector pixel dir 1', None))
    pvs.append((f"{p}DetectorSetup:PixelDirection2",    'Detector pixel dir 2', None))
    pvs.append((f"{p}DetectorSetup:CenterChannelPixel", 'Detector center (px)', None))
    pvs.append((f"{p}DetectorSetup:Size",               'Detector size (mm)', None))
    pvs.append((f"{p}DetectorSetup:Distance",           'Detector distance (mm)', None))
    pvs.append((f"{p}DetectorSetup:Units",              'Detector units', None))
    pvs.append((f"{p}ScanOn:Value",   'Scan on flag', None))
    pvs.append((f"{p}FilePath:Value", 'File path', None))
    pvs.append((f"{p}FileName:Value", 'File name', None))
    return pvs


def _run_gui(prefix: str, config: ioc_rsm_store.IOCRSMConfig, send_cmd, restart_ioc,
             pv_values: dict, pv_status: dict, pv_lock, save_config, reload_config,
             profile_label: str = '') -> None:
    """
    pv_values / pv_status / pv_lock : shared dicts populated by _fwd() in main().
                           pv_status maps axis name (or 'ENERGY') -> 'none' |
                           'down' | 'ok', driving the None/red-'--'/value display
                           for rows backed by a source PV (see _all_pv_names).
    config               : IOCRSMConfig currently active (axes + geometry).
    send_cmd             : live PV-source rebind on the running IOC (no restart).
    restart_ioc          : restart_ioc(new_prefix, new_config) — relaunch the IOC.
    save_config          : save_config(cfg) — persist cfg to the active profile.
    reload_config        : reload_config() -> (IOCRSMConfig, profile_label).
    profile_label        : name of the profile/TOML the config was loaded from,
                           so the user can see where it's coming from.
    """
    from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt5.QtGui import QBrush, QColor
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    from dashpva.gui import configure_app
    from dashpva.viewer.core.base_window import BaseWindow

    class _AxisTable(QTableWidget):
        """Editable table of IOC-published motor axes (one axis per row)."""

        COLS = ['Name', 'Source PV or static value', 'Axis #', 'Direction', 'Role']

        def __init__(self, parent=None):
            super().__init__(0, len(self.COLS), parent)
            self.setHorizontalHeaderLabels(self.COLS)
            self.verticalHeader().setVisible(False)
            hdr = self.horizontalHeader()
            hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(1, QHeaderView.Stretch)
            hdr.setSectionResizeMode(2, QHeaderView.Fixed)
            hdr.setSectionResizeMode(3, QHeaderView.Fixed)
            hdr.setSectionResizeMode(4, QHeaderView.Fixed)
            self.setColumnWidth(2, 60)
            self.setColumnWidth(3, 80)
            self.setColumnWidth(4, 90)
            self.setSelectionBehavior(QTableWidget.SelectRows)
            self.setMinimumHeight(150)

        def add_axis(self, spec=None):
            spec = spec or ioc_rsm_store.AxisSpec(name=f'Axis{self.rowCount() + 1}')
            r = self.rowCount()
            self.insertRow(r)
            self.setItem(r, 0, QTableWidgetItem(spec.name))
            self.setItem(r, 1, QTableWidgetItem(spec.source_pv))
            spin = QSpinBox()
            spin.setRange(1, 8)
            spin.setValue(spec.axis_number)
            self.setCellWidget(r, 2, spin)
            self.setItem(r, 3, QTableWidgetItem(spec.direction))
            role = QComboBox()
            role.addItems(['sample', 'detector'])
            role.setCurrentText(spec.role if spec.role in ('sample', 'detector') else 'sample')
            self.setCellWidget(r, 4, role)

        def remove_selected(self):
            rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True)
            if not rows and self.rowCount():
                rows = [self.rowCount() - 1]
            for r in rows:
                self.removeRow(r)

        def axes(self):
            out = []
            for r in range(self.rowCount()):
                name_item = self.item(r, 0)
                pv_item = self.item(r, 1)
                dir_item = self.item(r, 3)
                out.append(ioc_rsm_store.AxisSpec(
                    name=(name_item.text() if name_item else '').strip(),
                    source_pv=(pv_item.text() if pv_item else '').strip(),
                    axis_number=self.cellWidget(r, 2).value(),
                    direction=(dir_item.text() if dir_item else '').strip(),
                    role=self.cellWidget(r, 4).currentText(),
                ))
            return out

    def _lbl(text, bold=False):
        w = QLabel(text)
        if bold:
            f = w.font()
            f.setBold(True)
            w.setFont(f)
        return w

    def _le(text):
        return QLineEdit(str(text))

    def _fmt(v):
        if v is None:
            return '—'
        if isinstance(v, list):
            return '[' + ', '.join(f'{x:.4g}' if isinstance(x, float) else str(x)
                                   for x in v) + ']'
        if isinstance(v, float):
            return f'{v:.6g}'
        return str(v)

    class PollWorker(QThread):
        """Reads PV values from the shared dict and emits (text, is_down) pairs.

        Rows backed by a source PV (status_key set) show 'None' when unconfigured
        or a red '--' when the source PV isn't connecting — never a fabricated
        value; everything else shows the live formatted value."""
        results_ready = pyqtSignal(list)

        def __init__(self, pvs):
            super().__init__()
            self._pvs    = pvs
            self._running = True

        def run(self):
            while self._running:
                with pv_lock:
                    snap = dict(pv_values)
                    status = dict(pv_status)
                results = []
                for pv, _desc, status_key in self._pvs:
                    st = status.get(status_key) if status_key else None
                    if st == 'none':
                        results.append(('None', False))
                    elif st == 'down':
                        results.append(('--', True))
                    else:
                        results.append((_fmt(snap.get(pv)), False))
                self.results_ready.emit(results)
                # ~20 Hz display refresh
                for _ in range(10):
                    if not self._running:
                        break
                    time.sleep(0.005)

        def stop(self):
            self._running = False

    class SimulatorWindow(BaseWindow):
        def __init__(self):
            super().__init__(viewer_name='IOC RSM Parameter', visible_actions=[])
            self.setWindowTitle('IOC for RSM conversion parameter')
            self._prefix = prefix
            self._config = config
            self._profile_label = profile_label
            self._build_ui()
            self._worker = PollWorker(_all_pv_names(self._prefix, self._config))
            self._worker.results_ready.connect(self._apply_results)
            self._worker.start()

        def _build_ui(self):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            self.setCentralWidget(scroll)
            root_w = QWidget()
            root = QVBoxLayout(root_w)
            root.setSpacing(8)
            root.setContentsMargins(10, 10, 10, 10)
            scroll.setWidget(root_w)
            root.addWidget(self._build_input_group())
            root.addWidget(self._build_pv_table_group())
            root.addStretch()
            self._progress = QProgressBar()
            self._progress.setFixedWidth(170)
            self._progress.setTextVisible(False)
            self._progress.setVisible(False)
            self.statusBar().addPermanentWidget(self._progress)
            self.statusBar().showMessage('Starting…')
            self.resize(780, 980)

        def _flash_applying(self):
            self._progress.setRange(0, 0)   # indeterminate "busy" sweep
            self._progress.setVisible(True)
            QTimer.singleShot(800, self._hide_progress)

        def _hide_progress(self):
            self._progress.setVisible(False)

        def _restart_and_refresh_pv_table(self, new_prefix, new_config, status_msg):
            """Relaunch the IOC subprocess, then rebuild the poll worker and the
               All IOC Records table for the (possibly changed) record set."""
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setVisible(True)
            QApplication.processEvents()
            self._prefix = new_prefix
            self._config = new_config
            restart_ioc(new_prefix, new_config)
            self._progress.setValue(40)
            QApplication.processEvents()
            self._worker.stop()
            self._worker.wait(2000)
            self._progress.setValue(70)
            QApplication.processEvents()
            self._all_pvs = _all_pv_names(self._prefix, self._config)
            self._pv_table.setRowCount(len(self._all_pvs))
            self._pv_val_items = []
            for row, (pv, _desc, _status_key) in enumerate(self._all_pvs):
                self._pv_table.setItem(row, 0, QTableWidgetItem(pv))
                item = QTableWidgetItem('—')
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self._pv_table.setItem(row, 1, item)
                self._pv_val_items.append(item)
            self._fit_pv_table_height()
            self._worker = PollWorker(_all_pv_names(self._prefix, self._config))
            self._worker.results_ready.connect(self._apply_results)
            self._worker.start()
            self._progress.setValue(100)
            QApplication.processEvents()
            self.statusBar().showMessage(status_msg)
            self._progress.setVisible(False)

        def _build_input_group(self):
            grp = QGroupBox('Source PV Inputs  —  enter a PV name or a static number')
            root = QVBoxLayout(grp)
            root.setSpacing(6)

            # Prefix + Reload row
            top_row = QWidget()
            top_lay = QHBoxLayout(top_row)
            top_lay.setContentsMargins(0, 0, 0, 0)
            top_lay.setSpacing(4)
            top_lay.addWidget(_lbl('Prefix', bold=True))
            self._prefix_edit = QLineEdit(self._prefix)
            self._prefix_edit.setPlaceholderText('e.g. 6idb:')
            top_lay.addWidget(self._prefix_edit, 1)
            self._profile_lbl = _lbl(f'Profile: {self._profile_label or "(none — using defaults)"}')
            top_lay.addWidget(self._profile_lbl)
            btn_apply_prefix = QPushButton('Apply prefix')
            def _on_apply_prefix():
                new_prefix = self._prefix_edit.text().strip()
                if new_prefix and not new_prefix.endswith(':'):
                    new_prefix += ':'
                    self._prefix_edit.setText(new_prefix)
                from dashpva import settings
                settings.save_ioc_prefix(new_prefix.rstrip(':'))
                grp.setTitle(f'Source PV Inputs  —  prefix: {new_prefix}')
                self._restart_and_refresh_pv_table(
                    new_prefix, self._config, f'IOC restarted with prefix: {new_prefix}')
            btn_apply_prefix.clicked.connect(_on_apply_prefix)
            top_lay.addWidget(btn_apply_prefix)
            btn_reload = QPushButton('Reload from active profile')
            def _on_reload():
                new_config, label = reload_config()
                self._profile_label = label
                self._profile_lbl.setText(f'Profile: {label or "(none — using defaults)"}')
                self._axis_table.blockSignals(True)
                self._axis_table.setRowCount(0)
                for ax in new_config.axes:
                    self._axis_table.add_axis(ax)
                self._axis_table.blockSignals(False)
                self._energy_edit.setText(new_config.energy_source_pv)
                self._restart_and_refresh_pv_table(
                    self._prefix, new_config, 'IOC reloaded from active profile')
            btn_reload.clicked.connect(_on_reload)
            top_lay.addWidget(btn_reload)
            root.addWidget(top_row)

            # Axis table — Name/Axis#/Direction edits are structural (need "Apply
            # axes" below); Source PV edits live-rebind immediately, no restart.
            root.addWidget(_lbl('Motor Axes', bold=True))
            self._axis_table = _AxisTable()
            for ax in self._config.axes:
                self._axis_table.add_axis(ax)
            root.addWidget(self._axis_table)

            axis_btn_row = QWidget()
            axis_btn_lay = QHBoxLayout(axis_btn_row)
            axis_btn_lay.setContentsMargins(0, 0, 0, 0)
            btn_add_axis = QPushButton('+ Add axis')
            btn_add_axis.clicked.connect(lambda: self._axis_table.add_axis())
            btn_rm_axis = QPushButton('− Remove selected')
            btn_rm_axis.clicked.connect(self._axis_table.remove_selected)
            btn_apply_axes = QPushButton('Apply axes')
            def _on_apply_axes():
                new_config = ioc_rsm_store.IOCRSMConfig(
                    axes=self._axis_table.axes(),
                    energy_source_pv=self._energy_edit.text().strip(),
                    ub_matrix=list(self._config.ub_matrix),
                    primary_beam=list(self._config.primary_beam),
                    inplane_ref=list(self._config.inplane_ref),
                    sample_normal=list(self._config.sample_normal),
                    detector=dict(self._config.detector),
                )
                ok = save_config(new_config)
                msg = ('IOC restarted with updated axes' if ok else
                       'IOC restarted with updated axes — NOT saved to profile (no active profile)')
                self._restart_and_refresh_pv_table(self._prefix, new_config, msg)
            btn_apply_axes.clicked.connect(_on_apply_axes)
            axis_btn_lay.addWidget(btn_add_axis)
            axis_btn_lay.addWidget(btn_rm_axis)
            axis_btn_lay.addStretch()
            axis_btn_lay.addWidget(btn_apply_axes)
            root.addWidget(axis_btn_row)

            # Energy row
            energy_row = QWidget()
            energy_lay = QHBoxLayout(energy_row)
            energy_lay.setContentsMargins(0, 0, 0, 0)
            energy_lay.addWidget(_lbl('X-ray Energy'))
            self._energy_edit = _le(self._config.energy_source_pv)
            self._energy_edit.setPlaceholderText(
                'PV name or static value  (e.g. 6idb:spec:Energy)')
            def _on_energy():
                val = self._energy_edit.text().strip()
                self._config.energy_source_pv = val
                send_cmd({'type': 'energy', 'value': val})
                if not save_config(self._config):
                    self.statusBar().showMessage('NOT saved to profile (no active profile)')
                self._flash_applying()
            self._energy_edit.editingFinished.connect(_on_energy)
            energy_lay.addWidget(self._energy_edit, 1)
            root.addWidget(energy_row)

            def _on_axis_item_changed(item):
                # Live rebind for the Source PV column only — the published record
                # set is unchanged, so no restart is needed. Name/axis#/direction
                # edits require the explicit "Apply axes" button (they change the
                # .db layout the IOC subprocess was built with).
                if item.column() != 1:
                    return
                row = item.row()
                name_item = self._axis_table.item(row, 0)
                if name_item is None or row >= len(self._config.axes):
                    return
                name = name_item.text().strip()
                val = item.text().strip()
                self._config.axes[row].source_pv = val
                send_cmd({'type': 'motor', 'name': name, 'value': val})
                if not save_config(self._config):
                    self.statusBar().showMessage('NOT saved to profile (no active profile)')
                self._flash_applying()
            self._axis_table.itemChanged.connect(_on_axis_item_changed)

            return grp

        def _build_pv_table_group(self):
            grp = QGroupBox('All IOC Records')
            lay = QVBoxLayout(grp)
            self._all_pvs = _all_pv_names(self._prefix, self._config)
            self._pv_table = QTableWidget(len(self._all_pvs), 2)
            self._pv_table.setHorizontalHeaderLabels(['PV Name', 'Value'])
            self._pv_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.Stretch)
            self._pv_table.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeToContents)
            self._pv_table.verticalHeader().setVisible(False)
            self._pv_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self._pv_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._pv_val_items = []
            for row, (pv, _desc, _status_key) in enumerate(self._all_pvs):
                self._pv_table.setItem(row, 0, QTableWidgetItem(pv))
                item = QTableWidgetItem('—')
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self._pv_table.setItem(row, 1, item)
                self._pv_val_items.append(item)
            self._fit_pv_table_height()
            lay.addWidget(self._pv_table)
            return grp

        def _fit_pv_table_height(self):
            """Size the records table to show every row; the window scroll area
               handles overflow instead of an inner scrollbar."""
            t = self._pv_table
            vh = t.verticalHeader()
            row_total = vh.length() or (vh.defaultSectionSize() * t.rowCount())
            header = t.horizontalHeader().height() or 24
            t.setMinimumHeight(row_total + header + 2 * t.frameWidth())

        def _apply_results(self, results):
            for item, (text, is_down) in zip(self._pv_val_items, results):
                item.setText(text)
                item.setForeground(QBrush(QColor(Qt.red)) if is_down else QBrush())
            self.statusBar().showMessage(
                f'Last update: {time.strftime("%H:%M:%S")}')

        def closeEvent(self, event):
            self._worker.stop()
            self._worker.wait(2000)
            super().closeEvent(event)

    app = QApplication(sys.argv)
    configure_app(app)
    win = SimulatorWindow()
    win.show()
    sys.exit(app.exec_())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='IOC for RSM conversion parameter')
    parser.add_argument('--prefix', default=None,
                         help='Defaults to the active profile\'s IOC_PREFIX')
    parser.add_argument('--config-file', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--ioc-mode', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.prefix is None:
        from dashpva import settings
        if not args.ioc_mode:
            try:
                settings.reload()
            except Exception:
                pass
        args.prefix = settings.IOC_PREFIX or DEFAULT_PREFIX

    if args.prefix and not args.prefix.endswith(':'):
        args.prefix += ':'

    if args.ioc_mode:
        # Fresh subprocess, no shared Python state — the config is handed off via
        # a one-shot temp JSON file written by _launch_ioc_proc() below.
        if args.config_file:
            with open(args.config_file) as f:
                config = ioc_rsm_store.IOCRSMConfig.from_full_dict(json.load(f))
            os.unlink(args.config_file)
        else:
            config = ioc_rsm_store.default_config()
        _run_ioc(args.prefix, config)
        return

    src, profile_label = ioc_rsm_store.active_source()
    config = ioc_rsm_store.load_config(src)
    _src_holder = [src]

    # Shared dicts: IOC subprocess writes JSON snapshots to stdout;
    # _fwd() parses them here so the GUI never needs caget.
    _pv_values = {}
    _pv_status = {}   # pv_name -> 'none' | 'down' | 'ok' (see _run_ioc._status)
    _pv_lock   = threading.Lock()
    _ioc_handle = [None]   # mutable holder so restart_ioc can swap the process
    _cmd_lock   = threading.Lock()

    _ready = threading.Event()

    def _launch_ioc_proc(prefix, cfg):
        tmp = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
        json.dump(cfg.to_full_dict(), tmp)
        tmp.close()
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__),
             '--ioc-mode', '--prefix', prefix, '--config-file', tmp.name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _ioc_handle[0] = proc

        def _fwd():
            for raw in proc.stdout:
                text = raw.decode(errors='replace').strip()
                try:
                    msg = json.loads(text)
                    if msg.get('type') == 'values':
                        with _pv_lock:
                            _pv_values.update(msg['data'])
                            _pv_status.update(msg.get('status') or {})
                        _ready.set()
                        continue
                except (json.JSONDecodeError, AttributeError):
                    pass
                print(text, flush=True)

        threading.Thread(target=_fwd, daemon=True).start()
        return proc

    def send_cmd(msg: dict):
        line = json.dumps(msg) + '\n'
        with _cmd_lock:
            proc = _ioc_handle[0]
            if proc is None:
                return
            try:
                proc.stdin.write(line.encode())
                proc.stdin.flush()
            except Exception as e:
                print(f'send_cmd error: {e}', flush=True)

    def restart_ioc(new_prefix, new_config):
        old = _ioc_handle[0]
        if old is not None:
            try:
                old.terminate()
                old.wait(timeout=5)
            except Exception:
                try:
                    old.kill()
                except Exception:
                    pass
        with _pv_lock:
            _pv_values.clear()
            _pv_status.clear()
        _launch_ioc_proc(new_prefix, new_config)
        print(f'IOC restarted with prefix={new_prefix}', flush=True)

    def save_config(cfg):
        if _src_holder[0] is None:
            # No profile was resolvable at GUI startup (e.g. tool launched before
            # any profile was selected) — re-resolve once before giving up.
            _src_holder[0], _ = ioc_rsm_store.active_source()
        return ioc_rsm_store.save_config(_src_holder[0], cfg)

    def reload_config():
        new_src, label = ioc_rsm_store.active_source()
        _src_holder[0] = new_src
        return ioc_rsm_store.load_config(new_src), label

    _launch_ioc_proc(args.prefix, config)

    # Wait up to 15 s for the first JSON snapshot from the IOC
    print(f'Waiting for IOC (prefix={args.prefix}) …', flush=True)
    if not _ready.wait(timeout=15):
        print('Warning: IOC did not respond within 15 s — opening GUI anyway.', flush=True)

    try:
        _run_gui(args.prefix, config, send_cmd, restart_ioc, _pv_values, _pv_status,
                 _pv_lock, save_config, reload_config, profile_label)
    finally:
        proc = _ioc_handle[0]
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == '__main__':
    main()
