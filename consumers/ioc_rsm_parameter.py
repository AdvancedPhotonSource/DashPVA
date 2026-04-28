#!/usr/bin/env python3
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
import threading
import time
import concurrent.futures

# ─────────────────────────────────────────────────────────────────────────────
# Persistent config
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_FILE = os.path.expanduser('~/.config/ioc_rsm_parameter.json')

def _load_config() -> dict:
    try:
        with open(_CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_config(cfg: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_CONFIG_FILE), exist_ok=True)
        with open(_CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f'Config save error: {e}', flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PREFIX = '6idb:'

MOTOR_AXES = [
    {'name': 'Mu',    'source_pv': '6idb1:m28.RBV', 'axis_number': 1, 'direction': 'x+'},
    {'name': 'Eta',   'source_pv': '6idb1:m17.RBV', 'axis_number': 2, 'direction': 'z-'},
    {'name': 'Chi',   'source_pv': '6idb1:m19.RBV', 'axis_number': 3, 'direction': 'y+'},
    {'name': 'Phi',   'source_pv': '6idb1:m20.RBV', 'axis_number': 4, 'direction': 'z-'},
    {'name': 'Nu',    'source_pv': '6idb1:m29.RBV', 'axis_number': 1, 'direction': 'x+'},
    {'name': 'Delta', 'source_pv': '6idb1:m18.RBV', 'axis_number': 2, 'direction': 'z-'},
]

DEFAULT_ENERGY_SOURCE_PV = '6idb:spec:Energy'
DEFAULT_UB            = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
DEFAULT_PRIMARY_BEAM  = [0, 1, 0]
DEFAULT_INPLANE_REF   = [0, 1, 0]
DEFAULT_SAMPLE_NORMAL = [0, 0, 1]
DEFAULT_DETECTOR = {
    'pixel_dir1': 'z-',
    'pixel_dir2': 'x-',
    'center':     [300, 300],
    'size':       [28.38, 28.38],
    'distance':   400.644,
    'units':      'mm',
}

POLL_INTERVAL    = 0.01   # IOC publish rate  (100 Hz)
CAGET_INTERVAL   = 0.02   # PV-source refresh (50 Hz)
SNAPSHOT_EVERY   = 5      # emit GUI snapshot every N IOC cycles (~20 Hz)


# ─────────────────────────────────────────────────────────────────────────────
# IOC subprocess  (pvaccess only — NO PyQt5)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ioc(prefix: str) -> None:
    # Duplicate stdin fd NOW, before pvaccess/EPICS init may close or redirect fd 0
    _cmd_fd = os.dup(0)

    import ctypes.util
    import tempfile
    import numpy as np
    import pvaccess as pva
    from epics import caget as _caget, PV as _PV

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
        for ax in MOTOR_AXES:
            lines.append(_ai(f"{p}{ax['name']}:Position"))
            lines.append(_ai(f"{p}{ax['name']}:AxisNumber"))
            lines.append(_so(f"{p}{ax['name']}:DirectionAxis", ax['direction']))
            lines.append(_so(f"{p}{ax['name']}:SpecMotorName", ax['name']))
        lines.append(_ai(f"{p}spec:Energy:Value"))
        lines.append(_wf(f"{p}spec:UB_matrix:Value", 9))
        for grp in ['PrimaryBeamDirection', 'InplaneReferenceDirection',
                    'SampleSurfaceNormalDirection']:
            for i in [1, 2, 3]:
                lines.append(_ai(f"{p}{grp}:AxisNumber{i}"))
        d = DEFAULT_DETECTOR
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

    def safe_float(pv):
        try:
            v = _caget(pv, timeout=0.3)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    def resolve(src):
        """Return (value, is_static). If src is a float string → static value.
           Otherwise treat as PV name and caget."""
        try:
            return float(src), True
        except ValueError:
            return safe_float(src), False

    # ── Local state (updated from stdin pipe) ─────────────────────────────
    _lock = threading.Lock()
    _state = {
        'motor': {ax['name']: ax['source_pv'] for ax in MOTOR_AXES},
        'energy': DEFAULT_ENERGY_SOURCE_PV,
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
            os.environ['EPICS_DB_INCLUDE_PATH'] = dbd
        elif os.environ.get('EPICS_BASE'):
            os.environ['EPICS_DB_INCLUDE_PATH'] = os.path.join(os.environ['EPICS_BASE'], 'dbd')

    dbd_dir = os.environ.get('EPICS_DB_INCLUDE_PATH', '')
    base_dbd = os.path.join(dbd_dir, 'base.dbd') if dbd_dir else 'base.dbd'

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
    for ax in MOTOR_AXES:
        ioc_put(caIoc, f"{p}{ax['name']}:AxisNumber",    float(ax['axis_number']))
        ioc_put(caIoc, f"{p}{ax['name']}:DirectionAxis", ax['direction'])
        ioc_put(caIoc, f"{p}{ax['name']}:SpecMotorName", ax['name'])
    ioc_put(caIoc, f"{p}spec:UB_matrix:Value", DEFAULT_UB[:])
    for i, v in enumerate(DEFAULT_PRIMARY_BEAM):
        ioc_put(caIoc, f"{p}PrimaryBeamDirection:AxisNumber{i+1}", float(v))
    for i, v in enumerate(DEFAULT_INPLANE_REF):
        ioc_put(caIoc, f"{p}InplaneReferenceDirection:AxisNumber{i+1}", float(v))
    for i, v in enumerate(DEFAULT_SAMPLE_NORMAL):
        ioc_put(caIoc, f"{p}SampleSurfaceNormalDirection:AxisNumber{i+1}", float(v))
    d = DEFAULT_DETECTOR
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
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())

    def _get(src):
        """Return float: static if src parses as float, else latest CA monitor value."""
        try:
            return float(src)
        except ValueError:
            with _pv_mons_lock:
                if src not in _pv_mons:
                    _pv_mons[src] = _PV(src, auto_monitor=True)
                v = _pv_mons[src].get(use_monitor=True)
            return float(v) if v is not None else 0.0

    # ── Background caget poll for static IOC records (2 Hz) ──────────────
    # Uses caget on our own IOC's PV names so external CA writes (caput from
    # scan software, alignment tools, etc.) are reflected in the snapshot.
    _static_pv_names = list(_current_vals.keys())   # all keys populated by startup ioc_puts

    def _poll_static():
        while not _stop.is_set():
            for pv_name in _static_pv_names:
                try:
                    v = _caget(pv_name, timeout=0.3)
                    if v is None:
                        continue
                    if hasattr(v, 'tolist'):          # numpy array → plain list
                        v = v.tolist()
                    elif hasattr(v, 'item'):          # numpy scalar → Python scalar
                        v = v.item()
                    _current_vals[pv_name] = v
                except Exception:
                    pass
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

            motor_vals = {ax['name']: _get(motor_srcs[ax['name']]) for ax in MOTOR_AXES}
            energy_val = _get(energy_src)

            for ax in MOTOR_AXES:
                ioc_put(caIoc, f"{p}{ax['name']}:Position", motor_vals[ax['name']])
            ioc_put(caIoc, f"{p}spec:Energy:Value", energy_val)

            # Snapshot: _current_vals has every value ever written via ioc_put;
            # overlay the fast-changing motor/energy values on top.
            _loop_n += 1
            if _loop_n % SNAPSHOT_EVERY == 0:
                snap = dict(_current_vals)
                print(json.dumps({'type': 'values', 'data': snap}), flush=True)

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

def _all_pv_names(prefix: str) -> list:
    """Return [(pv_name, description)] for every record the IOC publishes."""
    p = prefix
    pvs = []
    for ax in MOTOR_AXES:
        pvs.append((f"{p}{ax['name']}:Position",      f"{ax['name']} position"))
        pvs.append((f"{p}{ax['name']}:AxisNumber",    f"{ax['name']} axis number"))
        pvs.append((f"{p}{ax['name']}:DirectionAxis", f"{ax['name']} direction"))
        pvs.append((f"{p}{ax['name']}:SpecMotorName", f"{ax['name']} spec name"))
    pvs.append((f"{p}spec:Energy:Value",         'Energy value (keV)'))
    pvs.append((f"{p}spec:UB_matrix:Value",      'UB matrix (9 elements)'))
    for grp, desc in [
        ('PrimaryBeamDirection',         'Primary beam'),
        ('InplaneReferenceDirection',    'Inplane ref'),
        ('SampleSurfaceNormalDirection', 'Sample normal'),
    ]:
        for i in [1, 2, 3]:
            pvs.append((f"{p}{grp}:AxisNumber{i}", f'{desc} axis {i}'))
    pvs.append((f"{p}DetectorSetup:PixelDirection1",    'Detector pixel dir 1'))
    pvs.append((f"{p}DetectorSetup:PixelDirection2",    'Detector pixel dir 2'))
    pvs.append((f"{p}DetectorSetup:CenterChannelPixel", 'Detector center (px)'))
    pvs.append((f"{p}DetectorSetup:Size",               'Detector size (mm)'))
    pvs.append((f"{p}DetectorSetup:Distance",           'Detector distance (mm)'))
    pvs.append((f"{p}DetectorSetup:Units",              'Detector units'))
    pvs.append((f"{p}ScanOn:Value",   'Scan on flag'))
    pvs.append((f"{p}FilePath:Value", 'File path'))
    pvs.append((f"{p}FileName:Value", 'File name'))
    return pvs


def _run_gui(prefix: str, send_cmd, restart_ioc, pv_values: dict, pv_lock,
             init_motors: dict, init_energy: str, on_config_change) -> None:
    """
    pv_values / pv_lock : shared dict populated by _fwd() in main().
    init_motors          : {axis_name: source} restored from config.
    init_energy          : energy source restored from config.
    on_config_change     : called with (motor_sources, energy_source) on any edit.
    """
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QLineEdit, QGroupBox, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
        QStatusBar,
    )

    def _lbl(text, bold=False):
        w = QLabel(text)
        if bold:
            f = w.font(); f.setBold(True); w.setFont(f)
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
        """Reads PV values from the shared dict and emits formatted strings."""
        results_ready = pyqtSignal(list)

        def __init__(self, pvs):
            super().__init__()
            self._pvs    = pvs
            self._running = True

        def run(self):
            while self._running:
                with pv_lock:
                    snap = dict(pv_values)
                values = [_fmt(snap.get(pv)) for pv, _ in self._pvs]
                self.results_ready.emit(values)
                # ~20 Hz display refresh
                for _ in range(10):
                    if not self._running:
                        break
                    time.sleep(0.005)

        def stop(self):
            self._running = False

    class SimulatorWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('IOC for RSM conversion parameter')
            self._prefix = prefix
            self._build_ui()
            self._worker = PollWorker(_all_pv_names(self._prefix))
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
            self.setStatusBar(QStatusBar())
            self.statusBar().showMessage('Starting…')
            self.resize(780, 820)

        def _build_input_group(self):
            grp = QGroupBox('Source PV Inputs  —  enter a PV name or a static number')
            grid = QGridLayout(grp)
            grid.setSpacing(6)

            # Prefix row
            grid.addWidget(_lbl('Prefix', bold=True), 0, 0)
            prefix_row = QWidget()
            prefix_lay = QHBoxLayout(prefix_row)
            prefix_lay.setContentsMargins(0, 0, 0, 0)
            prefix_lay.setSpacing(4)
            self._prefix_edit = QLineEdit(self._prefix)
            self._prefix_edit.setPlaceholderText('e.g. 6idb:')
            btn_apply_prefix = QPushButton('Apply')
            btn_apply_prefix.setFixedWidth(60)
            def _on_apply_prefix():
                new_prefix = self._prefix_edit.text().strip()
                if not new_prefix.endswith(':'):
                    new_prefix += ':'
                    self._prefix_edit.setText(new_prefix)
                self._prefix = new_prefix
                grp.setTitle(f'Source PV Inputs  —  prefix: {new_prefix}')
                restart_ioc(new_prefix)
                # Restart the PV poll worker with the new prefix PV names
                self._worker.stop()
                self._worker.wait(2000)
                self._all_pvs = _all_pv_names(new_prefix)
                self._pv_table.setRowCount(len(self._all_pvs))
                self._pv_val_items = []
                for row, (pv, _) in enumerate(self._all_pvs):
                    self._pv_table.setItem(row, 0, QTableWidgetItem(pv))
                    from PyQt5.QtCore import Qt as _Qt
                    item = QTableWidgetItem('—')
                    item.setTextAlignment(_Qt.AlignLeft | _Qt.AlignVCenter)
                    self._pv_table.setItem(row, 1, item)
                    self._pv_val_items.append(item)
                self._worker = PollWorker(_all_pv_names(new_prefix))
                self._worker.results_ready.connect(self._apply_results)
                self._worker.start()
                self.statusBar().showMessage(f'IOC restarted with prefix: {new_prefix}')
            btn_apply_prefix.clicked.connect(_on_apply_prefix)
            prefix_lay.addWidget(self._prefix_edit)
            prefix_lay.addWidget(btn_apply_prefix)
            grid.addWidget(prefix_row, 0, 1)

            grid.addWidget(_lbl('Axis / Channel', bold=True), 1, 0)
            grid.addWidget(_lbl('Source PV or static value (editable)', bold=True), 1, 1)

            # Current motor sources — kept in sync for config saving
            self._cur_motors = {
                ax['name']: init_motors.get(ax['name'], ax['source_pv'])
                for ax in MOTOR_AXES
            }
            self._cur_energy = init_energy

            def _notify():
                on_config_change(dict(self._cur_motors), self._cur_energy)

            self._motor_pv_edits = []
            for row, ax in enumerate(MOTOR_AXES, start=2):
                grid.addWidget(_lbl(ax['name']), row, 0)
                edit = _le(self._cur_motors[ax['name']])
                def _on_motor(name=ax['name'], e=edit):
                    val = e.text().strip()
                    self._cur_motors[name] = val
                    send_cmd({'type': 'motor', 'name': name, 'value': val})
                    _notify()
                edit.editingFinished.connect(_on_motor)
                grid.addWidget(edit, row, 1)
                self._motor_pv_edits.append(edit)

            energy_row = len(MOTOR_AXES) + 2
            grid.addWidget(_lbl('X-ray Energy'), energy_row, 0)
            self._energy_edit = _le(self._cur_energy)
            def _on_energy():
                val = self._energy_edit.text().strip()
                self._cur_energy = val
                send_cmd({'type': 'energy', 'value': val})
                _notify()
            self._energy_edit.editingFinished.connect(_on_energy)
            grid.addWidget(self._energy_edit, energy_row, 1)
            grid.setColumnStretch(1, 1)
            return grp

        def _build_pv_table_group(self):
            grp = QGroupBox('All IOC Records')
            lay = QVBoxLayout(grp)
            self._all_pvs = _all_pv_names(self._prefix)
            self._pv_table = QTableWidget(len(self._all_pvs), 2)
            self._pv_table.setHorizontalHeaderLabels(['PV Name', 'Value'])
            self._pv_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.Stretch)
            self._pv_table.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeToContents)
            self._pv_table.verticalHeader().setVisible(False)
            self._pv_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self._pv_val_items = []
            for row, (pv, _) in enumerate(self._all_pvs):
                self._pv_table.setItem(row, 0, QTableWidgetItem(pv))
                item = QTableWidgetItem('—')
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self._pv_table.setItem(row, 1, item)
                self._pv_val_items.append(item)
            lay.addWidget(self._pv_table)
            return grp

        def _apply_results(self, values):
            for item, text in zip(self._pv_val_items, values):
                item.setText(text)
            self.statusBar().showMessage(
                f'Last update: {time.strftime("%H:%M:%S")}')

        def closeEvent(self, event):
            self._worker.stop()
            self._worker.wait(2000)
            super().closeEvent(event)

    app = QApplication(sys.argv)
    win = SimulatorWindow()
    win.show()
    sys.exit(app.exec_())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = _load_config()

    parser = argparse.ArgumentParser(description='IOC for RSM conversion parameter')
    parser.add_argument('--prefix', default=cfg.get('prefix', DEFAULT_PREFIX))
    parser.add_argument('--ioc-mode', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.prefix.endswith(':'):
        args.prefix += ':'

    if args.ioc_mode:
        _run_ioc(args.prefix)
        return

    # Persist the prefix used this session
    cfg['prefix'] = args.prefix
    _save_config(cfg)

    # Shared dict: IOC subprocess writes JSON snapshots to stdout;
    # _fwd() parses them here so the GUI never needs caget.
    _pv_values = {}
    _pv_lock   = threading.Lock()
    _ioc_handle = [None]   # mutable holder so restart_ioc can swap the process
    _cmd_lock   = threading.Lock()

    _ready = threading.Event()

    def _launch_ioc_proc(prefix):
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__),
             '--ioc-mode', '--prefix', prefix],
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

    def restart_ioc(new_prefix):
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
        _launch_ioc_proc(new_prefix)
        cfg['prefix'] = new_prefix
        _save_config(cfg)
        print(f'IOC restarted with prefix={new_prefix}', flush=True)

    _launch_ioc_proc(args.prefix)

    # Wait up to 15 s for the first JSON snapshot from the IOC
    print(f'Waiting for IOC (prefix={args.prefix}) …', flush=True)
    if not _ready.wait(timeout=15):
        print('Warning: IOC did not respond within 15 s — opening GUI anyway.', flush=True)

    # Restore saved motor / energy sources and push them to the IOC
    saved_motors = cfg.get('motor_sources', {})
    saved_energy = cfg.get('energy_source', DEFAULT_ENERGY_SOURCE_PV)
    for ax in MOTOR_AXES:
        src = saved_motors.get(ax['name'], ax['source_pv'])
        if src != ax['source_pv']:
            send_cmd({'type': 'motor', 'name': ax['name'], 'value': src})
    if saved_energy != DEFAULT_ENERGY_SOURCE_PV:
        send_cmd({'type': 'energy', 'value': saved_energy})

    def _on_config_change(motor_sources: dict, energy_source: str):
        cfg['motor_sources'] = motor_sources
        cfg['energy_source'] = energy_source
        _save_config(cfg)

    try:
        _run_gui(args.prefix, send_cmd, restart_ioc, _pv_values, _pv_lock,
                 saved_motors, saved_energy, _on_config_change)
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
