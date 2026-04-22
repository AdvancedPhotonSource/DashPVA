#!/usr/bin/env python3
"""
XRD Phase Fitter — Standalone PyQt5 Application

Generic phase fitting tool for powder XRD patterns. Supports two data modes:
  - File mode: Load TIF images + PONI + mask, integrate via pyFAI, then fit
  - Live mode: Subscribe to a PVA channel (e.g. pyFAI output) for real-time fitting

Uses ssrl_xrd_tools for all fitting logic and fast_phase_fit for optimized
direct-scipy fitting.

Usage:
    python phase_fitter.py
    python phase_fitter.py --pv-address pvapy:image:pyFAI --cif-dir /path/to/cifs
    python phase_fitter.py --config fit_config.json

Designed for integration into DashPVA as a real-time analysis module.
"""

import sys
import os
import re
import time
import json
import logging
import argparse
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict, deque

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

# --- PyQt5 ---
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QLabel, QPushButton, QComboBox, QCheckBox,
    QDoubleSpinBox, QSpinBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QProgressBar, QStatusBar, QLineEdit, QFormLayout, QHeaderView,
    QMessageBox, QAction, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal

# --- pyqtgraph ---
import pyqtgraph as pg

# --- natsort ---
from natsort import natsorted

# --- ssrl_xrd_tools ---
from ssrl_xrd_tools.analysis.phase import PhaseModel
from ssrl_xrd_tools.analysis.fitting import (
    PhaseFitter, FitConfig, FitResultStore, fit_sequence,
)
from ssrl_xrd_tools.analysis.fitting.phase_fitting import MultiPhaseResult
from ssrl_xrd_tools.io.image import read_image, load_mask
from ssrl_xrd_tools.integrate import integrate_1d, load_poni, poni_to_integrator

from utils.fast_phase_fit import fast_fit, fast_fit_sequence

# --- pvaccess (optional, for live mode) ---
try:
    import pvaccess as pva
    HAS_PVA = True
except ImportError:
    HAS_PVA = False

logger = logging.getLogger(__name__)

# Color palette for phases (cycles for arbitrary number of phases)
PHASE_PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
]

# Regex for samz extraction
_SAMZ_RE = re.compile(r'_(-?\d+)samz')
_INDEX_RE = re.compile(r'_(\d+)\.tif$')


def parse_group_key(fname):
    m = _SAMZ_RE.search(fname)
    if m:
        val = int(m.group(1))
        return (f'samz={val}', val)
    m = _INDEX_RE.search(fname)
    if m:
        return (m.group(1), int(m.group(1)))
    return (fname, 0)


def get_phase_color(name, idx=0):
    return PHASE_PALETTE[idx % len(PHASE_PALETTE)]


# =========================================================================
# FitWorker — runs fitting in a background QThread
# =========================================================================

class FitWorker(QThread):
    single_done = pyqtSignal(int, object, float)
    batch_progress = pyqtSignal(int, int)
    batch_done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = 'single'
        self.patterns = []
        self.phases = []
        self.config = None
        self.pattern_index = 0
        self.sequential = True
        self.labels = []
        self.fit_background_template = None
        self.prev_params = None
        self.use_fast_fit = True

    def configure_single(self, patterns, phases, config, index,
                         fit_background_template=None, prev_params=None,
                         use_fast_fit=True):
        self.mode = 'single'
        self.patterns = patterns
        self.phases = phases
        self.config = config
        self.pattern_index = index
        self.fit_background_template = fit_background_template
        self.prev_params = prev_params
        self.use_fast_fit = use_fast_fit

    def configure_batch(self, patterns, phases, config, sequential=True,
                        labels=None, fit_background_template=None,
                        use_fast_fit=True):
        self.mode = 'batch'
        self.patterns = patterns
        self.phases = phases
        self.config = config
        self.sequential = sequential
        self.labels = labels or [str(i) for i in range(len(patterns))]
        self.fit_background_template = fit_background_template
        self.use_fast_fit = use_fast_fit

    def run(self):
        try:
            if self.mode == 'single':
                self._run_single()
            else:
                self._run_batch()
        except Exception as e:
            self.error.emit(str(e))

    def _run_single(self):
        t0 = time.perf_counter()
        pat = self.patterns[self.pattern_index]
        q, y = pat[0], pat[1]
        sigma = pat[2] if len(pat) > 2 else None

        init_kw = dict(self.config.init_kw)
        if self.fit_background_template is not None:
            init_kw['fit_background_template'] = self.fit_background_template

        selected_phases = [
            p for p in self.phases
            if getattr(p, 'name', None) in self.config.phase_names
        ]

        fitter = PhaseFitter(q, y, sigma=sigma, **init_kw)
        for ph in selected_phases:
            fitter.add_phase(ph, min_intensity=self.config.min_intensity)

        fit_kw = dict(self.config.fit_kw)
        if self.prev_params is not None:
            fit_kw['params'] = self.prev_params

        if self.use_fast_fit:
            result = fast_fit(fitter, **fit_kw)
        else:
            result = fitter.fit(**fit_kw)
        elapsed = time.perf_counter() - t0
        self.single_done.emit(self.pattern_index, result, elapsed)

    def _run_batch(self):
        n = len(self.patterns)

        def _progress(i, total, result):
            self.batch_progress.emit(i, total)
            if result is not None:
                self.single_done.emit(i, result, 0.0)

        fit_seq_fn = fast_fit_sequence if self.use_fast_fit else fit_sequence
        store = fit_seq_fn(
            self.patterns, self.phases, self.config,
            sequential=self.sequential,
            labels=self.labels,
            fit_background_template=self.fit_background_template,
            progress_callback=_progress,
        )

        self.batch_progress.emit(n, n)
        self.batch_done.emit(store)


# =========================================================================
# LiveFitWorker — runs a single live fit in a background thread
# =========================================================================

class LiveFitWorker(QThread):
    done = pyqtSignal(int, object, float, str)  # frame, result, elapsed, error

    def __init__(self, q, intensity, sigma, phases, init_kw, fit_kw,
                 min_intensity, use_fast, frame_number, parent=None):
        super().__init__(parent)
        self.q = q
        self.intensity = intensity
        self.sigma = sigma
        self.phases = phases
        self.init_kw = init_kw
        self.fit_kw = fit_kw
        self.min_intensity = min_intensity
        self.use_fast = use_fast
        self.frame_number = frame_number

    def run(self):
        try:
            t0 = time.perf_counter()
            fitter = PhaseFitter(self.q, self.intensity,
                                 sigma=self.sigma, **self.init_kw)
            for ph in self.phases:
                fitter.add_phase(ph, min_intensity=self.min_intensity)
            if self.use_fast:
                result = fast_fit(fitter, **self.fit_kw)
            else:
                result = fitter.fit(**self.fit_kw)
            elapsed = time.perf_counter() - t0
            self.done.emit(self.frame_number, result, elapsed, '')
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.done.emit(self.frame_number, None, elapsed, str(e))


# =========================================================================
# PhaseLoadWorker — loads CIF files in a background thread
# =========================================================================

class PhaseLoadWorker(QThread):
    progress = pyqtSignal(int, int, str)  # current, total, name
    done = pyqtSignal(list, float)        # phases, wavelength_A

    def __init__(self, cif_dir, wavelength_A, parent=None):
        super().__init__(parent)
        self.cif_dir = Path(cif_dir)
        self.wavelength_A = wavelength_A

    def run(self):
        cif_files = sorted(self.cif_dir.glob('*.cif'))
        phases = []
        for i, cif_path in enumerate(cif_files):
            self.progress.emit(i, len(cif_files), cif_path.stem)
            try:
                ph = PhaseModel.from_cif(cif_path)
                ph.calculate_peaks(wavelength=self.wavelength_A)
                phases.append(ph)
                print(f"  [{i+1}/{len(cif_files)}] Loaded {ph.name}: "
                      f"{len(ph.peaks)} peaks")
            except Exception as e:
                print(f"  [{i+1}/{len(cif_files)}] FAILED {cif_path.name}: {e}")
        self.done.emit(phases, self.wavelength_A)


# =========================================================================
# IntegrationWorker — runs TIF integration in background
# =========================================================================

class IntegrationWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(list, list, object)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_dir = None
        self.poni_file = None
        self.mask_file = None
        self.npt = 2000
        self.q_range = (1.0, 5.8)
        self.threshold = 10000
        self.tif_pattern = '*.tif'
        self.substrate_dir = None

    def run(self):
        try:
            data_dir = Path(self.data_dir)
            poni = load_poni(self.poni_file)
            ai = poni_to_integrator(poni)

            tif_files = natsorted(data_dir.glob(self.tif_pattern))
            if not tif_files:
                self.error.emit(f"No TIF files matching '{self.tif_pattern}' in {data_dir}")
                return

            mask = load_mask(self.mask_file, threshold=self.threshold,
                             data=read_image(tif_files[0]))

            samz_groups = OrderedDict()
            for f in tif_files:
                label, _ = parse_group_key(f.name)
                samz_groups.setdefault(label, []).append(f)

            samz_positions = sorted(
                samz_groups.keys(),
                key=lambda lbl: parse_group_key(
                    next(iter(samz_groups[lbl])).name)[1],
            )

            patterns = []
            labels = []
            total = len(samz_positions)

            for idx, pos in enumerate(samz_positions):
                self.progress.emit(idx, total)
                frames = samz_groups[pos]
                frame_results = []
                for f in frames:
                    img = read_image(f)
                    r = integrate_1d(
                        img, ai, npt=self.npt, unit='q_A^-1',
                        mask=mask, radial_range=self.q_range,
                        error_model='poisson',
                    )
                    frame_results.append(r)

                q = frame_results[0].radial
                I_stack = np.array([r.intensity for r in frame_results])
                I_avg = I_stack.mean(axis=0)

                if frame_results[0].sigma is not None:
                    sig_stack = np.array([r.sigma for r in frame_results])
                    sig_avg = np.sqrt((sig_stack ** 2).sum(axis=0)) / len(frame_results)
                else:
                    sig_avg = None

                patterns.append((q, I_avg, sig_avg))
                labels.append(str(pos))

            template = None
            if self.substrate_dir and Path(self.substrate_dir).exists():
                sub_dir = Path(self.substrate_dir)
                sub_tifs = natsorted(sub_dir.glob('*.tif'))
                if sub_tifs:
                    sub_results = []
                    for f in sub_tifs:
                        img = read_image(f)
                        r = integrate_1d(
                            img, ai, npt=self.npt, unit='q_A^-1',
                            mask=mask, radial_range=self.q_range,
                            error_model='poisson',
                        )
                        sub_results.append(r)
                    q_t = sub_results[0].radial
                    I_t = np.mean([r.intensity for r in sub_results], axis=0)
                    template = (q_t, I_t)

            self.progress.emit(total, total)
            self.done.emit(patterns, labels, template)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# =========================================================================
# PhaseFitApp — Main Application Window
# =========================================================================

class PhaseFitApp(QMainWindow):
    """Generic standalone PyQt5 phase fitting application."""

    live_data_received = pyqtSignal(object, object, object, int, float)  # unused — kept for compat

    def __init__(self, pv_address=None, poni_file=None, cif_dir=None,
                 wavelength_A=None, config_path=None):
        super().__init__()
        self.setWindowTitle("XRD Phase Fitter")
        self.resize(1400, 900)

        # State
        self.patterns = []
        self.labels = []
        self.phases = []
        self.results_cache = {}
        self.fit_background_template = None
        self.wavelength_A = wavelength_A
        self.fit_worker = None
        self.integration_worker = None
        self.store = None
        self.phase_checkboxes = {}

        # Live mode state
        self._live_channel = None
        self._live_connected = False
        self._live_frame_count = 0
        self._live_prev_params = None
        self._max_history = 300
        self._live_history = deque(maxlen=self._max_history)
        self._live_fitting = False
        self._live_pending = None
        self._live_fit_worker = None
        self._phases_loading = False
        self._phase_load_worker = None
        self._output_pv_server = None
        self._output_pv_address = None
        self._output_pv_type = None

        # CLI defaults
        self._init_pv_address = pv_address
        self._init_poni_file = poni_file or ''
        self._init_cif_dir = cif_dir or ''
        self._init_config = config_path

        self._build_ui()
        self._build_menu()
        self._setup_plots()

        self._latest_live_frame = None
        self._live_rx_count = 0
        self._live_displayed_count = 0
        self._live_poll_timer = QTimer(self)
        self._live_poll_timer.setInterval(100)  # 10 Hz max GUI update
        self._live_poll_timer.timeout.connect(self._poll_live_data)

        # Apply CLI config if provided
        if self._init_config and Path(self._init_config).exists():
            config = FitConfig.load(self._init_config)
            self._apply_config(config)

        self._pending_integration = None

        # Watch mode state
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(60_000)
        self._watch_timer.timeout.connect(self._watch_tick)
        self._watch_seen_files = set()
        self._watch_fitting = False
        self._watch_integration_info = None
        self._watch_index_offset = 0
        self._watch_worker = None

        # Auto-load phases if CIF dir + wavelength provided (live mode launch)
        if self._init_cif_dir and self.wavelength_A and self.wavelength_A > 0:
            if Path(self._init_cif_dir).exists():
                print(f"[Phase Fitter] Auto-loading CIFs with "
                      f"wavelength={self.wavelength_A:.4f} Å")
                self._load_phases_async(self._init_cif_dir, self.wavelength_A)
        elif self._init_cif_dir and self._init_poni_file:
            if Path(self._init_cif_dir).exists() and Path(self._init_poni_file).exists():
                try:
                    poni = load_poni(self._init_poni_file)
                    ai = poni_to_integrator(poni)
                    wl = ai.wavelength * 1e10
                    self._load_phases_async(self._init_cif_dir, wl)
                except Exception as e:
                    print(f"[Phase Fitter] Failed to load PONI: {e}")

        # Auto-connect in live mode if PV address provided
        if self._init_pv_address:
            self.txt_pv_address.setText(self._init_pv_address)

        self.statusBar().showMessage("Ready. Load data or connect to live stream.")

    # -----------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        load_action = QAction("Load Data Directory...", self)
        load_action.triggered.connect(self._browse_data_dir)
        file_menu.addAction(load_action)

        load_config = QAction("Load Config...", self)
        load_config.triggered.connect(self.load_config)
        file_menu.addAction(load_config)

        save_config = QAction("Save Config...", self)
        save_config.triggered.connect(self.save_config)
        file_menu.addAction(save_config)

        file_menu.addSeparator()

        export_action = QAction("Export CSV...", self)
        export_action.triggered.connect(self.export_csv)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ---- LEFT: Controls Panel (scrollable) ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(330)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # -- File Mode --
        file_group = QGroupBox("File Mode")
        file_form = QFormLayout()

        self.txt_data_dir = QLineEdit(self._init_cif_dir)
        self.btn_browse_data = QPushButton("...")
        self.btn_browse_data.setFixedWidth(30)
        self.btn_browse_data.clicked.connect(self._browse_data_dir)
        row = QHBoxLayout()
        row.addWidget(self.txt_data_dir)
        row.addWidget(self.btn_browse_data)
        file_form.addRow("Data Dir:", row)

        self.txt_poni = QLineEdit(self._init_poni_file)
        self.btn_browse_poni = QPushButton("...")
        self.btn_browse_poni.setFixedWidth(30)
        self.btn_browse_poni.clicked.connect(self._browse_poni)
        row2 = QHBoxLayout()
        row2.addWidget(self.txt_poni)
        row2.addWidget(self.btn_browse_poni)
        file_form.addRow("PONI:", row2)

        self.txt_mask = QLineEdit('')
        self.btn_browse_mask = QPushButton("...")
        self.btn_browse_mask.setFixedWidth(30)
        self.btn_browse_mask.clicked.connect(self._browse_mask)
        row3 = QHBoxLayout()
        row3.addWidget(self.txt_mask)
        row3.addWidget(self.btn_browse_mask)
        file_form.addRow("Mask:", row3)

        self.txt_cif_dir = QLineEdit(self._init_cif_dir)
        self.btn_browse_cif = QPushButton("...")
        self.btn_browse_cif.setFixedWidth(30)
        self.btn_browse_cif.clicked.connect(self._browse_cif_dir)
        row4 = QHBoxLayout()
        row4.addWidget(self.txt_cif_dir)
        row4.addWidget(self.btn_browse_cif)
        file_form.addRow("CIF Dir:", row4)

        self.txt_substrate = QLineEdit('')
        self.btn_browse_sub = QPushButton("...")
        self.btn_browse_sub.setFixedWidth(30)
        self.btn_browse_sub.clicked.connect(self._browse_substrate)
        row5 = QHBoxLayout()
        row5.addWidget(self.txt_substrate)
        row5.addWidget(self.btn_browse_sub)
        file_form.addRow("Substrate:", row5)

        self.txt_tif_pattern = QLineEdit('*.tif')
        file_form.addRow("TIF Pattern:", self.txt_tif_pattern)

        self.btn_load = QPushButton("Load && Integrate")
        self.btn_load.clicked.connect(self.load_and_integrate)
        file_form.addRow(self.btn_load)

        self.btn_load_cifs = QPushButton("Load CIFs Only")
        self.btn_load_cifs.setToolTip(
            "Load CIF phases using PONI wavelength (or manual entry). "
            "Use this to pre-load phases before connecting to live stream.")
        self.btn_load_cifs.clicked.connect(self._load_cifs_only)
        file_form.addRow(self.btn_load_cifs)

        self.chk_watch = QCheckBox("Watch for new data")
        self.chk_watch.setToolTip(
            "Check every 60s for new TIF files and integrate them.\n"
            "When active, Fit All will continuously fit new data.")
        self.chk_watch.stateChanged.connect(self._toggle_watch_mode)
        file_form.addRow(self.chk_watch)

        file_group.setLayout(file_form)

        self._file_mode_widgets = [
            self.txt_data_dir, self.btn_browse_data,
            self.txt_poni, self.btn_browse_poni,
            self.txt_mask, self.btn_browse_mask,
            self.txt_substrate, self.btn_browse_sub,
            self.txt_tif_pattern, self.btn_load,
        ]
        left_layout.addWidget(file_group)

        # -- Live Mode --
        live_group = QGroupBox("Live Mode")
        live_form = QFormLayout()

        self.txt_pv_address = QLineEdit(self._init_pv_address or 'pvapy:image:pyFAI')
        live_form.addRow("PV Address:", self.txt_pv_address)

        self.btn_live_connect = QPushButton("Connect")
        self.btn_live_connect.clicked.connect(self._toggle_live_connection)
        if not HAS_PVA:
            self.btn_live_connect.setEnabled(False)
            self.btn_live_connect.setToolTip("pvaccess not installed")
        live_form.addRow(self.btn_live_connect)

        self.chk_auto_fit = QCheckBox("Auto-fit incoming frames")
        self.chk_auto_fit.setChecked(True)
        live_form.addRow(self.chk_auto_fit)

        self.lbl_live_status = QLabel("Disconnected")
        live_form.addRow("Status:", self.lbl_live_status)

        self.spn_max_history = QSpinBox()
        self.spn_max_history.setRange(10, 10000)
        self.spn_max_history.setValue(self._max_history)
        self.spn_max_history.valueChanged.connect(self._on_max_history_changed)
        live_form.addRow("Max history:", self.spn_max_history)

        live_group.setLayout(live_form)
        left_layout.addWidget(live_group)

        # -- Pattern Selection --
        pat_group = QGroupBox("Pattern Selection")
        pat_form = QFormLayout()

        self.cmb_pattern = QComboBox()
        self.cmb_pattern.currentIndexChanged.connect(self._on_pattern_changed)
        pat_form.addRow("Pattern:", self.cmb_pattern)

        self.chk_sequential = QCheckBox("Sequential fitting")
        self.chk_sequential.setChecked(True)
        pat_form.addRow(self.chk_sequential)

        self.chk_fast_fit = QCheckBox("Fast fit (direct scipy)")
        self.chk_fast_fit.setChecked(True)
        pat_form.addRow(self.chk_fast_fit)

        pat_group.setLayout(pat_form)
        left_layout.addWidget(pat_group)

        # -- Phases --
        phase_group = QGroupBox("Phases")
        self._phase_group_layout = QVBoxLayout()
        self._phase_placeholder = QLabel("Load data to discover phases")
        self._phase_placeholder.setStyleSheet("color: gray; font-style: italic;")
        self._phase_group_layout.addWidget(self._phase_placeholder)

        phase_btn_row = QHBoxLayout()
        btn_sel_all = QPushButton("All")
        btn_sel_none = QPushButton("None")
        btn_sel_all.clicked.connect(lambda: self._set_all_phase_checks(True))
        btn_sel_none.clicked.connect(lambda: self._set_all_phase_checks(False))
        phase_btn_row.addWidget(btn_sel_all)
        phase_btn_row.addWidget(btn_sel_none)
        self._phase_group_layout.addLayout(phase_btn_row)

        phase_group.setLayout(self._phase_group_layout)
        left_layout.addWidget(phase_group)

        # -- Fit Settings --
        fit_group = QGroupBox("Fit Settings")
        fit_form = QFormLayout()

        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems([
            'pseudovoigt', 'gaussian', 'lorentzian', 'voigt',
            'pearson7', 'lorentzian_squared', 'splitlorentzian',
        ])
        fit_form.addRow("Profile:", self.cmb_profile)

        self.cmb_prefit_bg = QComboBox()
        self.cmb_prefit_bg.addItems(['snip', 'chebyshev', 'none'])
        fit_form.addRow("Pre-fit BG:", self.cmb_prefit_bg)

        self.cmb_fit_bg = QComboBox()
        self.cmb_fit_bg.addItems([
            'chebyshev3', 'template', 'polynomial3', 'none',
            'chebyshev5', 'polynomial5', 'template+poly2',
        ])
        fit_form.addRow("Fit BG:", self.cmb_fit_bg)

        self.cmb_texture = QComboBox()
        self.cmb_texture.addItems(['free', 'none', 'march_dollase'])
        fit_form.addRow("Texture:", self.cmb_texture)

        self.cmb_amorphous = QComboBox()
        self.cmb_amorphous.addItems(['gaussian', 'pseudovoigt', 'none'])
        fit_form.addRow("Amorphous:", self.cmb_amorphous)

        self.spn_lattice_pct = QDoubleSpinBox()
        self.spn_lattice_pct.setRange(0.001, 0.20)
        self.spn_lattice_pct.setSingleStep(0.005)
        self.spn_lattice_pct.setValue(0.05)
        self.spn_lattice_pct.setDecimals(3)
        fit_form.addRow("Lattice %:", self.spn_lattice_pct)

        self.spn_qshift = QDoubleSpinBox()
        self.spn_qshift.setRange(0.001, 0.20)
        self.spn_qshift.setSingleStep(0.005)
        self.spn_qshift.setValue(0.05)
        self.spn_qshift.setDecimals(3)
        fit_form.addRow("Q-shift:", self.spn_qshift)

        self.spn_snip_width = QSpinBox()
        self.spn_snip_width.setRange(5, 200)
        self.spn_snip_width.setValue(30)
        fit_form.addRow("SNIP width:", self.spn_snip_width)

        self.chk_caglioti = QCheckBox("Caglioti width model")
        self.chk_caglioti.setChecked(True)
        fit_form.addRow(self.chk_caglioti)

        self.chk_lock_cross = QCheckBox("Lock cross-phase order")
        self.chk_lock_cross.setChecked(True)
        fit_form.addRow(self.chk_lock_cross)

        self.spn_width_min = QDoubleSpinBox()
        self.spn_width_min.setRange(0.001, 1.0)
        self.spn_width_min.setValue(0.02)
        self.spn_width_min.setDecimals(3)
        fit_form.addRow("Width min:", self.spn_width_min)

        self.spn_width_max = QDoubleSpinBox()
        self.spn_width_max.setRange(0.01, 1.0)
        self.spn_width_max.setValue(0.15)
        self.spn_width_max.setDecimals(3)
        fit_form.addRow("Width max:", self.spn_width_max)

        self.spn_min_intensity = QDoubleSpinBox()
        self.spn_min_intensity.setRange(0.0, 100.0)
        self.spn_min_intensity.setValue(5.0)
        self.spn_min_intensity.setDecimals(1)
        fit_form.addRow("Min intensity:", self.spn_min_intensity)

        self.spn_max_nfev = QSpinBox()
        self.spn_max_nfev.setRange(100, 50000)
        self.spn_max_nfev.setValue(3000)
        self.spn_max_nfev.setSingleStep(500)
        fit_form.addRow("Max evals:", self.spn_max_nfev)

        fit_group.setLayout(fit_form)
        left_layout.addWidget(fit_group)

        # -- Actions --
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()

        self.btn_fit_current = QPushButton("Fit Current")
        self.btn_fit_current.clicked.connect(self.fit_current)
        action_layout.addWidget(self.btn_fit_current)

        self.btn_fit_all = QPushButton("Fit All")
        self.btn_fit_all.clicked.connect(self.fit_all)
        action_layout.addWidget(self.btn_fit_all)

        self.btn_save_config = QPushButton("Save Config")
        self.btn_save_config.clicked.connect(self.save_config)
        action_layout.addWidget(self.btn_save_config)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.clicked.connect(self.export_csv)
        action_layout.addWidget(self.btn_export)

        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)

        # -- Progress --
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        left_layout.addWidget(self.progress_bar)

        left_layout.addStretch()

        scroll.setWidget(left_widget)

        # ---- RIGHT: Plot Area ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(2, 2, 2, 2)

        plot_splitter = QSplitter(Qt.Vertical)

        self.main_plot = pg.PlotWidget(title="XRD Pattern")
        self.main_plot.setLabel('bottom', 'q', units='\u00c5\u207b\u00b9')
        self.main_plot.setLabel('left', 'Intensity', units='a.u.')
        self.main_plot.addLegend(offset=(60, 10))
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)
        plot_splitter.addWidget(self.main_plot)

        self.resid_plot = pg.PlotWidget(title="Residual")
        self.resid_plot.setLabel('bottom', 'q', units='\u00c5\u207b\u00b9')
        self.resid_plot.setLabel('left', 'Residual')
        self.resid_plot.showGrid(x=True, y=True, alpha=0.3)
        plot_splitter.addWidget(self.resid_plot)

        self.trend_plot = pg.PlotWidget(title="Phase Fractions")
        self.trend_plot.setLabel('bottom', 'Index')
        self.trend_plot.setLabel('left', 'Fraction')
        self.trend_plot.addLegend(offset=(60, 10))
        self.trend_plot.showGrid(x=True, y=True, alpha=0.3)
        self.trend_plot.setYRange(-0.05, 1.05)
        plot_splitter.addWidget(self.trend_plot)

        plot_splitter.setSizes([500, 150, 200])

        right_layout.addWidget(plot_splitter, stretch=3)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            'Phase', 'Fraction', 'a (\u00c5)', 'b (\u00c5)', 'c (\u00c5)',
            'Scale', '\u03c7\u00b2_red',
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.results_table.setMaximumHeight(180)
        right_layout.addWidget(self.results_table, stretch=0)

        splitter.addWidget(scroll)
        splitter.addWidget(right_widget)
        splitter.setSizes([330, 1070])

    def _setup_plots(self):
        self.curve_data = self.main_plot.plot(
            [], [], pen=None, symbol='o', symbolSize=2,
            symbolBrush='#333333', name='Data')
        self.curve_fit = self.main_plot.plot(
            [], [], pen=pg.mkPen('#ff0000', width=2), name='Fit')
        self.curve_bg = self.main_plot.plot(
            [], [], pen=pg.mkPen('#888888', width=1, style=Qt.DashLine),
            name='Background')
        self.curve_amorphous = self.main_plot.plot(
            [], [], pen=pg.mkPen('#ff8c00', width=1, style=Qt.DashLine),
            name='Amorphous')
        self.phase_curves = []
        self.peak_markers = []

        self.curve_resid = self.resid_plot.plot(
            [], [], pen=pg.mkPen('#333333', width=1))
        self.resid_zero = self.resid_plot.addLine(y=0, pen=pg.mkPen('#999999',
                                                                     width=1,
                                                                     style=Qt.DashLine))
        self.trend_curves = {}

    # -----------------------------------------------------------------
    # File Browsing
    # -----------------------------------------------------------------

    def _browse_data_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Data Directory",
                                             self.txt_data_dir.text())
        if d:
            self.txt_data_dir.setText(d)
            self._auto_detect_paths(Path(d))

    def _auto_detect_paths(self, data_dir):
        found = []
        missing = []

        cif_found = False
        for sub in ['CIF Files', 'CIF', 'cif', 'cifs']:
            cif = data_dir / sub
            if cif.exists():
                self.txt_cif_dir.setText(str(cif))
                found.append(f"CIF Dir ({sub}/)")
                cif_found = True
                break
        if not cif_found:
            missing.append("CIF Dir")

        poni_found = False
        for sub in ['LaB6', 'calibration', 'calib']:
            cal = data_dir / sub
            if cal.exists():
                ponis = sorted(cal.glob('*.poni'))
                if ponis:
                    self.txt_poni.setText(str(ponis[0]))
                    found.append(f"PONI ({ponis[0].name})")
                    poni_found = True
                    break
        if not poni_found:
            missing.append("PONI")

        mask_found = False
        for name in ['mask.edf', 'mask.npy', 'mask.tif']:
            m = data_dir / name
            if m.exists():
                self.txt_mask.setText(str(m))
                found.append(f"Mask ({name})")
                mask_found = True
                break
        if not mask_found:
            missing.append("Mask")

        sub_found = False
        for sub in ['Fused_Silica', 'Substrate', 'substrate', 'background']:
            s = data_dir / sub
            if s.exists():
                self.txt_substrate.setText(str(s))
                found.append(f"Substrate ({sub}/)")
                sub_found = True
                break
        if not sub_found:
            missing.append("Substrate")

        parts = []
        if found:
            parts.append("Found: " + ", ".join(found))
        if missing:
            parts.append("Not found: " + ", ".join(missing))
        self.statusBar().showMessage(" | ".join(parts))

    def _browse_poni(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select PONI File", self.txt_poni.text(),
            "PONI Files (*.poni);;All Files (*)")
        if f:
            self.txt_poni.setText(f)

    def _browse_mask(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Mask File", self.txt_mask.text(),
            "Mask Files (*.edf *.npy *.tif);;All Files (*)")
        if f:
            self.txt_mask.setText(f)

    def _browse_cif_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select CIF Directory",
                                             self.txt_cif_dir.text())
        if d:
            self.txt_cif_dir.setText(d)

    def _browse_substrate(self):
        d = QFileDialog.getExistingDirectory(self, "Select Substrate Directory",
                                             self.txt_substrate.text())
        if d:
            self.txt_substrate.setText(d)

    # -----------------------------------------------------------------
    # Phase Selection
    # -----------------------------------------------------------------

    @property
    def selected_phases(self):
        return [ph for ph in self.phases
                if self.phase_checkboxes.get(ph.name, QCheckBox()).isChecked()]

    def _populate_phase_checkboxes(self):
        for cb in self.phase_checkboxes.values():
            self._phase_group_layout.removeWidget(cb)
            cb.deleteLater()
        self.phase_checkboxes.clear()

        if self._phase_placeholder is not None:
            self._phase_placeholder.hide()

        for ph in self.phases:
            cb = QCheckBox(ph.name)
            cb.setChecked(True)
            self.phase_checkboxes[ph.name] = cb
            idx = self._phase_group_layout.count() - 1
            self._phase_group_layout.insertWidget(idx, cb)

    def _set_all_phase_checks(self, checked):
        for cb in self.phase_checkboxes.values():
            cb.setChecked(checked)

    # -----------------------------------------------------------------
    # Phase Loading (generic — discovers all CIF files)
    # -----------------------------------------------------------------

    def _load_phases_async(self, cif_dir, wavelength_A):
        cif_dir = Path(cif_dir)
        if not cif_dir.exists():
            self.statusBar().showMessage(f"CIF directory not found: {cif_dir}")
            return

        cif_files = sorted(cif_dir.glob('*.cif'))
        if not cif_files:
            self.statusBar().showMessage(f"No CIF files found in {cif_dir}")
            return

        self._phases_loading = True
        self.btn_load_cifs.setEnabled(False)
        self.btn_load_cifs.setText("Loading CIFs...")
        print(f"[Phase Fitter] Loading {len(cif_files)} CIF files from {cif_dir} "
              f"(wavelength={wavelength_A:.4f} Å)...")
        self.statusBar().showMessage(
            f"Loading {len(cif_files)} CIF files... (fitting paused)")

        self._phase_load_worker = PhaseLoadWorker(cif_dir, wavelength_A)
        self._phase_load_worker.progress.connect(self._on_phase_load_progress)
        self._phase_load_worker.done.connect(self._on_phase_load_done)
        self._phase_load_worker.start()

    def _on_phase_load_progress(self, current, total, name):
        self.statusBar().showMessage(
            f"Loading CIF [{current+1}/{total}]: {name}...")
        print(f"[Phase Fitter] Loading CIF [{current+1}/{total}]: {name}")

    def _on_phase_load_done(self, phases, wavelength_A):
        self._phases_loading = False
        self.btn_load_cifs.setEnabled(True)
        self.btn_load_cifs.setText("Load CIFs Only")
        self.phases = phases
        self.wavelength_A = wavelength_A
        self._populate_phase_checkboxes()
        names = ', '.join(ph.name for ph in phases)
        print(f"[Phase Fitter] Loaded {len(phases)} phases at "
              f"{wavelength_A:.4f} Å: {names}")
        self.statusBar().showMessage(
            f"Loaded {len(self.phases)} phases at {wavelength_A:.4f} Å. "
            f"Ready to fit.")

    def _load_cifs_only(self):
        cif_dir = self.txt_cif_dir.text().strip()
        if not cif_dir or not Path(cif_dir).exists():
            QMessageBox.warning(self, "Error",
                "Please set a valid CIF directory first.")
            return

        if self.wavelength_A and self.wavelength_A > 0:
            default_wl = self.wavelength_A
        else:
            default_wl = 0.7293

        from PyQt5.QtWidgets import QInputDialog
        wl, ok = QInputDialog.getDouble(
            self, "Wavelength",
            "Enter wavelength in Å\n"
            "(In live mode this will update automatically from the stream):",
            value=default_wl, min=0.1, max=5.0, decimals=4)
        if not ok:
            return
        print(f"[Phase Fitter] Wavelength: {wl:.4f} Å")

        self._load_phases_async(cif_dir, wl)

    # -----------------------------------------------------------------
    # Data Loading & Integration (File Mode)
    # -----------------------------------------------------------------

    def load_and_integrate(self):
        data_dir = self.txt_data_dir.text()
        poni_file = self.txt_poni.text()
        mask_file = self.txt_mask.text()
        cif_dir = self.txt_cif_dir.text()

        if not Path(data_dir).exists():
            QMessageBox.warning(self, "Error", f"Data directory not found: {data_dir}")
            return
        if not Path(poni_file).exists():
            QMessageBox.warning(self, "Error", f"PONI file not found: {poni_file}")
            return

        self.btn_load.setEnabled(False)
        self.btn_load.setText("Loading CIFs...")
        self.progress_bar.setValue(0)

        poni = load_poni(poni_file)
        ai = poni_to_integrator(poni)
        wl = ai.wavelength * 1e10

        self._pending_integration = {
            'data_dir': data_dir, 'poni_file': poni_file,
            'mask_file': mask_file, 'tif_pattern': self.txt_tif_pattern.text(),
            'substrate_dir': self.txt_substrate.text(),
        }

        self._phase_load_worker = PhaseLoadWorker(cif_dir, wl)
        self._phase_load_worker.progress.connect(self._on_phase_load_progress)
        self._phase_load_worker.done.connect(self._on_phases_loaded_then_integrate)
        self._phases_loading = True
        self._phase_load_worker.start()

    def _on_phases_loaded_then_integrate(self, phases, wavelength_A):
        self._phases_loading = False
        self.btn_load_cifs.setEnabled(True)
        self.btn_load_cifs.setText("Load CIFs Only")
        self.phases = phases
        self.wavelength_A = wavelength_A
        self._populate_phase_checkboxes()
        names = ', '.join(ph.name for ph in phases)
        print(f"[Phase Fitter] Loaded {len(phases)} phases at "
              f"{wavelength_A:.4f} Å: {names}")

        info = self._pending_integration
        self._pending_integration = None

        self.btn_load.setText("Integrating...")
        self.statusBar().showMessage("Phases loaded. Starting integration...")

        self.integration_worker = IntegrationWorker()
        self.integration_worker.data_dir = info['data_dir']
        self.integration_worker.poni_file = info['poni_file']
        self.integration_worker.mask_file = info['mask_file']
        self.integration_worker.tif_pattern = info['tif_pattern']
        self.integration_worker.substrate_dir = info['substrate_dir']

        self.integration_worker.progress.connect(self._on_integration_progress)
        self.integration_worker.done.connect(self._on_integration_done)
        self.integration_worker.error.connect(self._on_integration_error)
        self.integration_worker.start()

    def _on_integration_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))
        self.statusBar().showMessage(
            f"Integrating pattern {current + 1}/{total}...")

    def _on_integration_done(self, patterns, labels, template):
        self.patterns = patterns
        self.labels = labels
        self.fit_background_template = template
        self.results_cache.clear()

        self.cmb_pattern.blockSignals(True)
        self.cmb_pattern.clear()
        for i, lab in enumerate(labels):
            self.cmb_pattern.addItem(f"[{i}] {lab}")
        self.cmb_pattern.blockSignals(False)

        if labels:
            self.cmb_pattern.setCurrentIndex(0)
            self._on_pattern_changed(0)

        self.btn_load.setEnabled(True)
        self.btn_load.setText("Load && Integrate")
        self.progress_bar.setValue(100)
        self.statusBar().showMessage(
            f"Loaded {len(patterns)} patterns. "
            f"Template: {'yes' if template else 'no'}.")

    def _on_integration_error(self, msg):
        self.btn_load.setEnabled(True)
        self.btn_load.setText("Load && Integrate")
        QMessageBox.critical(self, "Integration Error", msg)
        self.statusBar().showMessage("Integration failed.")

    # -----------------------------------------------------------------
    # Watch Mode — periodic scan for new TIF files
    # -----------------------------------------------------------------

    def _toggle_watch_mode(self, state):
        if state == Qt.Checked:
            data_dir = self.txt_data_dir.text().strip()
            poni_file = self.txt_poni.text().strip()
            if not data_dir or not Path(data_dir).exists():
                QMessageBox.warning(self, "Watch Mode",
                    "Set a valid Data Dir before enabling watch mode.")
                self.chk_watch.setChecked(False)
                return
            if not poni_file or not Path(poni_file).exists():
                QMessageBox.warning(self, "Watch Mode",
                    "Set a valid PONI file before enabling watch mode.")
                self.chk_watch.setChecked(False)
                return

            reply = QMessageBox.information(self, "Watch Mode",
                "Watch mode will check for new TIF files every 60 seconds.\n\n"
                "When Fit All is active, new files will be automatically\n"
                "integrated and fitted as they appear.\n\n"
                "Memory usage will grow over time with accumulated data.",
                QMessageBox.Ok | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                self.chk_watch.setChecked(False)
                return

            # Snapshot current files so we only process new ones
            tif_pattern = self.txt_tif_pattern.text() or '*.tif'
            self._watch_seen_files = {
                str(f) for f in Path(data_dir).glob(tif_pattern)}

            self._watch_integration_info = {
                'data_dir': data_dir,
                'poni_file': poni_file,
                'mask_file': self.txt_mask.text().strip(),
                'tif_pattern': tif_pattern,
                'substrate_dir': self.txt_substrate.text().strip(),
            }

            self._watch_timer.start()
            self.statusBar().showMessage(
                f"Watch mode ON — monitoring {data_dir} "
                f"({len(self._watch_seen_files)} existing files)")
        else:
            self._watch_timer.stop()
            self._watch_fitting = False
            self.btn_fit_all.setText("Fit All")
            self.statusBar().showMessage("Watch mode OFF")

    def _watch_tick(self):
        info = self._watch_integration_info
        if info is None:
            return

        data_dir = Path(info['data_dir'])
        tif_pattern = info['tif_pattern']
        all_files = {str(f) for f in data_dir.glob(tif_pattern)}
        new_files = sorted(all_files - self._watch_seen_files)

        if not new_files:
            self.statusBar().showMessage(
                f"Watch: no new files ({len(self._watch_seen_files)} total)")
            return

        self._watch_seen_files.update(new_files)
        print(f"[Watch] Found {len(new_files)} new TIF files")
        self.statusBar().showMessage(
            f"Watch: integrating {len(new_files)} new files...")

        # Integrate new files in a worker
        worker = IntegrationWorker()
        worker.poni_file = info['poni_file']
        worker.mask_file = info['mask_file']
        worker.tif_pattern = tif_pattern
        worker.substrate_dir = ''
        # Override data_dir with a temp dir containing only new files —
        # instead, we set data_dir and filter in done handler.
        # Actually, IntegrationWorker globs data_dir, so we pass the real
        # dir but track which results are new via label count.
        worker.data_dir = info['data_dir']

        worker.done.connect(self._on_watch_integration_done)
        worker.error.connect(self._on_watch_integration_error)
        worker.progress.connect(self._on_integration_progress)
        self._watch_worker = worker
        worker.start()

    def _on_watch_integration_done(self, patterns, labels, template):
        # Only append patterns we haven't seen yet
        existing_labels = set(self.labels)
        new_count = 0
        for pat, lab in zip(patterns, labels):
            if lab not in existing_labels:
                self.patterns.append(pat)
                self.labels.append(lab)
                idx = len(self.patterns) - 1
                self.cmb_pattern.addItem(f"[{idx}] {lab}")
                new_count += 1

        if template and self.fit_background_template is None:
            self.fit_background_template = template

        msg = f"Watch: added {new_count} new patterns ({len(self.patterns)} total)"
        print(f"[Watch] {msg}")
        self.statusBar().showMessage(msg)

        # Auto-fit new patterns if watch fitting is active
        if self._watch_fitting and new_count > 0:
            self._watch_fit_new()

    def _on_watch_integration_error(self, msg):
        print(f"[Watch] Integration error: {msg}")
        self.statusBar().showMessage(f"Watch: integration error — {msg[:60]}")

    def _watch_fit_new(self):
        """Fit any patterns that don't have cached results yet."""
        unfitted = [i for i in range(len(self.patterns))
                    if i not in self.results_cache]
        if not unfitted or not self.phases:
            return

        config = self._build_config()
        patterns_to_fit = [self.patterns[i] for i in unfitted]
        labels_to_fit = [self.labels[i] for i in unfitted]

        self._watch_index_offset = unfitted[0]
        self.btn_fit_all.setText("Stop")
        self.progress_bar.setValue(0)

        self.fit_worker = FitWorker(self)
        self.fit_worker.configure_batch(
            patterns_to_fit, self.phases, config,
            sequential=self.chk_sequential.isChecked(),
            labels=labels_to_fit,
            fit_background_template=self.fit_background_template,
            use_fast_fit=self.chk_fast_fit.isChecked(),
        )
        self.fit_worker.single_done.connect(self._on_watch_single_done)
        self.fit_worker.batch_progress.connect(self._on_batch_progress)
        self.fit_worker.batch_done.connect(self._on_watch_batch_done)
        self.fit_worker.error.connect(self._on_fit_error)
        self.fit_worker.start()

    def _on_watch_single_done(self, idx, result, elapsed):
        real_idx = idx + self._watch_index_offset
        self.results_cache[real_idx] = (result, elapsed)
        if self.cmb_pattern.currentIndex() == real_idx:
            self._update_fit_plot(real_idx, result)
            self._update_results_table(result)

    def _on_watch_batch_done(self, store):
        offset = self._watch_index_offset
        for entry in store:
            idx = entry['index'] + offset
            result = entry['result']
            elapsed = entry['elapsed']
            self.results_cache[idx] = (result, elapsed)

        self._update_trend_plot()

        n_ok = sum(1 for e in store if e['success'])
        self.statusBar().showMessage(
            f"Watch fit: {n_ok}/{len(store)} converged "
            f"({len(self.results_cache)}/{len(self.patterns)} total)")

        if self._watch_fitting:
            self.btn_fit_all.setText("Stop")
        else:
            self.btn_fit_all.setText("Fit All")
        self.btn_fit_all.setEnabled(True)
        self.btn_fit_current.setEnabled(True)

        idx = self.cmb_pattern.currentIndex()
        if idx in self.results_cache:
            result, elapsed = self.results_cache[idx]
            if result is not None:
                self._update_fit_plot(idx, result)
                self._update_results_table(result)

    # -----------------------------------------------------------------
    # Live / File Mode switching
    # -----------------------------------------------------------------

    def _set_file_mode_enabled(self, enabled):
        for w in self._file_mode_widgets:
            w.setEnabled(enabled)

    # -----------------------------------------------------------------
    # Live Mode — PVA subscription
    # -----------------------------------------------------------------

    def _toggle_live_connection(self):
        if self._live_connected:
            self._disconnect_live()
        else:
            self._connect_live()

    def _connect_live(self):
        if not HAS_PVA:
            QMessageBox.warning(self, "Error", "pvaccess is not installed.")
            return

        address = self.txt_pv_address.text().strip()
        if not address:
            QMessageBox.warning(self, "Error", "Please enter a PV address.")
            return

        cif_dir = self.txt_cif_dir.text().strip()
        if not cif_dir or not Path(cif_dir).exists():
            QMessageBox.warning(self, "CIF Directory Required",
                "Please set a valid CIF directory before connecting.\n"
                "Phases need CIF files to define crystal structures.")
            return

        if self._phases_loading:
            QMessageBox.warning(self, "Please Wait",
                "CIF files are still loading. Please wait for them to finish.")
            return

        output_pv = f"{address}:phaseFit"
        selected = self.selected_phases
        if selected:
            phase_info = (f"{len(selected)} selected "
                          f"({', '.join(p.name for p in selected)})")
        elif self.phases:
            phase_info = (f"{len(self.phases)} available (none selected — "
                          f"check phases to enable fitting)")
        else:
            phase_info = "will load automatically when wavelength arrives"
        auto_fit = "enabled" if self.chk_auto_fit.isChecked() else "disabled"

        msg = (f"PV Address: {address}\n"
               f"Phases: {phase_info}\n"
               f"Broadcasting results to: {output_pv}\n"
               f"Auto-fit: {auto_fit}")

        reply = QMessageBox.information(
            self, "Connecting to Live Stream", msg,
            QMessageBox.Ok | QMessageBox.Cancel)
        if reply != QMessageBox.Ok:
            return

        try:
            print(f"[Phase Fitter] Connecting to {address}...")
            self._live_channel = pva.Channel(address, pva.PVA)
            self._live_channel.subscribe("update", self._pva_callback)
            self._live_channel.startMonitor()
            self._live_connected = True
            self._live_frame_count = 0
            self._live_rx_count = 0
            self._live_displayed_count = 0
            self._live_prev_params = None
            self._live_history.clear()
            self._latest_live_frame = None

            self._output_pv_address = output_pv
            self._init_output_pv()

            self._live_poll_timer.start()

            self._set_file_mode_enabled(False)
            self.btn_live_connect.setText("Disconnect")
            self.lbl_live_status.setText(f"Connected: {address}")
            out_msg = f" | Publishing to: {self._output_pv_address}" if self._output_pv_server else ""
            self.statusBar().showMessage(f"Connected to {address}{out_msg}")
            print(f"[Phase Fitter] Connected. Polling at "
                  f"{self._live_poll_timer.interval()} ms. "
                  f"Waiting for data...")
            if not self.phases:
                print(f"[Phase Fitter] No phases loaded yet — "
                      f"will load from CIF dir when wavelength arrives.")
        except Exception as e:
            self._set_file_mode_enabled(True)
            print(f"[Phase Fitter] Connection failed: {e}")
            QMessageBox.warning(self, "Connection Error", str(e))

    def _disconnect_live(self):
        self._live_poll_timer.stop()
        self._latest_live_frame = None
        if self._live_channel is not None:
            try:
                self._live_channel.unsubscribe("update")
                self._live_channel.stopMonitor()
            except Exception:
                pass
            self._live_channel = None
        if self._output_pv_server is not None:
            try:
                self._output_pv_server.stop()
            except Exception:
                pass
            self._output_pv_server = None

        if self._live_fit_worker is not None:
            self._live_fit_worker.blockSignals(True)
            self._live_fit_worker = None

        self._live_connected = False
        self._live_fitting = False
        self._live_pending = None
        self._live_prev_params = None
        self._live_frame_count = 0
        self._live_rx_count = 0
        self._live_displayed_count = 0
        self._live_history.clear()

        self._clear_fit_plot()
        self.trend_plot.clear()
        self._trend_curves = {}
        self._peak_marker_key = None
        self.curve_data.setData([], [])
        self.results_cache.clear()
        self.patterns = []
        self.labels = []
        self.cmb_pattern.clear()

        self._set_file_mode_enabled(True)
        self.btn_live_connect.setText("Connect")
        self.lbl_live_status.setText("Disconnected")
        self.statusBar().showMessage("Disconnected. All live state cleared.")
        print("[Phase Fitter] Disconnected. Live state reset.")

    def _init_output_pv(self):
        if not HAS_PVA:
            return
        try:
            self._output_pv_type = {
                'frame_number': pva.INT,
                'redchi': pva.DOUBLE,
                'phase_names': [pva.STRING],
                'phase_fractions': [pva.DOUBLE],
                'timeStamp': pva.PvTimeStamp(),
            }
            self._output_pv_server = pva.PvaServer()
            initial = pva.PvObject(self._output_pv_type)
            self._output_pv_server.addRecord(
                self._output_pv_address, initial, None)
            logger.info(f"Publishing fit results to: {self._output_pv_address}")
        except Exception as e:
            logger.error(f"Failed to init output PV: {e}")
            self._output_pv_server = None

    def _publish_fit_result(self, frame_number, result):
        if self._output_pv_server is None or self._output_pv_type is None:
            return
        try:
            fracs = result.phase_fractions()
            names = list(fracs.keys())
            values = [float(fracs[n]) for n in names]
            pv_obj = pva.PvObject(self._output_pv_type, {
                'frame_number': int(frame_number),
                'redchi': float(result.redchi),
                'phase_names': names,
                'phase_fractions': values,
                'timeStamp': pva.PvTimeStamp(time.time()),
            })
            self._output_pv_server.updateUnchecked(
                self._output_pv_address, pv_obj)
        except Exception as e:
            logger.warning(f"Failed to publish fit result: {e}")

    def _pva_callback(self, pv_object):
        try:
            q = np.array(pv_object['q_values'])
            intensity = np.array(pv_object['intensity'])
            sigma = None
            try:
                sig = pv_object['sigma']
                if sig is not None and len(sig) > 0:
                    sigma = np.array(sig)
            except (KeyError, Exception):
                pass
            wavelength_A = 0.0
            try:
                wavelength_A = float(pv_object['wavelength'])
            except (KeyError, Exception):
                pass
            frame_number = int(pv_object['frame_number'])
            self._live_rx_count += 1
            self._latest_live_frame = (q, intensity, sigma, frame_number, wavelength_A)
        except Exception as e:
            logger.warning(f"Error in PVA callback: {e}")

    def _poll_live_data(self):
        frame = self._latest_live_frame
        if frame is None:
            return
        self._latest_live_frame = None
        try:
            self._on_live_data(*frame)
        except Exception as e:
            print(f"[Phase Fitter] Error processing frame: {e}")
            self.statusBar().showMessage(f"Error: {e}")

    def _on_live_data(self, q, intensity, sigma, frame_number, wavelength_A=0.0):
        if sigma is None or len(sigma) == 0:
            sigma = np.sqrt(np.maximum(intensity, 1.0))

        if not self.phases and not self._phases_loading and wavelength_A > 0:
            cif_dir = self.txt_cif_dir.text().strip()
            if cif_dir and Path(cif_dir).exists():
                print(f"[Phase Fitter] Wavelength {wavelength_A:.4f} Å received "
                      f"from stream — loading CIFs...")
                self._load_phases_async(cif_dir, wavelength_A)

        self._live_displayed_count += 1
        dropped = self._live_rx_count - self._live_displayed_count

        status_parts = [f"Frame {frame_number}",
                        f"Rx {self._live_rx_count}"]
        if dropped > 0:
            status_parts.append(f"dropped {dropped}")
        if self._phases_loading:
            status_parts.append("loading CIFs...")
        elif not self.selected_phases:
            status_parts.append("no phases selected")
        elif self._live_fitting:
            status_parts.append("fitting...")
        self.lbl_live_status.setText(" | ".join(status_parts))

        self.patterns = [(q, intensity, sigma)]
        self.labels = [f'frame_{frame_number}']

        self.cmb_pattern.blockSignals(True)
        self.cmb_pattern.clear()
        self.cmb_pattern.addItem(f"Live: frame {frame_number}")
        self.cmb_pattern.blockSignals(False)

        self._update_data_plot(0)

        if (self.chk_auto_fit.isChecked()
                and self.selected_phases
                and not self._phases_loading):
            if self._live_fitting:
                self._live_pending = (q, intensity, sigma, frame_number)
            else:
                self._start_live_fit(q, intensity, sigma, frame_number)

    def _start_live_fit(self, q, intensity, sigma, frame_number):
        self._live_fitting = True
        self._live_pending = None
        self._live_fit_frame_number = frame_number

        config = self._build_config()
        init_kw = dict(config.init_kw)
        selected_phases = [
            p for p in self.phases
            if getattr(p, 'name', None) in config.phase_names
        ]

        fit_kw = dict(config.fit_kw)
        if self._live_prev_params is not None:
            fit_kw['params'] = self._live_prev_params

        use_fast = self.chk_fast_fit.isChecked()

        print(f"[Phase Fitter] Fitting frame {frame_number} "
              f"({len(selected_phases)} phases, "
              f"{'fast' if use_fast else 'lmfit'})...")

        self._live_fit_worker = LiveFitWorker(
            q, intensity, sigma, selected_phases,
            init_kw, fit_kw, config.min_intensity,
            use_fast, frame_number)
        self._live_fit_worker.done.connect(self._on_live_fit_done)
        self._live_fit_worker.start()

    def _on_live_fit_done(self, frame_number, result, elapsed, error_msg):
        self._live_fitting = False

        if error_msg:
            msg = (f"Frame {frame_number}: FAILED after {elapsed:.1f}s "
                   f"— {error_msg}")
            print(f"[Phase Fitter] {msg}")
            self.statusBar().showMessage(msg)
            self.lbl_live_status.setText(
                f"Frame {frame_number} | FAILED | moving to next frame")
        elif result is not None:
            if result.success:
                self._live_prev_params = deepcopy(result.params)

            self.results_cache[0] = (result, elapsed)
            self._update_fit_plot(0, result)
            self._update_results_table(result)

            fracs = result.phase_fractions()
            self._live_history.append({
                'frame': frame_number,
                'fracs': fracs,
                'redchi': result.redchi,
                'elapsed': elapsed,
                'result': result,
            })
            self._update_live_trend_plot()
            self._publish_fit_result(frame_number, result)

            nfev = getattr(result, 'nfev', getattr(
                getattr(result, 'lmfit_result', None), 'nfev', '?'))
            skipped = max(0, self._live_frame_count - frame_number)
            frac_str = ' '.join(f"{k}={v:.3f}" for k, v in fracs.items())
            msg = (f"Frame {frame_number}: redchi={result.redchi:.1f} "
                   f"nfev={nfev} {elapsed:.2f}s {frac_str}")
            if skipped > 0:
                msg += f" [skipped {skipped}]"
            self.statusBar().showMessage(msg)
            print(f"[Phase Fitter] {msg}")

        pending = self._live_pending
        if pending is not None and self.chk_auto_fit.isChecked():
            self._start_live_fit(*pending)

    def _update_live_trend_plot(self):
        if not self._live_history:
            return

        phase_names = list(self._live_history[-1]['fracs'].keys())
        frames = np.array([h['frame'] for h in self._live_history])

        if not hasattr(self, '_trend_curves') or \
                set(self._trend_curves.keys()) != set(phase_names):
            self.trend_plot.clear()
            self.trend_plot.addLegend(offset=(60, 10))
            self.trend_plot.setYRange(-0.05, 1.05)
            self._trend_curves = {}
            for pi, name in enumerate(phase_names):
                color = get_phase_color(name, pi)
                curve = self.trend_plot.plot(
                    [], [], pen=pg.mkPen(color, width=2),
                    name=name)
                self._trend_curves[name] = curve

        for name, curve in self._trend_curves.items():
            y = np.array([h['fracs'].get(name, 0) for h in self._live_history])
            curve.setData(frames, y)

    def _on_max_history_changed(self, value):
        self._max_history = value
        old = list(self._live_history)
        self._live_history = deque(old[-value:], maxlen=value)

    # -----------------------------------------------------------------
    # Pattern Display
    # -----------------------------------------------------------------

    def _on_pattern_changed(self, idx):
        if idx < 0 or idx >= len(self.patterns):
            return
        self._update_data_plot(idx)
        if idx in self.results_cache:
            result, elapsed = self.results_cache[idx]
            self._update_fit_plot(idx, result)
            self._update_results_table(result)
            self.statusBar().showMessage(
                f"Pattern {idx}: redchi={result.redchi:.3g}, "
                f"elapsed={elapsed:.1f}s, "
                f"{'OK' if result.success else 'FAIL'}")
        else:
            self._clear_fit_plot()
            self._clear_results_table()

    def _update_data_plot(self, idx):
        q, y = self.patterns[idx][0], self.patterns[idx][1]
        self.curve_data.setData(q, y)

        current_phases = tuple(ph.name for ph in self.selected_phases)
        q_min, q_max = q.min(), q.max()
        q_range = (round(q_min, 2), round(q_max, 2))

        if not hasattr(self, '_peak_marker_key') or \
                self._peak_marker_key != (current_phases, q_range):
            for m in self.peak_markers:
                self.main_plot.removeItem(m)
            self.peak_markers.clear()

            for pi, ph in enumerate(self.selected_phases):
                color = get_phase_color(ph.name, pi)
                for pk in ph.peaks:
                    if pk.intensity > 1 and q_min <= pk.q <= q_max:
                        line = pg.InfiniteLine(
                            pos=pk.q, angle=90,
                            pen=pg.mkPen(color, width=0.5, style=Qt.DotLine))
                        self.main_plot.addItem(line)
                        self.peak_markers.append(line)
            self._peak_marker_key = (current_phases, q_range)

    def _update_fit_plot(self, idx, result):
        fitter = result.fitter
        params = result.params
        q = fitter.x

        y_model = fitter.eval_model(params)
        self.curve_fit.setData(q, y_model)

        bg = fitter.background.copy()
        bg_fit = fitter.eval_fit_background(params)
        if bg_fit is not None:
            bg = bg + bg_fit
        self.curve_bg.setData(q, bg)

        am = fitter.eval_amorphous(params)
        if am is not None:
            self.curve_amorphous.setData(q, am + bg)
        else:
            self.curve_amorphous.setData([], [])

        for c in self.phase_curves:
            self.main_plot.removeItem(c)
        self.phase_curves.clear()

        for pi, ph in enumerate(fitter.phases):
            color = get_phase_color(ph.name, pi)
            y_ph = fitter.eval_phase(pi, params)
            curve = self.main_plot.plot(
                q, y_ph + bg,
                pen=pg.mkPen(color, width=1.5),
                name=ph.name)
            self.phase_curves.append(curve)

        y_fit_model = fitter.composite.eval(params=params, x=q)
        resid = fitter.y_fit - y_fit_model
        self.curve_resid.setData(q, resid)

    def _clear_fit_plot(self):
        self.curve_fit.setData([], [])
        self.curve_bg.setData([], [])
        self.curve_amorphous.setData([], [])
        self.curve_resid.setData([], [])
        for c in self.phase_curves:
            self.main_plot.removeItem(c)
        self.phase_curves.clear()

    def _update_results_table(self, result):
        fracs = result.phase_fractions()
        self.results_table.setRowCount(len(result.fitter.phases))
        for i, ph in enumerate(result.fitter.phases):
            lp = result.lattice_params(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(ph.name))
            self.results_table.setItem(i, 1, QTableWidgetItem(
                f"{fracs.get(ph.name, 0):.4f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(
                f"{lp.get('a', 0):.5f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(
                f"{lp.get('b', 0):.5f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(
                f"{lp.get('c', 0):.5f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(
                f"{result.phase_scale(i):.4g}"))
            if i == 0:
                self.results_table.setItem(i, 6, QTableWidgetItem(
                    f"{result.redchi:.4g}"))

    def _clear_results_table(self):
        self.results_table.setRowCount(0)

    # -----------------------------------------------------------------
    # Trend Plot (File Mode)
    # -----------------------------------------------------------------

    def _update_trend_plot(self):
        self.trend_plot.clear()
        self.trend_plot.addLegend(offset=(60, 10))
        self.trend_plot.setYRange(-0.05, 1.05)

        if not self.results_cache:
            return

        indices = sorted(self.results_cache.keys())
        if not self.phases:
            return

        phase_names = [ph.name for ph in self.selected_phases]
        fracs_by_phase = {name: [] for name in phase_names}
        valid_indices = []

        for idx in indices:
            result, _ = self.results_cache[idx]
            if result is None:
                continue
            fracs = result.phase_fractions()
            valid_indices.append(idx)
            for name in phase_names:
                fracs_by_phase[name].append(fracs.get(name, 0))

        x = np.array(valid_indices)
        for pi, name in enumerate(phase_names):
            y = np.array(fracs_by_phase[name])
            color = get_phase_color(name, pi)
            self.trend_plot.plot(
                x, y,
                pen=pg.mkPen(color, width=2),
                symbol='o', symbolSize=5,
                symbolBrush=color, name=name)

    # -----------------------------------------------------------------
    # Config Build / Save / Load
    # -----------------------------------------------------------------

    def _build_config(self):
        prefit = self.cmb_prefit_bg.currentText()
        fit_bg = self.cmb_fit_bg.currentText()
        amorphous = self.cmb_amorphous.currentText()

        init_kw = {}
        if prefit != 'none':
            init_kw['prefit_background'] = prefit
            if prefit == 'snip':
                init_kw['prefit_background_kwargs'] = {
                    'snip_width': self.spn_snip_width.value()
                }
        else:
            init_kw['prefit_background'] = 'none'

        if fit_bg != 'none':
            init_kw['fit_background'] = fit_bg

        if amorphous != 'none':
            init_kw['amorphous_peak'] = amorphous
            init_kw['amorphous_init'] = {'center': 1.58, 'sigma': 0.3}

        fit_kw = {
            'caglioti': self.chk_caglioti.isChecked(),
            'phase_profile': self.cmb_profile.currentText(),
            'lattice_pct': self.spn_lattice_pct.value(),
            'q_shift_bound': self.spn_qshift.value(),
            'max_nfev': self.spn_max_nfev.value(),
            'lock_cross_phase': self.chk_lock_cross.isChecked(),
            'texture': self.cmb_texture.currentText(),
            'width_max': self.spn_width_max.value(),
            'width_min': self.spn_width_min.value(),
        }

        phase_names = [ph.name for ph in self.selected_phases]

        return FitConfig(
            init_kw=init_kw,
            fit_kw=fit_kw,
            phase_names=phase_names,
            min_intensity=self.spn_min_intensity.value(),
        )

    def _apply_config(self, config):
        init_kw = config.init_kw
        fit_kw = config.fit_kw

        prefit = init_kw.get('prefit_background', 'none')
        idx = self.cmb_prefit_bg.findText(prefit)
        if idx >= 0:
            self.cmb_prefit_bg.setCurrentIndex(idx)

        if 'prefit_background_kwargs' in init_kw:
            sw = init_kw['prefit_background_kwargs'].get('snip_width', 30)
            self.spn_snip_width.setValue(sw)

        fb = init_kw.get('fit_background', 'none')
        idx = self.cmb_fit_bg.findText(fb)
        if idx >= 0:
            self.cmb_fit_bg.setCurrentIndex(idx)

        am = init_kw.get('amorphous_peak', 'none')
        idx = self.cmb_amorphous.findText(am)
        if idx >= 0:
            self.cmb_amorphous.setCurrentIndex(idx)

        profile = fit_kw.get('phase_profile', 'pseudovoigt')
        idx = self.cmb_profile.findText(profile)
        if idx >= 0:
            self.cmb_profile.setCurrentIndex(idx)

        texture = fit_kw.get('texture', 'none')
        idx = self.cmb_texture.findText(texture)
        if idx >= 0:
            self.cmb_texture.setCurrentIndex(idx)

        self.spn_lattice_pct.setValue(fit_kw.get('lattice_pct', 0.05))
        self.spn_qshift.setValue(fit_kw.get('q_shift_bound', 0.05))
        self.spn_max_nfev.setValue(fit_kw.get('max_nfev', 3000))
        self.chk_caglioti.setChecked(fit_kw.get('caglioti', True))
        self.chk_lock_cross.setChecked(fit_kw.get('lock_cross_phase', True))

        if 'width_max' in fit_kw:
            self.spn_width_max.setValue(fit_kw['width_max'])
        if 'width_min' in fit_kw:
            self.spn_width_min.setValue(fit_kw['width_min'])

        self.spn_min_intensity.setValue(config.min_intensity)

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Fit Config",
            str(Path(self.txt_data_dir.text()) / 'fit_config.json'),
            "JSON Files (*.json)")
        if path:
            config = self._build_config()
            config.save(path)
            self.statusBar().showMessage(f"Config saved: {path}")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Fit Config",
            str(Path(self.txt_data_dir.text()) / 'fit_config.json'),
            "JSON Files (*.json)")
        if path:
            try:
                config = FitConfig.load(path)
                self._apply_config(config)
                self.statusBar().showMessage(f"Config loaded: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load config: {e}")

    # -----------------------------------------------------------------
    # Fitting (File Mode)
    # -----------------------------------------------------------------

    def fit_current(self):
        idx = self.cmb_pattern.currentIndex()
        if idx < 0 or not self.patterns:
            self.statusBar().showMessage("No pattern selected.")
            return
        if not self.phases:
            self.statusBar().showMessage("No phases loaded.")
            return

        config = self._build_config()

        prev_params = None
        if self.chk_sequential.isChecked() and idx > 0 and (idx - 1) in self.results_cache:
            prev_result, _ = self.results_cache[idx - 1]
            if prev_result is not None and prev_result.success:
                prev_params = prev_result.params

        self.btn_fit_current.setEnabled(False)
        self.btn_fit_current.setText("Fitting...")
        self.statusBar().showMessage(f"Fitting pattern {idx}...")

        self.fit_worker = FitWorker(self)
        self.fit_worker.configure_single(
            self.patterns, self.phases, config, idx,
            fit_background_template=self.fit_background_template,
            prev_params=prev_params,
            use_fast_fit=self.chk_fast_fit.isChecked(),
        )
        self.fit_worker.single_done.connect(self._on_single_fit_done)
        self.fit_worker.error.connect(self._on_fit_error)
        self.fit_worker.start()

    def fit_all(self):
        # In watch mode, Fit All / Stop is a toggle
        if self.chk_watch.isChecked() and self._watch_fitting:
            self._watch_fitting = False
            self.btn_fit_all.setText("Fit All")
            self.statusBar().showMessage("Watch fitting stopped.")
            return

        if not self.patterns:
            self.statusBar().showMessage("No patterns loaded.")
            return
        if not self.phases:
            self.statusBar().showMessage("No phases loaded.")
            return

        if self.chk_watch.isChecked():
            self._watch_fitting = True
            self._watch_fit_new()
            return

        config = self._build_config()

        self.btn_fit_all.setEnabled(False)
        self.btn_fit_all.setText("Fitting All...")
        self.btn_fit_current.setEnabled(False)
        self.progress_bar.setValue(0)
        self.results_cache.clear()

        self.fit_worker = FitWorker(self)
        self.fit_worker.configure_batch(
            self.patterns, self.phases, config,
            sequential=self.chk_sequential.isChecked(),
            labels=self.labels,
            fit_background_template=self.fit_background_template,
            use_fast_fit=self.chk_fast_fit.isChecked(),
        )
        self.fit_worker.single_done.connect(self._on_batch_single_done)
        self.fit_worker.batch_progress.connect(self._on_batch_progress)
        self.fit_worker.batch_done.connect(self._on_batch_done)
        self.fit_worker.error.connect(self._on_fit_error)
        self.fit_worker.start()

    def _on_single_fit_done(self, idx, result, elapsed):
        self.results_cache[idx] = (result, elapsed)
        self.btn_fit_current.setEnabled(True)
        self.btn_fit_current.setText("Fit Current")

        if self.cmb_pattern.currentIndex() == idx:
            self._update_fit_plot(idx, result)
            self._update_results_table(result)

        self._update_trend_plot()
        self.statusBar().showMessage(
            f"Fit [{idx}] {self.labels[idx] if idx < len(self.labels) else ''}: "
            f"{'OK' if result.success else 'FAIL'} | "
            f"redchi={result.redchi:.3g} | {elapsed:.1f}s")

    def _on_batch_single_done(self, idx, result, elapsed):
        self.results_cache[idx] = (result, elapsed)
        if self.cmb_pattern.currentIndex() == idx:
            self._update_fit_plot(idx, result)
            self._update_results_table(result)

    def _on_batch_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))
        self.statusBar().showMessage(
            f"Batch fitting: {current}/{total}...")

    def _on_batch_done(self, store):
        self.store = store
        self.btn_fit_all.setEnabled(True)
        if self._watch_fitting:
            self.btn_fit_all.setText("Stop")
        else:
            self.btn_fit_all.setText("Fit All")
        self.btn_fit_current.setEnabled(True)
        self.progress_bar.setValue(100)

        for entry in store:
            idx = entry['index']
            result = entry['result']
            elapsed = entry['elapsed']
            self.results_cache[idx] = (result, elapsed)

        self._update_trend_plot()

        n_ok = sum(1 for e in store if e['success'])
        self.statusBar().showMessage(
            f"Batch complete: {n_ok}/{len(self.patterns)} patterns converged.")

        idx = self.cmb_pattern.currentIndex()
        if idx in self.results_cache:
            result, elapsed = self.results_cache[idx]
            if result is not None:
                self._update_fit_plot(idx, result)
                self._update_results_table(result)

    def _on_fit_error(self, msg):
        self.btn_fit_current.setEnabled(True)
        self.btn_fit_current.setText("Fit Current")
        self.btn_fit_all.setEnabled(True)
        if self._watch_fitting:
            self.btn_fit_all.setText("Stop")
        else:
            self.btn_fit_all.setText("Fit All")
        QMessageBox.critical(self, "Fit Error", msg)
        self.statusBar().showMessage(f"Fit error: {msg[:80]}")

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    def export_csv(self):
        if not self.results_cache and not self._live_history:
            self.statusBar().showMessage("No results to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            str(Path(self.txt_data_dir.text() or '.') / 'phase_fractions.csv'),
            "CSV Files (*.csv)")
        if not path:
            return

        store = FitResultStore()

        if self._live_history:
            for i, h in enumerate(self._live_history):
                store.append(h['result'], index=i,
                             label=f"frame_{h['frame']}",
                             elapsed=h['elapsed'])
        else:
            for idx in sorted(self.results_cache.keys()):
                result, elapsed = self.results_cache[idx]
                if result is not None:
                    label = self.labels[idx] if idx < len(self.labels) else str(idx)
                    store.append(result, index=idx, label=label, elapsed=elapsed)

        df = store.to_dataframe()
        df.to_csv(path, index=False)
        self.statusBar().showMessage(f"Exported {len(df)} rows to {path}")

    # -----------------------------------------------------------------
    # Public API for DashPVA Integration
    # -----------------------------------------------------------------

    def set_patterns(self, patterns, labels=None):
        self.patterns = patterns
        self.labels = labels or [str(i) for i in range(len(patterns))]
        self.results_cache.clear()

        self.cmb_pattern.blockSignals(True)
        self.cmb_pattern.clear()
        for i, lab in enumerate(self.labels):
            self.cmb_pattern.addItem(f"[{i}] {lab}")
        self.cmb_pattern.blockSignals(False)

        if self.labels:
            self.cmb_pattern.setCurrentIndex(0)
            self._on_pattern_changed(0)

    def closeEvent(self, event):
        self._disconnect_live()
        super().closeEvent(event)


# =========================================================================
# Entry Point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='XRD Phase Fitter')
    parser.add_argument('--pv-address', type=str, default=None,
                        help='PVA channel for live 1D data (e.g. pvapy:image:pyFAI)')
    parser.add_argument('--poni-file', type=str, default=None,
                        help='PONI file (for wavelength in file mode)')
    parser.add_argument('--cif-dir', type=str, default=None,
                        help='Directory with CIF files')
    parser.add_argument('--wavelength', type=float, default=None,
                        help='Wavelength in Angstroms (overrides PONI)')
    parser.add_argument('--config', type=str, default=None,
                        help='Fit config JSON file')
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    pg.setConfigOptions(antialias=True, background='w', foreground='k')

    window = PhaseFitApp(
        pv_address=args.pv_address,
        poni_file=args.poni_file,
        cif_dir=args.cif_dir,
        wavelength_A=args.wavelength,
        config_path=args.config,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
