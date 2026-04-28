#!/usr/bin/env python3
"""
ROI Calculator Dock (UI skeleton)

- UI-only implementation per spec. No analysis logic or file I/O yet.
- Provides widgets to select ROI A and ROI B from memory or file, choose operation,
  and show placeholder results. Basic signal wiring for enable/disable behavior.
"""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QRadioButton,
    QLineEdit,
    QPushButton,
    QComboBox,
    QLabel,
    QCheckBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QSpacerItem,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg
import h5py
import qtawesome as qta

from utils.roi_ops import extract_roi_stack, align_stacks, per_frame_mean

from viewer.core.docks.base_dock import BaseDock
import os


class ROICalcDock(BaseDock):
    """Dock providing UI for ROI calculation configuration.

    UI-only for now: no computation or file I/O. Implements enable/disable wiring
    for memory vs file selection and basic Close behavior.
    """

    def __init__(self, main_window, dock_area=Qt.RightDockWidgetArea, segment_name="other"):
        super().__init__(
            title="ROI Calculator",
            main_window=main_window,
            segment_name=segment_name,
            dock_area=dock_area,
            show=False,
        )
        self._build_ui()
        # Internal mappings for memory ROIs
        self._roi_by_id = {}
        # Try to subscribe to ROI manager updates to keep dropdowns in sync
        try:
            if hasattr(self.main_window, 'roi_manager') and self.main_window.roi_manager is not None:
                self.main_window.roi_manager.add_listener(self._on_roi_manager_event)
        except Exception:
            pass

    # ----- UI construction -----
    def _build_ui(self):
        # Central content widget and main vertical layout
        self.content = QWidget(self)
        self.content.setObjectName("content")
        self.vbox = QVBoxLayout(self.content)
        try:
            self.vbox.setContentsMargins(6, 6, 6, 6)
            self.vbox.setSpacing(6)
        except Exception:
            pass

        # Top-level Calculator Lock controls
        self.chkCalcLock = QCheckBox("Calculator Lock")
        self.chkCalcLock.setObjectName("chkCalcLock")
        self.chkCalcLock.setChecked(True)
        self.lblLockBanner = QLabel("Calculator lock enabled — ROIs are fixed")
        self.lblLockBanner.setObjectName("lblLockBanner")
        try:
            self.lblLockBanner.setStyleSheet(
                "background-color: #ffd6d6; color: #a94442; border: 1px solid #a94442; font-weight: bold; padding: 4px;"
            )
        except Exception:
            pass
        self.vbox.addWidget(self.chkCalcLock)
        self.vbox.addWidget(self.lblLockBanner)

        # ROI A group
        self.grpRoiA = QGroupBox("ROI A")
        self.grpRoiA.setObjectName("grpRoiA")
        vA = QVBoxLayout(self.grpRoiA)
        try:
            vA.setContentsMargins(6, 6, 6, 6)
            vA.setSpacing(6)
        except Exception:
            pass

        rowA_sel = QHBoxLayout()
        self.rdoAFromMemory = QRadioButton("From Memory")
        self.rdoAFromMemory.setObjectName("rdoAFromMemory")
        self.rdoAFromMemory.setChecked(True)
        self.rdoAFromFile = QRadioButton("From File")
        self.rdoAFromFile.setObjectName("rdoAFromFile")
        rowA_sel.addWidget(self.rdoAFromMemory)
        rowA_sel.addWidget(self.rdoAFromFile)
        vA.addLayout(rowA_sel)

        rowA_file = QHBoxLayout()
        self.txtAFile = QLineEdit()
        self.txtAFile.setObjectName("txtAFile")
        self.txtAFile.setPlaceholderText("Select ROI A file path")
        self.txtAFile.setEnabled(False)
        self.btnABrowse = QPushButton("Browse…")
        self.btnABrowse.setObjectName("btnABrowse")
        self.btnABrowse.setEnabled(False)
        rowA_file.addWidget(self.txtAFile)
        rowA_file.addWidget(self.btnABrowse)
        vA.addLayout(rowA_file)

        self.cboAName = QComboBox()
        self.cboAName.setObjectName("cboAName")
        vA.addWidget(self.cboAName)

        self.lblABounds = QLabel("Bounds: (not set)")
        self.lblABounds.setObjectName("lblABounds")
        vA.addWidget(self.lblABounds)
        # A status label reflecting lock state
        self.lblAStatus = QLabel("Status: (unknown)")
        self.lblAStatus.setObjectName("lblAStatus")
        vA.addWidget(self.lblAStatus)

        # ROI B group
        self.grpRoiB = QGroupBox("ROI B")
        self.grpRoiB.setObjectName("grpRoiB")
        vB = QVBoxLayout(self.grpRoiB)
        try:
            vB.setContentsMargins(6, 6, 6, 6)
            vB.setSpacing(6)
        except Exception:
            pass

        rowB_sel = QHBoxLayout()
        self.rdoBFromMemory = QRadioButton("From Memory")
        self.rdoBFromMemory.setObjectName("rdoBFromMemory")
        self.rdoBFromMemory.setChecked(True)
        self.rdoBFromFile = QRadioButton("From File")
        self.rdoBFromFile.setObjectName("rdoBFromFile")
        rowB_sel.addWidget(self.rdoBFromMemory)
        rowB_sel.addWidget(self.rdoBFromFile)
        vB.addLayout(rowB_sel)

        rowB_file = QHBoxLayout()
        self.txtBFile = QLineEdit()
        self.txtBFile.setObjectName("txtBFile")
        self.txtBFile.setPlaceholderText("Select ROI B file path")
        self.txtBFile.setEnabled(False)
        self.btnBBrowse = QPushButton("Browse…")
        self.btnBBrowse.setObjectName("btnBBrowse")
        self.btnBBrowse.setEnabled(False)
        rowB_file.addWidget(self.txtBFile)
        rowB_file.addWidget(self.btnBBrowse)
        vB.addLayout(rowB_file)

        self.cboBName = QComboBox()
        self.cboBName.setObjectName("cboBName")
        vB.addWidget(self.cboBName)

        self.lblBBounds = QLabel("Bounds: (not set)")
        self.lblBBounds.setObjectName("lblBBounds")
        vB.addWidget(self.lblBBounds)
        # B status label reflecting lock state
        self.lblBStatus = QLabel("Status: (unknown)")
        self.lblBStatus.setObjectName("lblBStatus")
        vB.addWidget(self.lblBStatus)

        # Operation group
        self.grpOperation = QGroupBox("Operation")
        self.grpOperation.setObjectName("grpOperation")
        vOp = QVBoxLayout(self.grpOperation)
        try:
            vOp.setContentsMargins(6, 6, 6, 6)
            vOp.setSpacing(6)
        except Exception:
            pass

        form = QFormLayout()
        lbl_op = QLabel("Operation:")
        self.cboOperation = QComboBox()
        self.cboOperation.setObjectName("cboOperation")
        self.cboOperation.addItem("Average Series: A − B")
        self.cboOperation.addItem("Pixel-wise: A − B")
        self.cboOperation.addItem("Custom Expression")
        # Helpful tooltips for operations
        try:
            # Combo tooltip
            self.cboOperation.setToolTip(
                "Select how to combine ROI A and ROI B for preview/results"
            )
            # Per-item tooltips
            self.cboOperation.setItemData(
                0,
                "Average over frames along the selected time axis for A and B,\n"
                "then compute A_mean − B_mean.",
                Qt.ToolTipRole,
            )
            self.cboOperation.setItemData(
                1,
                "Per-pixel subtraction: for each frame, compute A − B.",
                Qt.ToolTipRole,
            )
            self.cboOperation.setItemData(
                2,
                "Define a custom expression combining A and B (not implemented yet).",
                Qt.ToolTipRole,
            )
        except Exception:
            pass
        # Disable "Custom Expression" for now
        try:
            model = self.cboOperation.model()
            idx = self.cboOperation.model().index(2, 0)
            model.setData(idx, 0, Qt.UserRole - 1)  # disable item
        except Exception:
            pass
        form.addRow(lbl_op, self.cboOperation)
        vOp.addLayout(form)

        # Optional: add tooltips for related controls
        try:
            self.grpOperation.setToolTip(
                "Configure how the selected ROIs will be combined. UI-only for now."
            )
            self.chkStrictMatch.setToolTip(
                "Require A and B to have identical extents/frame counts."
            )
            self.chkAutoIntersect.setToolTip(
                "When Strict Match is off, automatically intersect A and B to their overlapping region."
            )
            self.cboTimeAxis.setToolTip(
                "For 3D stacks, choose which axis acts as time for averaging in 'Average Series'."
            )
            self.chkIncludeROIFrames.setToolTip(
                "Include individual ROI frame slices in results when applicable."
            )
        except Exception:
            pass

        row_match = QHBoxLayout()
        self.chkStrictMatch = QCheckBox("Strict Match")
        self.chkStrictMatch.setObjectName("chkStrictMatch")
        self.chkStrictMatch.setChecked(True)
        self.chkAutoIntersect = QCheckBox("Auto-Intersect")
        self.chkAutoIntersect.setObjectName("chkAutoIntersect")
        self.chkAutoIntersect.setDisabled(True)
        row_match.addWidget(self.chkStrictMatch)
        row_match.addWidget(self.chkAutoIntersect)
        vOp.addLayout(row_match)

        row_adv = QHBoxLayout()
        self.cboTimeAxis = QComboBox()
        self.cboTimeAxis.setObjectName("cboTimeAxis")
        self.cboTimeAxis.addItems(["H", "K", "L"])
        self.cboTimeAxis.setCurrentIndex(0)  # default H
        self.chkIncludeROIFrames = QCheckBox("Include ROI Frames")
        self.chkIncludeROIFrames.setObjectName("chkIncludeROIFrames")
        self.chkIncludeROIFrames.setChecked(False)
        row_adv.addWidget(self.cboTimeAxis)
        row_adv.addWidget(self.chkIncludeROIFrames)
        vOp.addLayout(row_adv)

        # Editable ROIs group (multi-select Allow drag)
        self.grpEditable = QGroupBox("Editable ROIs")
        self.grpEditable.setObjectName("grpEditable")
        self.vEditable = QVBoxLayout(self.grpEditable)
        try:
            self.vEditable.setContentsMargins(6, 6, 6, 6)
            self.vEditable.setSpacing(4)
        except Exception:
            pass
        # Placeholder until we populate
        self.lblEditablePlaceholder = QLabel("No ROIs in memory")
        self.lblEditablePlaceholder.setObjectName("lblEditablePlaceholder")
        self.vEditable.addWidget(self.lblEditablePlaceholder)
        # Map of roi id to checkbox
        self._editable_check_by_id = {}

        # Results tab widget
        self.tabResults = QTabWidget()
        self.tabResults.setObjectName("tabResults")

        # Series tab
        tab_series = QWidget()
        tab_series.setObjectName("tabSeries")
        v_series = QVBoxLayout(tab_series)
        try:
            v_series.setContentsMargins(6, 6, 6, 6)
            v_series.setSpacing(6)
        except Exception:
            pass
        # Series plot
        self.seriesPlot = pg.PlotWidget()
        self.seriesPlot.setBackground('k')
        try:
            self.seriesPlot.showGrid(x=True, y=True, alpha=0.3)
            self.seriesPlot.setLabel('bottom', 'Frame Index')
            self.seriesPlot.setLabel('left', 'A − B (mean)')
        except Exception:
            pass
        self._series_curve = None
        v_series.addWidget(self.seriesPlot)
        self.tblSeriesStats = QTableWidget(0, 2)
        self.tblSeriesStats.setObjectName("tblSeriesStats")
        try:
            self.tblSeriesStats.setHorizontalHeaderLabels(["Metric", "Value"])
        except Exception:
            pass
        v_series.addWidget(self.tblSeriesStats)
        self.tabResults.addTab(tab_series, "Series")

        # Image tab
        self.tab_image = QWidget()
        self.tab_image.setObjectName("tabImage")
        v_image = QVBoxLayout(self.tab_image)
        try:
            v_image.setContentsMargins(6, 6, 6, 6)
            v_image.setSpacing(6)
        except Exception:
            pass
        # Placeholder shown until first result
        self.lblImagePlaceholder = QLabel("Calculated ROI image/stack preview will appear here.\nFor 'Average Series', results are 1D and shown on the Series tab. 'Pixel-wise' will populate images.")
        self.lblImagePlaceholder.setObjectName("lblImagePlaceholder")
        self.lblImagePlaceholder.setWordWrap(True)
        v_image.addWidget(self.lblImagePlaceholder)
        # Add an ImageView to show computed results (hidden until available)
        try:
            self.imageView = pg.ImageView()
        except Exception:
            self.imageView = None
        if self.imageView is not None:
            try:
                # Lock aspect for square pixels
                self.imageView.view.setAspectLocked(True)
            except Exception:
                pass
            # Start hidden until we have a result
            try:
                self.imageView.hide()
            except Exception:
                pass
            v_image.addWidget(self.imageView)
        self.tabResults.addTab(self.tab_image, "Image")

        # Actions row (aligned right)
        actions_row = QHBoxLayout()
        actions_row.addStretch(1)
        self.btnLoadROIs = QPushButton("Import ROIs…")
        self.btnLoadROIs.setObjectName("btnLoadROIs")
        self.btnCompute = QPushButton("Compute")
        self.btnCompute.setObjectName("btnCompute")
        self.btnSendToWorkspace = QPushButton("Send to 2D Workspace")
        self.btnSendToWorkspace.setObjectName("btnSendToWorkspace")
        self.btnSendToWorkspace.setEnabled(False)
        self.btnSaveAsH5 = QPushButton("Save As HDF5…")
        self.btnSaveAsH5.setObjectName("btnSaveAsH5")
        self.btnSaveAsH5.setEnabled(False)
        self.btnClear = QPushButton("Clear")
        self.btnClear.setObjectName("btnClear")
        self.btnClose = QPushButton("Close")
        self.btnClose.setObjectName("btnClose")
        actions_row.addWidget(self.btnLoadROIs)
        actions_row.addWidget(self.btnCompute)
        actions_row.addWidget(self.btnSendToWorkspace)
        actions_row.addWidget(self.btnSaveAsH5)
        actions_row.addWidget(self.btnClear)
        actions_row.addWidget(self.btnClose)

        # Status row
        status_row = QHBoxLayout()
        self.lblStatus = QLabel("")
        self.lblStatus.setObjectName("lblStatus")
        status_row.addWidget(self.lblStatus)
        status_row.addStretch(1)
        self.progCompute = QProgressBar()
        self.progCompute.setObjectName("progCompute")
        self.progCompute.hide()  # hidden by default
        self.progCompute.setRange(0, 0)  # indeterminate when shown
        status_row.addWidget(self.progCompute)

        # Assemble into main vbox
        self.vbox.addWidget(self.grpRoiA)
        self.vbox.addWidget(self.grpRoiB)
        self.vbox.addWidget(self.grpOperation)
        self.vbox.addWidget(self.grpEditable)
        self.vbox.addWidget(self.tabResults)
        self.vbox.addLayout(actions_row)
        # Add a spacer to push status to bottom if space allows
        self.vbox.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.vbox.addLayout(status_row)

        # Install central widget
        self.setWidget(self.content)

        # Wire up interactions
        self._wire_signals()
        # Initial population from memory
        try:
            self._refresh_memory_rois()
            self._refresh_editable_rois()
        except Exception:
            pass
        # Persist last compute result/meta for Send/Save actions
        self._last_image_result = None  # np.ndarray (2D or 3D)
        self._last_series_diff = None   # np.ndarray (1D)
        self._last_meta = {}

    # ----- Signal wiring -----
    def _wire_signals(self):
        # Enable/disable file path controls based on selection for ROI A
        def update_a_file_controls(checked: bool):
            self.txtAFile.setEnabled(checked)
            self.btnABrowse.setEnabled(checked)

        self.rdoAFromFile.toggled.connect(update_a_file_controls)
        # Initialize state
        update_a_file_controls(self.rdoAFromFile.isChecked())

        # Enable/disable file path controls based on selection for ROI B
        def update_b_file_controls(checked: bool):
            self.txtBFile.setEnabled(checked)
            self.btnBBrowse.setEnabled(checked)

        self.rdoBFromFile.toggled.connect(update_b_file_controls)
        # Initialize state
        update_b_file_controls(self.rdoBFromFile.isChecked())

        # Strict match toggles auto-intersect availability
        self.chkStrictMatch.toggled.connect(lambda v: self.chkAutoIntersect.setDisabled(bool(v)))

        # Close button hides the dock
        self.btnClose.clicked.connect(self.hide)

        # Populate bounds when selection changes for A/B
        try:
            self.cboAName.currentIndexChanged.connect(lambda _: (self._update_bounds_label('A'), self._update_status_label('A')))
            self.cboBName.currentIndexChanged.connect(lambda _: (self._update_bounds_label('B'), self._update_status_label('B')))
        except Exception:
            pass

        # Wire Compute
        try:
            self.btnCompute.clicked.connect(self._on_compute_clicked)
        except Exception:
            pass

        # Wire Send to Workspace and Save As HDF5
        try:
            self.btnSendToWorkspace.clicked.connect(self._on_send_to_workspace)
        except Exception:
            pass
        try:
            self.btnSaveAsH5.clicked.connect(self._on_save_as_hdf5)
        except Exception:
            pass

        # Calculator Lock toggle
        try:
            self.chkCalcLock.toggled.connect(self._on_calc_lock_toggled)
            # Initialize banner/lock state
            self._on_calc_lock_toggled(self.chkCalcLock.isChecked())
        except Exception:
            pass

    # ----- Compute handling -----
    def _selected_rois(self):
        a = self._lookup_selected_roi('A')
        b = self._lookup_selected_roi('B')
        return a, b

    def _set_status(self, text: str):
        try:
            self.lblStatus.setText(str(text))
        except Exception:
            pass

    def _populate_series_results(self, diff: np.ndarray, avgA: np.ndarray, avgB: np.ndarray, A_aligned: np.ndarray):
        # Metrics table
        try:
            T = int(diff.shape[0]) if diff is not None else 0
            # ROI pixels from aligned stack spatial dims
            roi_pixels = 0
            if A_aligned is not None and A_aligned.ndim == 3:
                roi_pixels = int(A_aligned.shape[1] * A_aligned.shape[2])
            elif A_aligned is not None and A_aligned.ndim == 2:
                roi_pixels = int(A_aligned.shape[0] * A_aligned.shape[1])

            metrics = []
            if T > 0:
                metrics = [
                    ("Frames", str(T)),
                    ("ROI pixels", str(roi_pixels)),
                    ("mean(A)", f"{float(np.mean(avgA)):.6g}" if avgA is not None else "n/a"),
                    ("mean(B)", f"{float(np.mean(avgB)):.6g}" if avgB is not None else "n/a"),
                    ("mean(diff)", f"{float(np.mean(diff)):.6g}"),
                    ("min(diff)", f"{float(np.min(diff)):.6g}"),
                    ("max(diff)", f"{float(np.max(diff)):.6g}"),
                    ("std(diff)", f"{float(np.std(diff)):.6g}"),
                ]
            self.tblSeriesStats.setRowCount(len(metrics))
            for r, (k, v) in enumerate(metrics):
                try:
                    self.tblSeriesStats.setItem(r, 0, QTableWidgetItem(k))
                    self.tblSeriesStats.setItem(r, 1, QTableWidgetItem(v))
                except Exception:
                    pass
        except Exception:
            pass

        # Plot
        try:
            self.seriesPlot.clear()
            if diff is not None and diff.size > 0:
                x = np.arange(diff.shape[0], dtype=float)
                symbol = 'o' if diff.shape[0] <= 1 else None
                self._series_curve = self.seriesPlot.plot(x, diff, pen=pg.mkPen('y', width=2), symbol=symbol, symbolSize=8, symbolBrush='r')
        except Exception:
            pass

    def _on_compute_clicked(self):
        try:
            # Support both operations: Average Series and Pixel-wise
            roiA, roiB = self._selected_rois()
            if roiA is None or roiB is None:
                self._set_status("Please select ROI A and ROI B from memory")
                return
            # Extract stacks
            A = extract_roi_stack(self.main_window, roiA)
            B = extract_roi_stack(self.main_window, roiB)
            if A is None or B is None:
                self._set_status("Failed to extract ROI stacks from current data")
                return

            strict = bool(self.chkStrictMatch.isChecked())
            auto_intersect = bool(self.chkAutoIntersect.isChecked())
            A_al, B_al, info = align_stacks(A, B, strict=strict, auto_intersect=auto_intersect)
            if not info.get("ok", False):
                self._set_status("Alignment error: " + "; ".join(info.get("notes", [])))
                return
            # Compute per-frame series for context (used in both operations)
            avgA = per_frame_mean(A_al)
            avgB = per_frame_mean(B_al)
            if avgA is None or avgB is None:
                self._set_status("Unable to compute per-frame means")
                return
            T = min(len(avgA), len(avgB))
            series_diff = (avgA[:T] - avgB[:T]).astype(np.float32)

            # Operation-specific image result
            op_idx = None
            try:
                op_idx = int(self.cboOperation.currentIndex())
            except Exception:
                op_idx = 0
            image_result = None
            if op_idx == 0:
                # Average Series: produce 2D mean(A) - mean(B)
                try:
                    # A_al and B_al are normalized to 3D (T,H,W). Mean over frames -> (H,W)
                    meanA_img = np.mean(np.asarray(A_al), axis=0)
                    meanB_img = np.mean(np.asarray(B_al), axis=0)
                    image_result = (meanA_img - meanB_img).astype(np.float32)
                except Exception as e:
                    self._set_status(f"Error computing mean-difference image: {e}")
                    image_result = None
            elif op_idx == 1:
                # Pixel-wise: framewise A - B (2D or 3D stack)
                try:
                    image_result = (np.asarray(A_al, dtype=np.float32) - np.asarray(B_al, dtype=np.float32)).astype(np.float32)
                except Exception as e:
                    self._set_status(f"Error computing pixel-wise diff: {e}")
                    image_result = None
            else:
                self._set_status("Selected operation not implemented yet.")
                return

            # Update Series tab
            self._populate_series_results(series_diff, avgA[:T], avgB[:T], A_al)

            # Update Image tab if we have an image/stack
            self._last_image_result = image_result
            self._last_series_diff = series_diff
            self._last_meta = {
                'operation': 'Average Series: A − B' if op_idx == 0 else 'Pixel-wise: A − B',
                'roiA_name': getattr(self.main_window.roi_manager, 'get_roi_name', lambda r: 'ROI A')(roiA),
                'roiB_name': getattr(self.main_window.roi_manager, 'get_roi_name', lambda r: 'ROI B')(roiB),
                'strict': bool(strict),
                'auto_intersect': bool(auto_intersect),
                'notes': info.get('notes', []),
            }

            try:
                if image_result is not None and self.imageView is not None:
                    # Show image view and hide placeholder
                    try:
                        self.lblImagePlaceholder.hide()
                        self.imageView.show()
                    except Exception:
                        pass
                    auto_levels = False
                    try:
                        auto_levels = False  # keep as-is to avoid histogram jump; adjust as needed
                    except Exception:
                        auto_levels = False
                    # ImageView supports 2D or 3D stacks (frames along axis 0)
                    self.imageView.setImage(np.asarray(image_result, dtype=np.float32), autoLevels=auto_levels, autoRange=True)
                    # Switch to Image tab
                    try:
                        self.tabResults.setCurrentWidget(self.tab_image)
                    except Exception:
                        pass
            except Exception:
                pass

            # Update status and enable actions
            notes = info.get("notes", [])
            status = " | ".join(notes) if notes else "Compute finished"
            self._set_status(status)
            try:
                self.btnSendToWorkspace.setEnabled(image_result is not None)
                self.btnSaveAsH5.setEnabled(image_result is not None)
            except Exception:
                pass
        except Exception as e:
            try:
                self._set_status(f"Compute failed: {e}")
            except Exception:
                pass

    # ----- Actions: Send / Save -----
    def _on_send_to_workspace(self):
        try:
            data = getattr(self, '_last_image_result', None)
            if data is None:
                self._set_status("No result to send")
                return
            if hasattr(self.main_window, 'display_2d_data'):
                self.main_window.display_2d_data(data)
                # Attempt to switch main to 2D analysis tab if present
                try:
                    if hasattr(self.main_window, 'tabWidget_analysis'):
                        self.main_window.tabWidget_analysis.setCurrentIndex(0)
                except Exception:
                    pass
                # After sending result to workspace, hide all ROIs (no disk deletion)
                try:
                    rm = getattr(self.main_window, 'roi_manager', None)
                    if rm is not None:
                        for r in list(getattr(rm, 'rois', [])):
                            rm.set_roi_visibility(r, False)
                except Exception:
                    pass
                self._set_status("Result sent to 2D workspace")
        except Exception as e:
            self._set_status(f"Send failed: {e}")

    def _on_save_as_hdf5(self):
        try:
            data = getattr(self, '_last_image_result', None)
            if data is None:
                self._set_status("No result to save")
                return
            # Ask for a save path; default to current file directory/name
            save_path = None
            try:
                from PyQt5.QtWidgets import QFileDialog
                default_dir = os.path.dirname(getattr(self.main_window, 'current_file_path', '') or '')
                fname, _ = QFileDialog.getSaveFileName(self, "Save As HDF5", os.path.join(default_dir or '', "roi_calc_result.h5"), "HDF5 Files (*.h5 *.hdf5);;All Files (*)")
                save_path = fname if isinstance(fname, str) and fname else None
            except Exception:
                save_path = getattr(self.main_window, 'current_file_path', None)
            if not save_path:
                self._set_status("Save canceled")
                return

            arr = np.asarray(data, dtype=np.float32)
            try:
                with h5py.File(save_path, 'a') as h5f:
                    entry = h5f.require_group('entry')
                    data_grp = entry.require_group('data')
                    # Replace existing 'result' dataset
                    if 'result' in data_grp:
                        try:
                            del data_grp['result']
                        except Exception:
                            pass
                    dset = data_grp.create_dataset('result', data=arr, dtype=np.float32)
                    # Attach basic metadata
                    meta = getattr(self, '_last_meta', {}) or {}
                    try:
                        dset.attrs['operation'] = str(meta.get('operation', ''))
                        dset.attrs['roiA_name'] = str(meta.get('roiA_name', 'ROI A'))
                        dset.attrs['roiB_name'] = str(meta.get('roiB_name', 'ROI B'))
                        dset.attrs['strict'] = bool(meta.get('strict', True))
                        dset.attrs['auto_intersect'] = bool(meta.get('auto_intersect', False))
                        # Store notes as a single UTF-8 string joined by '\n'
                        notes = meta.get('notes', [])
                        if isinstance(notes, (list, tuple)):
                            notes_str = "\n".join(map(str, notes))
                        else:
                            notes_str = str(notes)
                        dt = h5py.string_dtype(encoding='utf-8')
                        dset.attrs.create('notes', np.array(notes_str, dtype=dt))
                    except Exception:
                        pass
                self._set_status(f"Saved result to {save_path}:/entry/data/result")
            except Exception as e:
                self._set_status(f"Save failed: {e}")
        except Exception as e:
            # Catch any errors not handled by the inner try/except
            try:
                self._set_status(f"Save failed: {e}")
            except Exception:
                pass

    # ----- ROI memory helpers -----
    def _on_roi_manager_event(self, event: str, roi=None):
        """Listener callback wired into ROIManager to keep dropdowns current."""
        try:
            if event in ('added', 'deleted', 'renamed', 'cleared'):
                self._refresh_memory_rois()
                self._refresh_editable_rois()
            elif event in ('lock-changed',):
                # Sync editable checkboxes and status labels
                self._sync_editable_checks_to_lock_state()
                self._update_status_label('A')
                self._update_status_label('B')
        except Exception:
            pass

    def _refresh_memory_rois(self):
        """Populate the A/B combo boxes with in-memory ROIs.

        Display format for each entry is:
            "<filename without extension> | <roi_name>"

        Where:
        - File path is taken from rm.get_roi_source(r).get('file_path') if available,
          otherwise from main_window.current_file_path. If no path is available,
          the basename defaults to 'Untitled'.
        - The filename is reduced to its basename and last extension is removed
          (e.g., 'sample.v2.h5' -> 'sample.v2'). If removing the extension yields
          an empty string, the original basename is used instead.
        - roi_name comes from ROIManager.get_roi_name(r), preserving user-renamed
          and dataset-based names.

        Notes:
        - The previous per-file counters and conditional dataset-name swapping
          have been removed. We always display the ROIManager-provided name after
          the pipe.
        - Examples:
            'sample.v2.h5' + 'ROI 1' -> 'sample.v2 | ROI 1'
            'myfile'       + 'ROI 2' -> 'myfile | ROI 2'
            missing path   + 'ROI 1' -> 'Untitled | ROI 1'
        """
        try:
            rm = getattr(self.main_window, 'roi_manager', None)
            if rm is None:
                return
            rois = list(getattr(rm, 'rois', []))
            # Reset mappings and combos
            self._roi_by_id = {}
            self.cboAName.clear()
            self.cboBName.clear()
            for r in rois:
                try:
                    src = rm.get_roi_source(r) if hasattr(rm, 'get_roi_source') else {}
                except Exception:
                    src = {}
                fp = src.get('file_path') or getattr(self.main_window, 'current_file_path', None)
                base = os.path.basename(fp) if isinstance(fp, str) and fp else 'Untitled'
                # Remove only the last extension; if that yields empty, fallback to basename
                try:
                    base_no_ext = os.path.splitext(base)[0]
                except Exception:
                    base_no_ext = base
                if not isinstance(base_no_ext, str) or base_no_ext.strip() == '':
                    base_no_ext = base
                # Get ROI display name from ROI manager (preserve user/dataset names)
                try:
                    roi_name = rm.get_roi_name(r) if hasattr(rm, 'get_roi_name') else None
                except Exception:
                    roi_name = None
                if not isinstance(roi_name, str) or roi_name.strip() == '':
                    roi_name = 'ROI'
                # Final display name per spec: "<basename_no_ext> | <roi_name>"
                disp_name = f"{base_no_ext} | {roi_name}"
                # Add entries
                rid = id(r)
                self._roi_by_id[rid] = r
                # Add to both A and B lists; store rid as user data
                try:
                    self.cboAName.addItem(disp_name, rid)
                except Exception:
                    self.cboAName.addItem(disp_name)
                try:
                    self.cboBName.addItem(disp_name, rid)
                except Exception:
                    self.cboBName.addItem(disp_name)
            # Update bounds labels after refresh
            self._update_bounds_label('A')
            self._update_bounds_label('B')
            self._update_status_label('A')
            self._update_status_label('B')
        except Exception:
            pass

    def _lookup_selected_roi(self, which: str):
        try:
            combo = self.cboAName if which == 'A' else self.cboBName
            idx = combo.currentIndex()
            if idx < 0:
                return None
            rid = None
            try:
                rid = combo.itemData(idx)
            except Exception:
                rid = None
            if rid in self._roi_by_id:
                return self._roi_by_id.get(rid)
            # Fallback: index-based
            rm = getattr(self.main_window, 'roi_manager', None)
            rois = list(getattr(rm, 'rois', [])) if rm else []
            return rois[idx] if 0 <= idx < len(rois) else None
        except Exception:
            return None

    def _roi_bounds_text(self, roi):
        try:
            if roi is None:
                return "Bounds: (not set)"
            pos = roi.pos(); size = roi.size()
            x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
            w = max(1, int(size.x())); h = max(1, int(size.y()))
            return f"Bounds: x={x0}, y={y0}, w={w}, h={h}"
        except Exception:
            return "Bounds: (not set)"

    def _update_bounds_label(self, which: str):
        try:
            roi = self._lookup_selected_roi(which)
            text = self._roi_bounds_text(roi)
            if which == 'A':
                self.lblABounds.setText(text)
            else:
                self.lblBBounds.setText(text)
        except Exception:
            pass

    # ----- Editable ROIs group -----
    def _clear_layout(self, layout: QVBoxLayout):
        try:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    try:
                        w.deleteLater()
                    except Exception:
                        pass
        except Exception:
            pass

    def _refresh_editable_rois(self):
        try:
            rm = getattr(self.main_window, 'roi_manager', None)
            rois = list(getattr(rm, 'rois', [])) if rm else []
            self._editable_check_by_id = {}
            self._clear_layout(self.vEditable)
            if not rois:
                self.vEditable.addWidget(self.lblEditablePlaceholder)
                return
            # Build a checkbox for each ROI: "<name> — Allow drag"
            for r in rois:
                rid = id(r)
                try:
                    name = rm.get_roi_name(r) if hasattr(rm, 'get_roi_name') else 'ROI'
                except Exception:
                    name = 'ROI'
                chk = QCheckBox(f"{name} — Allow drag")
                chk.setObjectName(f"chkAllowDrag_{rid}")
                # Checked means editable (unlocked)
                try:
                    is_locked = rm.is_locked(r) if hasattr(rm, 'is_locked') else True
                except Exception:
                    is_locked = True
                try:
                    chk.setChecked(not bool(is_locked))
                except Exception:
                    pass
                # Wire toggle
                def on_chk_toggled(checked, roi_ref=r):
                    try:
                        # If Calculator Lock is ON, ignore and revert
                        if self.chkCalcLock.isChecked():
                            # revert visual state to locked
                            prev = rm.is_locked(roi_ref) if hasattr(rm, 'is_locked') else True
                            self._set_checkbox_checked_safely(roi_ref, not prev)
                            return
                        if checked:
                            if hasattr(rm, 'unlock_roi'):
                                rm.unlock_roi(roi_ref)
                        else:
                            if hasattr(rm, 'lock_roi'):
                                rm.lock_roi(roi_ref)
                        # Update status labels for A/B if affected
                        self._update_status_label('A')
                        self._update_status_label('B')
                    except Exception:
                        pass
                try:
                    chk.toggled.connect(on_chk_toggled)
                except Exception:
                    pass
                self._editable_check_by_id[rid] = chk
                self.vEditable.addWidget(chk)
            # Apply enable/disable based on Calculator Lock
            self._apply_editable_checks_enabled_state()
        except Exception:
            pass

    def _apply_editable_checks_enabled_state(self):
        try:
            locked = self.chkCalcLock.isChecked()
            for chk in list(self._editable_check_by_id.values()):
                try:
                    chk.setEnabled(not locked)
                except Exception:
                    pass
        except Exception:
            pass

    def _sync_editable_checks_to_lock_state(self):
        try:
            rm = getattr(self.main_window, 'roi_manager', None)
            for rid, chk in list(self._editable_check_by_id.items()):
                # Find ROI by rid
                roi = None
                try:
                    # rm.rois contains actual objects; match by id
                    for r in list(getattr(rm, 'rois', [])):
                        if id(r) == rid:
                            roi = r
                            break
                except Exception:
                    roi = None
                if roi is None:
                    continue
                try:
                    is_locked = rm.is_locked(roi) if hasattr(rm, 'is_locked') else True
                    self._set_checkbox_checked_safely(roi, not bool(is_locked))
                except Exception:
                    pass
            # Ensure enabled state matches Calculator Lock
            self._apply_editable_checks_enabled_state()
        except Exception:
            pass

    def _set_checkbox_checked_safely(self, roi, checked: bool):
        try:
            chk = self._editable_check_by_id.get(id(roi))
            if not chk:
                return
            try:
                chk.blockSignals(True)
                chk.setChecked(bool(checked))
            except Exception:
                pass
            try:
                chk.blockSignals(False)
            except Exception:
                pass
        except Exception:
            pass

    def _on_calc_lock_toggled(self, checked: bool):
        try:
            # Banner visibility
            try:
                self.lblLockBanner.setVisible(bool(checked))
            except Exception:
                pass
            rm = getattr(self.main_window, 'roi_manager', None)
            if rm is None:
                return
            if bool(checked):
                # Lock all ROIs and disable per-ROI edit controls
                try:
                    if hasattr(rm, 'lock_all_rois'):
                        rm.lock_all_rois()
                except Exception:
                    pass
            else:
                # Apply per-ROI checkbox states: checked -> unlock, unchecked -> lock
                try:
                    for rid, chk in list(self._editable_check_by_id.items()):
                        # Find ROI
                        roi = None
                        for r in list(getattr(rm, 'rois', [])):
                            if id(r) == rid:
                                roi = r
                                break
                        if roi is None:
                            continue
                        if chk.isChecked():
                            if hasattr(rm, 'unlock_roi'):
                                rm.unlock_roi(roi)
                        else:
                            if hasattr(rm, 'lock_roi'):
                                rm.lock_roi(roi)
                except Exception:
                    pass
            # Enable/disable checkboxes based on Calculator Lock
            self._apply_editable_checks_enabled_state()
            # Update status labels
            self._update_status_label('A')
            self._update_status_label('B')
        except Exception:
            pass

    # ----- A/B status labels -----
    def _update_status_label(self, which: str):
        try:
            rm = getattr(self.main_window, 'roi_manager', None)
            roi = self._lookup_selected_roi(which)
            status_lbl = self.lblAStatus if which == 'A' else self.lblBStatus
            if roi is None or rm is None:
                try:
                    status_lbl.setText("Status: (unknown)")
                except Exception:
                    pass
                return
            locked = True
            try:
                locked = rm.is_locked(roi) if hasattr(rm, 'is_locked') else True
            except Exception:
                locked = True
            try:
                icon = "🔒" if locked else "🔓"
                status_lbl.setText(f"Status: {icon} {'(locked)' if locked else '(editable)'}")
            except Exception:
                pass
        except Exception:
            pass


__all__ = ["ROICalcDock"]
