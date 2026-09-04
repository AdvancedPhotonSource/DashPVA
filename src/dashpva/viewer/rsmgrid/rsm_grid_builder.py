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

"""RSM Volume Builder -- standalone launcher (`DashPVA rsmgrid`).

Merges one or more scan HDF5 files into a single gridded reciprocal-space
volume via dashpva.utils.rsm_gridder.build_volume, and saves the result
through the existing HDF5Loader.save_vol_to_h5 volume format so it opens
directly in Workbench's 3D volume viewer.
"""
import sys
from datetime import datetime
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

import dashpva.settings as settings
from dashpva.gui import configure_app, ui_path
from dashpva.utils.hdf5_loader import HDF5Loader
from dashpva.utils.mask_manager import MaskManager
from dashpva.utils.rsm_gridder import (
    GridBounds,
    RSMMergeError,
    build_volume,
    detector_shapes_for_files,
    ensure_memory_available,
    estimate_grid_memory,
    list_monitor_candidates,
    volume_result_to_metadata,
)

NO_MONITOR = '(none)'
AXES = ('h', 'k', 'l')


class _CancelledError(Exception):
    """Raised from RSMGridWorker's progress callback to unwind build_volume
    cleanly when the user cancels -- build_volume has no cancellation hook of
    its own, but progress_cb is called synchronously from inside its loop, so
    raising here is sufficient to abort it."""


class RSMGridWorker(QThread):
    progress = pyqtSignal(int, int)   # frames_done, frames_total (approx, see build_volume)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, filenames, nx, ny, nz, use_mask, output_path,
                 mask_transposed=False, monitor_dataset=None,
                 fixed_bounds=None, parent=None):
        super().__init__(parent)
        self.filenames = list(filenames)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.use_mask = use_mask
        self.output_path = output_path
        self.mask_transposed = mask_transposed
        self.monitor_dataset = monitor_dataset
        self.fixed_bounds = fixed_bounds
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def _progress_cb(self, done, total):
        if self._cancel_requested:
            raise _CancelledError("Cancelled by user")
        self.progress.emit(done, total)

    def run(self):
        try:
            mask_manager = MaskManager() if self.use_mask else None
            if self.use_mask and (mask_manager.mask is None):
                self.log.emit("No active mask found -- proceeding without masking.")

            if self.fixed_bounds is None:
                extent = "the full range of the data"
            else:
                bounds = self.fixed_bounds
                extent = (
                    f"H {bounds.xmin:g}..{bounds.xmax:g}, "
                    f"K {bounds.ymin:g}..{bounds.ymax:g}, "
                    f"L {bounds.zmin:g}..{bounds.zmax:g}"
                )
            self.log.emit(
                f"Merging {len(self.filenames)} file(s) into a "
                f"{self.nx}x{self.ny}x{self.nz} volume over {extent}..."
            )
            result = build_volume(
                self.filenames, self.nx, self.ny, self.nz,
                use_mask=self.use_mask, mask_manager=mask_manager,
                mask_transposed=self.mask_transposed,
                monitor_dataset=self.monitor_dataset,
                fixed_bounds=self.fixed_bounds,
                progress_cb=self._progress_cb,
                warn=lambda msg: self.log.emit(f"WARNING: {msg}"),
            )
            metadata = volume_result_to_metadata(result)
            exclusions = (
                f"{result.num_points_excluded_by_mask} excluded by mask, "
                f"{result.num_points_excluded_nonfinite} non-finite"
            )
            if self.fixed_bounds is not None:
                exclusions += (
                    f", {result.num_points_out_of_range} outside the HKL range"
                )
            self.log.emit(
                f"Gridded {result.num_points_binned} point(s) ({exclusions}). "
                f"Writing {self.output_path}..."
            )
            ok = HDF5Loader().save_vol_to_h5(self.output_path, result.volume, metadata=metadata)
            if ok:
                self.finished.emit(True, f"Volume saved to {self.output_path}")
            else:
                self.finished.emit(False, f"Failed to write volume to {self.output_path}")
        except _CancelledError:
            self.finished.emit(False, "Cancelled by user.")
        except RSMMergeError as e:
            self.finished.emit(False, str(e))
        except Exception as e:
            self.finished.emit(False, f"Unexpected error: {e}")


class RSMGridBuilderDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi(ui_path("rsmgrid", "rsm_grid_builder.ui"), self)
        self.worker = None
        self._detector_shapes = []
        self._detector_shape_error = None

        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_remove_file.clicked.connect(self._remove_selected_file)
        self.btn_browse_output.clicked.connect(self._browse_output)
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_close.clicked.connect(self.close)

        self._per_axis_widgets = (
            self.lbl_nx, self.spn_nx, self.lbl_ny, self.spn_ny,
            self.lbl_nz, self.spn_nz,
        )
        self._range_boxes = {
            axis: (getattr(self, f'spn_{axis}min'), getattr(self, f'spn_{axis}max'))
            for axis in AXES
        }
        self.spn_resolution.setValue(
            getattr(settings, 'RSM_GRID_DEFAULT_RESOLUTION', 200))
        self.cmb_resolution_units.currentIndexChanged.connect(self._on_units_changed)
        self.chk_per_axis.toggled.connect(self._on_per_axis_toggled)
        self.chk_auto_range.toggled.connect(self._on_auto_range_toggled)
        for spinbox in (self.spn_resolution, self.spn_nx, self.spn_ny, self.spn_nz):
            spinbox.valueChanged.connect(self._update_memory_estimate)
        for low, high in self._range_boxes.values():
            low.valueChanged.connect(self._update_memory_estimate)
            high.valueChanged.connect(self._update_memory_estimate)
        self._on_per_axis_toggled()

        self._refresh_monitors()

        try:
            out_base = getattr(settings, 'OUTPUT_PATH', './outputs')
        except Exception:
            out_base = './outputs'
        default_name = f"rsm_volume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        self.txt_output_file.setText(str(Path(out_base) / default_name))
        self._update_memory_estimate()

    # ---------- Helpers ----------
    def _append_log(self, text):
        self.txt_log.append(text)

    def _filenames(self):
        return [self.lst_files.item(i).text() for i in range(self.lst_files.count())]

    @staticmethod
    def _format_bytes(value):
        value = float(value)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024.0 or unit == "TiB":
                return f"{value:.2f} {unit}"
            value /= 1024.0

    def _per_axis(self):
        return self.chk_per_axis.isChecked()

    def _on_units_changed(self, *_args):
        """The unit is a label on the same number -- only the suffix moves."""
        suffix = " " + self.cmb_resolution_units.currentText()
        for box in (self.spn_resolution, self.spn_nx, self.spn_ny, self.spn_nz):
            box.setSuffix(suffix)

    def _grid_dims(self):
        """Pixel counts: one number cubed, or per-axis when that's selected."""
        if self._per_axis():
            return self.spn_nx.value(), self.spn_ny.value(), self.spn_nz.value()
        resolution = self.spn_resolution.value()
        return resolution, resolution, resolution

    def _fixed_bounds(self):
        """The typed HKL box, or None to let build_volume derive it."""
        if self.chk_auto_range.isChecked():
            return None
        (hmin, hmax), (kmin, kmax), (lmin, lmax) = (
            (low.value(), high.value()) for low, high in
            (self._range_boxes[axis] for axis in AXES)
        )
        return GridBounds(hmin, hmax, kmin, kmax, lmin, lmax)

    def _memory_estimate(self):
        if self._detector_shape_error:
            raise RSMMergeError(self._detector_shape_error)
        return estimate_grid_memory(
            *self._grid_dims(), detector_shapes=self._detector_shapes
        )

    def _on_per_axis_toggled(self, *_args):
        """Seed the per-axis boxes from the single value, so the two controls
        never disagree at the moment of the switch."""
        checked = self._per_axis()
        if checked:
            resolution = self.spn_resolution.value()
            for spinbox in (self.spn_nx, self.spn_ny, self.spn_nz):
                spinbox.blockSignals(True)
                spinbox.setValue(resolution)
                spinbox.blockSignals(False)
        self.spn_resolution.setVisible(not checked)
        self.lbl_resolution.setVisible(not checked)
        for widget in self._per_axis_widgets:
            widget.setVisible(checked)
        self._update_memory_estimate()

    def _on_auto_range_toggled(self, checked):
        for low, high in self._range_boxes.values():
            low.setEnabled(not checked)
            high.setEnabled(not checked)
        self._update_memory_estimate()

    def _refresh_detector_shapes(self):
        try:
            self._detector_shapes = detector_shapes_for_files(self._filenames())
            self._detector_shape_error = None
        except RSMMergeError as exc:
            self._detector_shapes = []
            self._detector_shape_error = str(exc)

    def _update_memory_estimate(self, *_args):
        nx, ny, nz = self._grid_dims()
        bounds = self._fixed_bounds()
        self.preview_box.set_grid(
            (nx, ny, nz),
            bounds=None if bounds is None else
            (bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax,
             bounds.zmin, bounds.zmax),
        )
        try:
            estimate = self._memory_estimate()
        except RSMMergeError as exc:
            self.lbl_memory_estimate.setText(f"Memory estimate unavailable: {exc}")
            return
        self.lbl_memory_estimate.setText(
            f"{nx} x {ny} x {nz} = {nx * ny * nz:,} voxels. "
            f"Estimated peak RAM: {self._format_bytes(estimate.peak_bytes)}; "
            f"float32 output: {self._format_bytes(estimate.output_bytes)}"
        )

    # ---------- File list ----------
    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 'Select scan HDF5 file(s)', '', 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        existing = set(self._filenames())
        for f in files:
            if f not in existing:
                self.lst_files.addItem(f)
                existing.add(f)
        self._refresh_monitors()
        self._refresh_detector_shapes()
        self._update_memory_estimate()

    def _remove_selected_file(self):
        for item in self.lst_files.selectedItems():
            self.lst_files.takeItem(self.lst_files.row(item))
        self._refresh_monitors()
        self._refresh_detector_shapes()
        self._update_memory_estimate()

    def _refresh_monitors(self):
        """Repopulate the monitor combo from the first file's CA metadata,
        preserving the current choice when it's still available."""
        previous = self.cmb_monitor.currentText()
        filenames = self._filenames()
        candidates = list_monitor_candidates(filenames[0]) if filenames else []
        self.cmb_monitor.clear()
        self.cmb_monitor.addItem(NO_MONITOR)
        self.cmb_monitor.addItems(candidates)
        index = self.cmb_monitor.findText(previous)
        self.cmb_monitor.setCurrentIndex(index if index >= 0 else 0)

    def _browse_output(self):
        start_dir = str(getattr(settings, 'OUTPUT_PATH', './outputs'))
        fname, _ = QFileDialog.getSaveFileName(
            self, 'Select output HDF5 file', start_dir, 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if fname:
            self.txt_output_file.setText(fname)

    # ---------- Run ----------
    def _start(self):
        filenames = self._filenames()
        if not filenames:
            QMessageBox.warning(self, 'No Files', 'Add at least one scan HDF5 file.')
            return
        output_path = self.txt_output_file.text().strip()
        if not output_path:
            QMessageBox.warning(self, 'No Output', 'Choose an output file.')
            return
        if not self.chk_auto_range.isChecked():
            for axis in AXES:
                low, high = self._range_boxes[axis]
                if high.value() <= low.value():
                    QMessageBox.warning(
                        self, 'Invalid HKL Range',
                        f'{axis.upper()} max must be greater than {axis.upper()} min.')
                    return
        try:
            ensure_memory_available(self._memory_estimate())
        except RSMMergeError as exc:
            QMessageBox.critical(self, 'Unsafe Grid Size', str(exc))
            return
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self.progress_bar.setValue(0)
        self.txt_log.clear()
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        monitor = self.cmb_monitor.currentText()
        self.worker = RSMGridWorker(
            filenames, *self._grid_dims(),
            self.chk_use_mask.isChecked(), output_path,
            mask_transposed=self.chk_mask_transposed.isChecked(),
            monitor_dataset=None if monitor == NO_MONITOR else monitor,
            fixed_bounds=self._fixed_bounds(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _cancel(self):
        if self.worker is not None:
            self.worker.request_cancel()
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, done, total):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(done)

    def _on_finished(self, success, message):
        self._append_log(message)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if success:
            QMessageBox.information(self, 'RSM Volume Builder', message)
        else:
            QMessageBox.critical(self, 'RSM Volume Builder', message)
        if self.worker is not None:
            self.worker.wait()
            self.worker = None

    def closeEvent(self, event):
        # Mirror the explicit-teardown / thread.quit()-then-wait() discipline
        # PR #127 established for QThread cleanup elsewhere in DashPVA.
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel()
            self.worker.wait()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    configure_app(app)
    dlg = RSMGridBuilderDialog()
    dlg.show()
    app.exec_()


if __name__ == '__main__':
    main()
