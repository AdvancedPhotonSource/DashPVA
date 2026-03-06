import sys
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QMessageBox
)
import settings

try:
    from utils.hdf5_loader import HDF5Loader
except Exception:
    # Fallback import path when running directly
    import pathlib as _pathlib
    sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))
    from utils.hdf5_loader import HDF5Loader


def natural_key(s: str):
    """Split string into list of text and integer chunks for natural sorting."""
    import re
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.findall(r"\d+|\D+", s)]


def list_images(src: Path, patterns: List[str], recursive: bool) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for ptn in patterns:
            files.extend(src.rglob(ptn))
    else:
        for ptn in patterns:
            files.extend(src.glob(ptn))
    # Deduplicate and sort naturally by name
    uniq = sorted({f.resolve() for f in files if f.is_file()}, key=lambda p: natural_key(p.name))
    return uniq


def load_image(path: Path) -> np.ndarray:
    """Load an image file with Pillow, convert to single-channel float32 array."""
    from PIL import Image
    with Image.open(str(path)) as img:
        # Convert to 32-bit float grayscale ('F' mode)
        img = img.convert('F')
        arr = np.array(img, dtype=np.float32)
        return arr


def stack_images(paths: List[Path], log: callable) -> Tuple[np.ndarray, Tuple[int, int], List[Path]]:
    """Load and stack images to shape (N,H,W). Skip any that don't match first image size.

    Returns: (volume, shape_hw, used_paths)
    """
    used: List[Path] = []
    arrays: List[np.ndarray] = []
    shape_hw: Tuple[int, int] = (0, 0)
    for i, p in enumerate(paths):
        try:
            arr = load_image(p)
        except Exception as e:
            log(f"Skip (failed to load): {p} — {e}")
            continue
        if i == 0:
            shape_hw = (int(arr.shape[0]), int(arr.shape[1]))
        # Enforce identical shape
        if arr.shape != (shape_hw[0], shape_hw[1]):
            log(f"Skip (size mismatch): {p} — got {arr.shape}, expected {shape_hw}")
            continue
        arrays.append(arr.astype(np.float32, copy=False))
        used.append(p)
    if not arrays:
        return (np.zeros((0, 0, 0), dtype=np.float32), (0, 0), [])
    vol = np.stack(arrays, axis=0)
    return (vol, shape_hw, used)


class FileConvertDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi('gui/tools/file_convert.ui', self)

        # Wire buttons
        if hasattr(self, 'btn_browse_source'):
            self.btn_browse_source.clicked.connect(self._browse_source)
        if hasattr(self, 'btn_browse_output_file'):
            self.btn_browse_output_file.clicked.connect(self._browse_output_file)
        if hasattr(self, 'btn_browse_output_dir'):
            self.btn_browse_output_dir.clicked.connect(self._browse_output_dir)
        if hasattr(self, 'btn_convert'):
            self.btn_convert.clicked.connect(self._convert)
        if hasattr(self, 'btn_close'):
            self.btn_close.clicked.connect(self.close)

        # Defaults
        if hasattr(self, 'cmb_filter'):
            self.cmb_filter.setCurrentIndex(0)
        # Default to per-subfolder mode as requested
        if hasattr(self, 'chk_per_subfolder'):
            self.chk_per_subfolder.setChecked(True)
            # Sync output controls enable state
            try:
                self.chk_per_subfolder.stateChanged.connect(self._sync_output_mode)
            except Exception:
                pass
            self._sync_output_mode()

        # Pre-fill default OUTPUT_PATH when available
        try:
            out_base = getattr(settings, 'OUTPUT_PATH', './outputs')
        except Exception:
            out_base = './outputs'
        if hasattr(self, 'txt_output_dir') and not getattr(self, 'txt_output_dir').text().strip():
            try:
                self.txt_output_dir.setText(str(out_base))
            except Exception:
                pass
        if hasattr(self, 'txt_output_file') and not getattr(self, 'txt_output_file').text().strip():
            try:
                from datetime import datetime as _dt
                default_name = f"stack_{_dt.now().strftime('%Y%m%d_%H%M%S')}.h5"
                self.txt_output_file.setText(str(Path(out_base) / default_name))
            except Exception:
                pass

    # ---------- Browsers ----------
    def _browse_source(self):
        dname = QFileDialog.getExistingDirectory(self, 'Select source folder', '')
        if dname and hasattr(self, 'txt_source_dir'):
            self.txt_source_dir.setText(dname)

    def _browse_output_file(self):
        start_dir = str(getattr(settings, 'OUTPUT_PATH', './outputs'))
        fname, _ = QFileDialog.getSaveFileName(self, 'Select output HDF5 file', start_dir, 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if fname and hasattr(self, 'txt_output_file'):
            self.txt_output_file.setText(fname)

    def _browse_output_dir(self):
        start_dir = str(getattr(settings, 'OUTPUT_PATH', './outputs'))
        dname = QFileDialog.getExistingDirectory(self, 'Select output directory', start_dir)
        if dname and hasattr(self, 'txt_output_dir'):
            self.txt_output_dir.setText(dname)

    # ---------- Helpers ----------
    def _append_log(self, text: str):
        if hasattr(self, 'txt_log'):
            self.txt_log.append(text)
        else:
            print(text)

    def _sync_output_mode(self):
        """Enable/disable output file vs directory controls based on per-subfolder checkbox."""
        per_sub = bool(self.chk_per_subfolder.isChecked()) if hasattr(self, 'chk_per_subfolder') else True
        # File controls
        for name in ('lbl_out_file', 'txt_output_file', 'btn_browse_output_file'):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(not per_sub)
        # Directory controls
        for name in ('lbl_out_dir', 'txt_output_dir', 'btn_browse_output_dir'):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(per_sub)

    def _patterns_for_filter(self) -> List[str]:
        label = self.cmb_filter.currentText() if hasattr(self, 'cmb_filter') else 'All'
        label = (label or '').lower()
        if 'tiff' in label:
            return ['*.tif', '*.tiff']
        if 'png' in label:
            return ['*.png']
        if 'jpeg' in label or 'jpg' in label:
            return ['*.jpg', '*.jpeg']
        if 'bmp' in label:
            return ['*.bmp']
        # All supported
        return ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']

    def _validate(self) -> Tuple[Path, bool, bool, Path, Path]:
        src_dir = Path(self.txt_source_dir.text().strip()) if hasattr(self, 'txt_source_dir') else Path('')
        recursive = bool(self.chk_recursive.isChecked()) if hasattr(self, 'chk_recursive') else False
        per_sub = bool(self.chk_per_subfolder.isChecked()) if hasattr(self, 'chk_per_subfolder') else True
        out_file = Path(self.txt_output_file.text().strip()) if hasattr(self, 'txt_output_file') else Path('')
        out_dir = Path(self.txt_output_dir.text().strip()) if hasattr(self, 'txt_output_dir') else Path('')

        if not src_dir.exists() or not src_dir.is_dir():
            QMessageBox.warning(self, 'Missing Source', 'Please select a valid source folder.')
            return Path(''), False, False, Path(''), Path('')
        if per_sub:
            if not out_dir:
                # Use settings.OUTPUT_PATH as default when not provided
                out_dir = Path(str(getattr(settings, 'OUTPUT_PATH', './outputs')))
                if hasattr(self, 'txt_output_dir'):
                    try:
                        self.txt_output_dir.setText(str(out_dir))
                    except Exception:
                        pass
        else:
            if not out_file:
                # Use a default file path under OUTPUT_PATH
                base = Path(str(getattr(settings, 'OUTPUT_PATH', './outputs')))
                base.mkdir(parents=True, exist_ok=True)
                out_file = base / 'stack.h5'
                if hasattr(self, 'txt_output_file'):
                    try:
                        self.txt_output_file.setText(str(out_file))
                    except Exception:
                        pass
        return src_dir, recursive, per_sub, out_file, out_dir

    # ---------- Conversion ----------
    def _convert(self):
        src_dir, recursive, per_sub, out_file, out_dir = self._validate()
        if not src_dir:
            return
        patterns = self._patterns_for_filter()
        self._append_log(f"Source: {src_dir}")
        self._append_log(f"Recursive: {recursive}")
        self._append_log(f"Mode: {'per-subfolder' if per_sub else 'single file'}")
        self._append_log(f"Filter: {patterns}")

        try:
            if per_sub:
                # Ensure output directory exists
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    QMessageBox.critical(self, 'Output Error', f'Failed to create output directory:\n{out_dir}\n\n{e}')
                    return
                subdirs = [d for d in src_dir.iterdir() if d.is_dir()]
                if not subdirs:
                    self._append_log('No subfolders found under source directory.')
                total_written = 0
                for sub in sorted(subdirs, key=lambda p: natural_key(p.name)):
                    files = list_images(sub, patterns, recursive=False)  # immediate subdirectory only
                    if not files:
                        self._append_log(f"Skip (no images): {sub}")
                        continue
                    self._append_log(f"Converting {len(files)} image(s) from: {sub}")
                    vol, shape_hw, used = stack_images(files, self._append_log)
                    if vol.size == 0:
                        self._append_log(f"Skip (no usable images): {sub}")
                        continue
                    out_path = out_dir / f"{sub.name}.h5"
                    md = {
                        'source_folder': str(sub),
                        'file_list': [p.name for p in used],
                        'num_images': int(vol.shape[0]),
                        'original_shape': [int(shape_hw[0]), int(shape_hw[1])],
                    }
                    ok = HDF5Loader().save_vol_to_h5(str(out_path), vol, metadata=md)
                    if ok:
                        total_written += 1
                        self._append_log(f"Wrote: {out_path}")
                    else:
                        self._append_log(f"Error writing: {out_path}")
                summary = f"Per-subfolder: wrote {total_written} file(s)."
                self._append_log(summary)
                QMessageBox.information(self, 'Conversion Summary', summary)
            else:
                files = list_images(src_dir, patterns, recursive=recursive)
                if not files:
                    QMessageBox.warning(self, 'No Images Found', 'No images matching the selected filter were found.')
                    return
                self._append_log(f"Converting {len(files)} image(s) from: {src_dir}")
                vol, shape_hw, used = stack_images(files, self._append_log)
                if vol.size == 0:
                    QMessageBox.warning(self, 'No Usable Images', 'No images with a consistent shape could be loaded.')
                    return
                # Ensure parent directory exists
                try:
                    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                md = {
                    'source_folder': str(src_dir),
                    'file_list': [str(p.relative_to(src_dir)) for p in used],
                    'num_images': int(vol.shape[0]),
                    'original_shape': [int(shape_hw[0]), int(shape_hw[1])],
                }
                ok = HDF5Loader().save_vol_to_h5(str(out_file), vol, metadata=md)
                if ok:
                    self._append_log(f"Wrote: {out_file}")
                    QMessageBox.information(self, 'Conversion Summary', f"Wrote: {out_file}")
                else:
                    self._append_log(f"Error writing: {out_file}")
                    QMessageBox.critical(self, 'Write Failed', f"Failed to write HDF5: {out_file}")
        except Exception as e:
            self._append_log(f"Error: {e}")
            QMessageBox.critical(self, 'Conversion Error', str(e))


def main():
    app = QApplication(sys.argv)
    dlg = FileConvertDialog()
    dlg.show()
    app.exec_()


if __name__ == '__main__':
    main()
