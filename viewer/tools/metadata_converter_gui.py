import sys
import os
from pathlib import Path
from typing import List

from PyQt5 import uic
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QMessageBox
)

import h5py

from utils.metadata_converter import convert_files_or_dir


def is_already_formatted(h5_path: Path, base_group: str) -> bool:
    """Return True if base_group/HKL exists AND there is at least one dataset named
    'NAME' under base_group/HKL/motor_positions (recursively).
    """
    try:
        with h5py.File(str(h5_path), 'r') as h5:
            hkl_group = f"{base_group}/HKL"
            if hkl_group not in h5:
                return False
            motor_root = f"{hkl_group}/motor_positions"
            if motor_root not in h5:
                return False

            found_name = False

            def visitor(name, obj):
                nonlocal found_name
                if found_name:
                    return
                try:
                    # Check leaf name equals 'NAME' and object is a dataset
                    leaf = name.split('/')[-1]
                    if leaf == 'NAME' and isinstance(obj, h5py.Dataset):
                        found_name = True
                except Exception:
                    pass

            h5[motor_root].visititems(visitor)
            return found_name
    except Exception:
        return False


class MetadataConverterDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi('gui/tools/metadata_converter.ui', self)

        # Wire up buttons
        if hasattr(self, 'btn_browse_hdf5_file'):
            self.btn_browse_hdf5_file.clicked.connect(self._browse_hdf5_file)
        if hasattr(self, 'btn_browse_hdf5_dir'):
            self.btn_browse_hdf5_dir.clicked.connect(self._browse_hdf5_dir)
        if hasattr(self, 'btn_browse_toml'):
            self.btn_browse_toml.clicked.connect(self._browse_toml)
        if hasattr(self, 'btn_convert'):
            self.btn_convert.clicked.connect(self._convert)
        if hasattr(self, 'btn_close'):
            self.btn_close.clicked.connect(self.close)

        # Defaults (ensure they exist if UI changed)
        if hasattr(self, 'txt_base_group') and not self.txt_base_group.text():
            self.txt_base_group.setText('entry/data/metadata')
        if hasattr(self, 'chk_include'):
            self.chk_include.setChecked(True)
        if hasattr(self, 'chk_in_place'):
            self.chk_in_place.setChecked(True)

    # ---------- Browsers ----------
    def _browse_hdf5_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select HDF5 file', '', 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if fname and hasattr(self, 'txt_hdf5_path'):
            self.txt_hdf5_path.setText(fname)

    def _browse_hdf5_dir(self):
        dname = QFileDialog.getExistingDirectory(self, 'Select directory containing HDF5 files', '')
        if dname and hasattr(self, 'txt_hdf5_path'):
            self.txt_hdf5_path.setText(dname)

    def _browse_toml(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select TOML mapping file', '', 'TOML Files (*.toml);;All Files (*)')
        if fname and hasattr(self, 'txt_toml_path'):
            self.txt_toml_path.setText(fname)

    # ---------- Conversion ----------
    def _append_log(self, text: str):
        if hasattr(self, 'txt_log'):
            self.txt_log.append(text)

    def _validate_inputs(self) -> tuple:
        hdf5_path = self.txt_hdf5_path.text().strip() if hasattr(self, 'txt_hdf5_path') else ''
        toml_path = self.txt_toml_path.text().strip() if hasattr(self, 'txt_toml_path') else ''
        base_group = self.txt_base_group.text().strip() if hasattr(self, 'txt_base_group') else 'entry/data/metadata'
        include = bool(self.chk_include.isChecked()) if hasattr(self, 'chk_include') else True
        in_place = bool(self.chk_in_place.isChecked()) if hasattr(self, 'chk_in_place') else True

        if not toml_path:
            QMessageBox.warning(self, 'Missing TOML', 'Please select a TOML mapping file.')
            return '', '', '', False, False
        if not hdf5_path:
            QMessageBox.warning(self, 'Missing Source', 'Please select a HDF5 file or directory.')
            return '', '', '', False, False
        return hdf5_path, toml_path, base_group, include, in_place

    def _convert(self):
        hdf5_path, toml_path, base_group, include, in_place = self._validate_inputs()
        if not hdf5_path:
            return

        src = Path(hdf5_path)
        converted_count = 0
        skipped_count = 0
        errors: List[str] = []

        try:
            if src.is_file():
                # single file
                if is_already_formatted(src, base_group):
                    self._append_log(f"Skip (already formatted): {src}")
                    skipped_count += 1
                else:
                    try:
                        outputs = convert_files_or_dir(
                            toml_path=toml_path,
                            hdf5_path=str(src),
                            base_group=base_group,
                            include=include,
                            in_place=in_place,
                            recursive=False
                        )
                        converted_count += 1 if outputs else 0
                        self._append_log(f"Converted: {src}")
                    except Exception as e:
                        errors.append(f"{src}: {e}")
                        self._append_log(f"Error converting {src}: {e}")
            elif src.is_dir():
                # directory: recurse
                files = list(src.rglob('*.h5'))
                if not files:
                    self._append_log('No .h5 files found in directory.')
                for f in files:
                    if is_already_formatted(f, base_group):
                        skipped_count += 1
                        self._append_log(f"Skip (already formatted): {f}")
                        continue
                    try:
                        outputs = convert_files_or_dir(
                            toml_path=toml_path,
                            hdf5_path=str(f),
                            base_group=base_group,
                            include=include,
                            in_place=in_place,
                            recursive=False
                        )
                        converted_count += 1 if outputs else 0
                        self._append_log(f"Converted: {f}")
                    except Exception as e:
                        errors.append(f"{f}: {e}")
                        self._append_log(f"Error converting {f}: {e}")
            else:
                QMessageBox.critical(self, 'Invalid Path', 'The selected HDF5 path is not a file or directory.')
                return
        finally:
            summary = f"Converted {converted_count} HDF5 file(s)."
            if skipped_count:
                summary += f" Skipped {skipped_count} already formatted."
            if errors:
                summary += f" Errors: {len(errors)}"
            self._append_log(summary)
            QMessageBox.information(self, 'Conversion Summary', summary)


def main():
    app = QApplication(sys.argv)
    dlg = MetadataConverterDialog()
    dlg.show()
    app.exec_()


if __name__ == '__main__':
    main()
