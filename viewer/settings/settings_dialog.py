import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QHBoxLayout, QDialogButtonBox, QFileDialog, QMessageBox
)
import toml
from pathlib import Path
import settings


class SettingsDialog(QDialog):
    """
    Minimal settings dialog for path configuration.
    Provides two directory pickers:
      - OUTPUT_PATH
      - LOG_PATH

    On Save, it updates the active TOML in place and calls settings.reload().
    """

    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setWindowTitle("Settings — Paths")

        # Inputs
        self.edit_output = QLineEdit(self)
        self.btn_browse_output = QPushButton("Browse…", self)
        self.edit_log = QLineEdit(self)
        self.btn_browse_log = QPushButton("Browse…", self)
        self.edit_toml = QLineEdit(self)
        self.btn_browse_toml = QPushButton("Browse…", self)

        # Prepopulate from current loaded settings (module-level or OO instance fallback)
        try:
            out_mod = getattr(settings, 'OUTPUT_PATH', None)
            out_obj = getattr(getattr(settings, 'SETTINGS', None), 'OUTPUT_PATH', None)
            out_val = out_mod or out_obj or ''
            if isinstance(out_val, str) and out_val:
                out_val = str(Path(out_val).expanduser())
            self.edit_output.setText(str(out_val or ''))
        except Exception:
            self.edit_output.setText('')
        try:
            log_mod = getattr(settings, 'LOG_PATH', None)
            log_obj = getattr(getattr(settings, 'SETTINGS', None), 'LOG_PATH', None)
            log_val = log_mod or log_obj or ''
            if isinstance(log_val, str) and log_val:
                log_val = str(Path(log_val).expanduser())
            self.edit_log.setText(str(log_val or ''))
        except Exception:
            self.edit_log.setText('')
        try:
            toml_val = getattr(settings, 'TOML_FILE', None) or ''
            self.edit_toml.setText(str(toml_val))
        except Exception:
            self.edit_toml.setText('')

        # Layouts
        form = QFormLayout()
        row1 = QHBoxLayout()
        row1.addWidget(self.edit_output)
        row1.addWidget(self.btn_browse_output)
        form.addRow("OUTPUT_PATH", row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.edit_log)
        row2.addWidget(self.btn_browse_log)
        form.addRow("LOG_PATH", row2)

        row3 = QHBoxLayout()
        row3.addWidget(self.edit_toml)
        row3.addWidget(self.btn_browse_toml)
        form.addRow("TOML Config", row3)

        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)

        root = QVBoxLayout()
        root.addLayout(form)
        root.addWidget(btns)
        self.setLayout(root)

        # Wire up
        self.btn_browse_output.clicked.connect(lambda: self._pick_dir(self.edit_output, "Select OUTPUT_PATH"))
        self.btn_browse_log.clicked.connect(lambda: self._pick_dir(self.edit_log, "Select LOG_PATH"))
        self.btn_browse_toml.clicked.connect(self._pick_toml)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)

    def _pick_dir(self, target_edit: QLineEdit, caption: str):
        start = target_edit.text().strip()
        path = QFileDialog.getExistingDirectory(self, caption, start or "")
        if path:
            target_edit.setText(path)

    def _pick_toml(self):
        start = self.edit_toml.text().strip()
        start_dir = str(Path(start).parent) if start else ""
        path, _ = QFileDialog.getOpenFileName(self, "Select TOML Config", start_dir, "TOML Files (*.toml);;All Files (*)")
        if path:
            self.edit_toml.setText(path)

    def _save(self):
        out_val = self.edit_output.text().strip()
        log_val = self.edit_log.text().strip()
        toml_val = self.edit_toml.text().strip()

        # If a new TOML file was selected, load it as the active config source first
        if toml_val and Path(toml_val).is_file():
            try:
                settings.set_locator(toml_val)
                settings.reload()
            except Exception:
                pass
        # Always keep TOML_FILE in sync with what the user entered
        try:
            settings.TOML_FILE = toml_val or None
        except Exception:
            pass

        # Update module-level settings immediately for runtime effect
        try:
            settings.OUTPUT_PATH = out_val or './outputs'
            settings.LOG_PATH = log_val or './logs'
            # Update default Settings instance too
            if hasattr(settings, 'SETTINGS'):
                setattr(settings.SETTINGS, 'OUTPUT_PATH', settings.OUTPUT_PATH)
                setattr(settings.SETTINGS, 'LOG_PATH', settings.LOG_PATH)
        except Exception:
            pass

        # Persist to TOML if a TOML source is active; otherwise skip persistence
        toml_path = settings.ensure_path()
        if toml_path and getattr(settings, 'SOURCE_TYPE', None) == 'toml':
            try:
                data = toml.load(toml_path)
            except Exception as e:
                QMessageBox.critical(self, "Load Failed", f"Failed to load TOML: {e}")
                return

            if out_val:
                data['OUTPUT_PATH'] = out_val
            else:
                data.pop('OUTPUT_PATH', None)
            if log_val:
                data['LOG_PATH'] = log_val
            else:
                data.pop('LOG_PATH', None)

            try:
                with open(toml_path, 'w') as f:
                    toml.dump(data, f)
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"Failed to save TOML: {e}")
                return

            # Reload from TOML so changes propagate consistently
            try:
                settings.reload()
            except Exception:
                pass

        self.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dlg = SettingsDialog()
    dlg.show()
    sys.exit(app.exec_())
