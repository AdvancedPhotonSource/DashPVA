import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QHBoxLayout, QFileDialog, QMessageBox, QLabel, QFrame
)
from pathlib import Path
import settings


# ── Lock categories ────────────────────────────────────────────────────────────
# Change these lists to control which fields fall into each category.
#
#   ALWAYS_EDITABLE  – editable regardless of lock state
#   UNLOCKABLE       – locked by default; user must toggle unlock to edit
#   ALWAYS_LOCKED    – read-only display only, never editable
# ──────────────────────────────────────────────────────────────────────────────
ALWAYS_EDITABLE = ['output_path', 'log_path']
UNLOCKABLE      = ['toml_config']
ALWAYS_LOCKED   = []   # e.g. add 'output_path' here to make it read-only forever


class _ToggleLock(QPushButton):
    """Styled toggle button that switches between Locked / Unlocked states."""

    _LOCKED_STYLE = (
        "QPushButton {"
        "  background-color: #c0392b; color: white;"
        "  border-radius: 10px; padding: 4px 16px;"
        "  font-weight: bold; font-size: 12px;"
        "}"
        "QPushButton:hover { background-color: #e74c3c; }"
    )
    _UNLOCKED_STYLE = (
        "QPushButton {"
        "  background-color: #27ae60; color: white;"
        "  border-radius: 10px; padding: 4px 16px;"
        "  font-weight: bold; font-size: 12px;"
        "}"
        "QPushButton:hover { background-color: #2ecc71; }"
    )

    def __init__(self, parent=None):
        super().__init__("Locked", parent)
        self._locked = True
        self.setCheckable(True)
        self.setFixedWidth(100)
        self.setStyleSheet(self._LOCKED_STYLE)
        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked: bool) -> None:
        self._locked = not checked
        if self._locked:
            self.setText("Locked")
            self.setStyleSheet(self._LOCKED_STYLE)
        else:
            self.setText("Unlocked")
            self.setStyleSheet(self._UNLOCKED_STYLE)

    @property
    def is_locked(self) -> bool:
        return self._locked


class SettingsDialog(QDialog):
    """
    Settings dialog with a lock/unlock toggle and Apply / Save confirmation.

    Lock categories are hardcoded at the top of this file:
      ALWAYS_EDITABLE – always editable
      UNLOCKABLE      – editable only when unlocked
      ALWAYS_LOCKED   – never editable (display only)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings — Paths")
        self.setMinimumWidth(420)
        self._build_ui()
        self._prepopulate()
        self._apply_lock()   # enforce initial (locked) state
        self.adjustSize()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Lock toggle row
        lock_row = QHBoxLayout()
        lock_row.addWidget(QLabel("Edit lock:"))
        self.lock_btn = _ToggleLock(self)
        self.lock_btn.toggled.connect(lambda _: self._apply_lock())
        lock_row.addWidget(self.lock_btn)
        lock_row.addStretch()

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)

        # Form fields
        self.edit_output = QLineEdit(self)
        self.btn_browse_output = QPushButton("Browse…", self)
        self.edit_log = QLineEdit(self)
        self.btn_browse_log = QPushButton("Browse…", self)
        self.edit_toml = QLineEdit(self)
        self.btn_browse_toml = QPushButton("Browse…", self)

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

        # Apply / Cancel buttons
        self.btn_apply = QPushButton("Apply / Save", self)
        self.btn_apply.setDefault(True)
        self.btn_cancel = QPushButton("Cancel", self)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_apply)

        root = QVBoxLayout()
        root.addLayout(lock_row)
        root.addWidget(sep)
        root.addLayout(form)
        root.addLayout(btn_row)
        self.setLayout(root)

        # Connections
        self.btn_browse_output.clicked.connect(
            lambda: self._pick_dir(self.edit_output, "Select OUTPUT_PATH"))
        self.btn_browse_log.clicked.connect(
            lambda: self._pick_dir(self.edit_log, "Select LOG_PATH"))
        self.btn_browse_toml.clicked.connect(self._pick_toml)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel.clicked.connect(self.reject)

    # ── Prepopulate from current settings ────────────────────────────────────

    def _prepopulate(self):
        def _get(attr):
            mod_val = getattr(settings, attr, None)
            obj_val = getattr(getattr(settings, 'SETTINGS', None), attr, None)
            val = mod_val or obj_val or ''
            if isinstance(val, str) and val:
                try:
                    val = str(Path(val).expanduser())
                except Exception:
                    pass
            return str(val or '')

        try:
            self.edit_output.setText(_get('OUTPUT_PATH'))
        except Exception:
            self.edit_output.setText('')
        try:
            self.edit_log.setText(_get('LOG_PATH'))
        except Exception:
            self.edit_log.setText('')
        try:
            self.edit_toml.setText(str(getattr(settings, 'TOML_FILE', '') or ''))
        except Exception:
            self.edit_toml.setText('')

    # ── Lock / unlock logic ───────────────────────────────────────────────────

    def _field_widgets(self, field_name):
        """Return (QLineEdit, QPushButton) for the given field key."""
        return {
            'output_path': (self.edit_output, self.btn_browse_output),
            'log_path':    (self.edit_log,    self.btn_browse_log),
            'toml_config': (self.edit_toml,   self.btn_browse_toml),
        }.get(field_name, ())

    def _set_field_enabled(self, field_name: str, enabled: bool):
        for widget in self._field_widgets(field_name):
            widget.setEnabled(enabled)

    def _apply_lock(self):
        locked = self.lock_btn.is_locked
        for field in ALWAYS_EDITABLE:
            self._set_field_enabled(field, True)
        for field in UNLOCKABLE:
            self._set_field_enabled(field, not locked)
        for field in ALWAYS_LOCKED:
            self._set_field_enabled(field, False)

    # ── File pickers ──────────────────────────────────────────────────────────

    def _pick_dir(self, target_edit: QLineEdit, caption: str):
        start = target_edit.text().strip()
        path = QFileDialog.getExistingDirectory(self, caption, start or "")
        if path:
            target_edit.setText(path)

    def _pick_toml(self):
        start = self.edit_toml.text().strip()
        start_dir = str(Path(start).parent) if start else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TOML Config", start_dir,
            "TOML Files (*.toml);;All Files (*)")
        if path:
            self.edit_toml.setText(path)

    # ── Apply / Save ──────────────────────────────────────────────────────────

    def _on_apply(self):
        reply = QMessageBox.question(
            self,
            "Save Changes",
            "Are you sure you want to save these changes?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._save()

    def _save(self):
        # Saving is not yet implemented — closes the dialog only.
        self.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dlg = SettingsDialog()
    dlg.show()
    sys.exit(app.exec_())
