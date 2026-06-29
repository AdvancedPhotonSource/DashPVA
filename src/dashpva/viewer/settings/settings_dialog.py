import sys
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

import dashpva.settings as settings
from dashpva.gui import configure_app
from dashpva.gui.theme_colors import (
    ERROR,
    ERROR_HOVER,
    FONT_BODY,
    SUCCESS,
    SUCCESS_HOVER,
)

# ── Lock categories ────────────────────────────────────────────────────────────
ALWAYS_EDITABLE = ['output_path', 'log_path']
UNLOCKABLE      = ['toml_config']
ALWAYS_LOCKED   = []


class _ToggleLock(QPushButton):
    """Styled toggle button that switches between Locked / Unlocked states."""

    _LOCKED_STYLE = (
        "QPushButton {"
        f"  background-color: {ERROR_HOVER}; color: white;"
        f"  border-radius: 10px; padding: 4px 16px;"
        f"  font-weight: bold; font-size: {FONT_BODY};"
        "}"
        f"QPushButton:hover {{ background-color: {ERROR}; }}"
    )
    _UNLOCKED_STYLE = (
        "QPushButton {"
        f"  background-color: {SUCCESS}; color: white;"
        f"  border-radius: 10px; padding: 4px 16px;"
        f"  font-weight: bold; font-size: {FONT_BODY};"
        "}"
        f"QPushButton:hover {{ background-color: {SUCCESS_HOVER}; }}"
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
        self._apply_lock()
        # Record baseline after prepopulate; connect change tracking after that
        self._initial = self._current_values()
        for edit in (self.edit_output, self.edit_log, self.edit_toml):
            edit.textChanged.connect(self._update_action_state)
        self._update_action_state()
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

        # Re-seed row (separate from the action bar)
        self.btn_reseed = QPushButton("Re-seed", self)
        self.btn_reseed.setToolTip("Re-seed missing default values for the currently selected profile")
        reseed_row = QHBoxLayout()
        reseed_row.addWidget(self.btn_reseed)
        reseed_row.addStretch()

        # Action bar — Clear and Apply / Save only
        self.btn_clear = QPushButton("Clear", self)
        self.btn_clear.setObjectName("btn_settings_clear")
        self.btn_clear.setEnabled(False)
        self.btn_apply = QPushButton("Apply / Save", self)
        self.btn_apply.setObjectName("btn_settings_apply")
        self.btn_apply.setEnabled(False)

        action_bar = QFrame(self)
        action_bar.setObjectName("action_bar")
        bar_layout = QHBoxLayout(action_bar)
        bar_layout.setContentsMargins(8, 6, 8, 6)
        bar_layout.addStretch()
        bar_layout.addWidget(self.btn_clear)
        bar_layout.addWidget(self.btn_apply)

        root = QVBoxLayout()
        root.addLayout(lock_row)
        root.addWidget(sep)
        root.addLayout(form)
        root.addLayout(reseed_row)
        root.addWidget(action_bar)
        self.setLayout(root)

        # Connections
        self.btn_browse_output.clicked.connect(
            lambda: self._pick_dir(self.edit_output, "Select OUTPUT_PATH"))
        self.btn_browse_log.clicked.connect(
            lambda: self._pick_dir(self.edit_log, "Select LOG_PATH"))
        self.btn_browse_toml.clicked.connect(self._pick_toml)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_reseed.clicked.connect(self._on_reseed)

    # ── Change tracking ───────────────────────────────────────────────────────

    def _current_values(self) -> dict:
        return {
            'output': self.edit_output.text(),
            'log':    self.edit_log.text(),
            'toml':   self.edit_toml.text(),
        }

    def _update_action_state(self):
        if not hasattr(self, '_initial'):
            return
        changed = self._current_values() != self._initial
        self.btn_apply.setEnabled(changed)
        self.btn_clear.setEnabled(changed)

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

    # ── Clear (reset to saved) ────────────────────────────────────────────────

    def _on_clear(self):
        self.edit_output.setText(self._initial['output'])
        self.edit_log.setText(self._initial['log'])
        self.edit_toml.setText(self._initial['toml'])

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
        output = self.edit_output.text().strip()
        log = self.edit_log.text().strip()
        toml = self.edit_toml.text().strip()

        if output:
            settings.OUTPUT_PATH = output
        if log:
            settings.LOG_PATH = log
        settings.TOML_FILE = toml or None

        if toml:
            try:
                settings.set_locator(toml)
                settings.reload()
            except Exception as e:
                QMessageBox.warning(self, "Config Reload", f"Settings saved but TOML reload failed:\n{e}")

        QMessageBox.information(self, "Settings", "Settings saved for this session.")
        self.accept()

    # ── Re-seed ───────────────────────────────────────────────────────────────

    def _on_reseed(self):
        """Add any missing default settings — never overwrites existing values."""
        try:
            from dashpva.scripts.seed_settings_defaults_sql import seed_defaults
            seed_defaults()
            QMessageBox.information(self, "Re-seed", "Missing defaults added. No existing values were changed.")
        except Exception as e:
            QMessageBox.critical(self, "Re-seed", f"Re-seeding failed:\n{e}")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    configure_app(app)
    dlg = SettingsDialog()
    dlg.show()
    sys.exit(app.exec_())
