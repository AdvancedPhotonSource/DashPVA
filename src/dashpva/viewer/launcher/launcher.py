import os
import signal
import subprocess
import sys
from collections import OrderedDict

from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from dashpva.gui import configure_app, ui_path
from dashpva.gui.theme_colors import (
    BORDER,
    FONT_SMALL,
    SUCCESS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WARNING,
    status_style,
)

from .registry import get_views


class LauncherDialog(QDialog):
    def __init__(self):
        super(LauncherDialog, self).__init__()
        uic.loadUi(ui_path("launcher", "launcher.ui"), self)

        try:
            import dashpva.settings as _settings
            beamline = _settings.BEAMLINE_NAME
            if beamline:
                self.lbl_header.setText(f'DashPVA: {beamline}')
                self.setWindowTitle(f'DashPVA Launcher — {beamline}')
            version = _settings.__VERSION__
            if hasattr(self, 'lbl_subtitle'):
                self.lbl_subtitle.setText(f'v{version}  ·  Select a module to launch')
        except Exception:
            pass

        self.processes = {}
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_processes)
        self._timer.start()

        self._insert_registry_sections()
        self._setup_update_check()

        if hasattr(self, 'btn_settings'):
            self.btn_settings.clicked.connect(self._open_settings)
        if hasattr(self, 'btn_exit'):
            self.btn_exit.clicked.connect(self.request_close)
        if hasattr(self, 'btn_shutdown_all'):
            self.btn_shutdown_all.clicked.connect(self._confirm_shutdown_all)

        self._update_status()

        try:
            if hasattr(self, 'lbl_info') and self.lbl_info is not None:
                self.lbl_info.setText("Note: On first time use loading may take a while")
        except Exception:
            pass

        # Wire up the logs link defined in launcher.ui
        if hasattr(self, 'lbl_logs'):
            self.lbl_logs.setCursor(QCursor(Qt.PointingHandCursor))
            self.lbl_logs.linkActivated.connect(self._open_logs)

        self.adjustSize()

    def _setup_update_check(self):
        """Insert an update-status row above lbl_status and start a background release check."""
        from .update_dialog import ReleaseCheckWorker, UpdateDialog

        layout = self.layout()
        if layout is None:
            return

        target = getattr(self, 'lbl_status', None)
        insert_at = layout.indexOf(target) if target is not None else layout.count()
        if insert_at < 0:
            insert_at = layout.count()

        row_widget = QWidget(self)
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._lbl_update_status = QLabel('Checking for updates…', row_widget)
        self._lbl_update_status.setStyleSheet(status_style(TEXT_MUTED))
        row.addWidget(self._lbl_update_status)
        row.addStretch()

        btn_updates = QPushButton('Updates', row_widget)
        btn_updates.setObjectName('btn_settings')  # reuse the same subtle style
        btn_updates.clicked.connect(lambda: UpdateDialog(self).exec_())
        row.addWidget(btn_updates)

        layout.insertWidget(insert_at, row_widget)

        self._release_worker = ReleaseCheckWorker()
        self._release_worker.result.connect(self._on_release_check)
        self._release_worker.error.connect(
            lambda _: self._lbl_update_status.setText('Could not check for updates')
        )
        self._release_worker.start()

    def _on_release_check(self, has_update, tag, _notes):
        if has_update:
            self._lbl_update_status.setText(f'● Update available: {tag}')
            self._lbl_update_status.setStyleSheet(status_style(WARNING, bold=True))
        else:
            self._lbl_update_status.setText('Up to date')
            self._lbl_update_status.setStyleSheet(status_style(SUCCESS))

    def _insert_registry_sections(self):
        """Build section dividers and button grids from the VIEWS registry."""
        layout = self.layout()
        if layout is None:
            return

        # Insert above status label
        insert_at = -1
        target = getattr(self, 'lbl_status', None)
        if target is not None:
            idx = layout.indexOf(target)
            if idx >= 0:
                insert_at = idx
        if insert_at < 0:
            target = getattr(self, 'horizontalLayout', None)
            if target is not None:
                idx = layout.indexOf(target)
                if idx >= 0:
                    insert_at = idx
        if insert_at < 0:
            insert_at = layout.count()

        sections: OrderedDict[str, list] = OrderedDict()
        for entry in get_views():
            sections.setdefault(entry.get('section', 'Other'), []).append(entry)

        for section_name, entries in sections.items():
            # Divider row: SECTION NAME ————
            divider = QWidget(self)
            divider_row = QHBoxLayout(divider)
            divider_row.setContentsMargins(0, 8, 0, 2)
            divider_row.setSpacing(6)

            sec_lbl = QLabel(section_name.upper(), divider)
            sec_lbl.setStyleSheet(
                f"font-size: {FONT_SMALL}; font-weight: 600; color: {TEXT_SECONDARY}; letter-spacing: 0.5px;"
            )
            line = QFrame(divider)
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Plain)
            line.setStyleSheet(f"background-color: {BORDER}; max-height: 1px; border: none;")

            divider_row.addWidget(sec_lbl)
            divider_row.addWidget(line, 1)

            layout.insertWidget(insert_at, divider)
            insert_at += 1

            # 2-column grid of buttons
            grid_widget = QWidget(self)
            grid = QGridLayout(grid_widget)
            grid.setSpacing(6)
            grid.setContentsMargins(0, 0, 0, 2)

            for i, entry in enumerate(entries):
                try:
                    label = entry.get('label', entry.get('key', ''))
                    tooltip = entry.get('tooltip', '')
                    btn = QPushButton(label, grid_widget)
                    if tooltip:
                        btn.setToolTip(tooltip)
                    row, col = divmod(i, 2)
                    grid.addWidget(btn, row, col)
                    btn.clicked.connect(
                        lambda _=False, e=entry, b=btn: self.launch(
                            e['key'], e['cmd'], b, e['running_text']
                        )
                    )
                except Exception:
                    pass

            layout.insertWidget(insert_at, grid_widget)
            insert_at += 1

    def launch(self, key, cmd, button, running_text, quiet=False):
        """Start a child process and update UI indicators."""
        if key in self.processes and self.processes[key]['popen'].poll() is None:
            return
        original_text = button.text()
        button.setEnabled(False)
        button.setText(f"{original_text} — Launching…")

        kwargs = {}
        if quiet:
            kwargs['stdout'] = subprocess.DEVNULL
            kwargs['stderr'] = subprocess.DEVNULL

        try:
            p = subprocess.Popen(cmd, start_new_session=True, **kwargs)
            button.setText(running_text)
            self.processes[key] = {
                'popen': p,
                'button': button,
                'original_text': original_text,
                'running_text': running_text
            }
        except Exception as e:
            QMessageBox.critical(
                self,
                'Launch Failed',
                f'Failed to launch:\n{" ".join(cmd)}\n\n{e}'
            )
            button.setText(original_text)
            button.setEnabled(True)

        self._update_status()

    def _poll_processes(self):
        """Periodic check for finished processes to restore UI state."""
        finished = []
        for key, entry in self.processes.items():
            p = entry['popen']
            if p.poll() is not None:
                entry['button'].setText(entry['original_text'])
                entry['button'].setEnabled(True)
                finished.append(key)
        for key in finished:
            self.processes.pop(key, None)
        self._update_status()

    def _update_status(self):
        """Update status label and button states."""
        count = len(self.processes)
        if hasattr(self, 'lbl_status'):
            if count == 0:
                self.lbl_status.setStyleSheet(status_style(TEXT_SECONDARY))
                self.lbl_status.setText('No modules running')
            else:
                self.lbl_status.setStyleSheet(status_style(SUCCESS, bold=True))
                noun = 'module' if count == 1 else 'modules'
                self.lbl_status.setText(f'● {count} {noun} running')
        if hasattr(self, 'btn_exit'):
            self.btn_exit.setEnabled(True)
        if hasattr(self, 'btn_shutdown_all'):
            self.btn_shutdown_all.setEnabled(count > 0)

    def _open_logs(self, _=None):
        """Open the log viewer dialog."""
        try:
            from .log_viewer_dialog import LogViewerDialog
            dlg = LogViewerDialog(self)
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, 'Logs', f'Failed to open log viewer:\n{e}')

    def _open_settings(self):
        """Open the Settings dialog modally and prefilled from current global settings."""
        try:
            from dashpva.viewer.settings.settings_dialog import (
                SettingsDialog as _SettingsDialog,
            )
            dlg = _SettingsDialog(self)
            dlg.exec_()
        except Exception as e:
            try:
                if hasattr(self, 'btn_settings'):
                    self.launch(
                        'settings',
                        [sys.executable, 'viewer/settings/settings_dialog.py'],
                        self.btn_settings,
                        'Settings — Running…'
                    )
                else:
                    raise e
            except Exception:
                QMessageBox.critical(self, 'Settings', f'Failed to open Settings dialog:\n{e}')

    def _format_running_modules_list(self):
        """Return a human-readable list of running modules and their PIDs."""
        lines = []
        for key, entry in self.processes.items():
            p = entry.get('popen')
            if p is None or p.poll() is not None:
                continue
            name = entry.get('running_text', key)
            if ' — ' in name:
                name = name.split(' — ')[0]
            try:
                pid = p.pid
            except Exception:
                pid = 'unknown'
            lines.append(f"- {name} (PID {pid})")
        if not lines:
            return "Running modules:\nNone"
        return "Running modules:\n" + "\n".join(lines)

    def _terminate_proc(self, p, timeout=3.0):
        """Terminate the process group, then force kill if still alive."""
        try:
            if p.poll() is not None:
                return
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                p.wait(timeout=2)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def shutdown_all(self):
        """Force-stop all running modules and restore UI state."""
        for key, entry in list(self.processes.items()):
            self._terminate_proc(entry['popen'])
            entry['button'].setText(entry['original_text'])
            entry['button'].setEnabled(True)
            self.processes.pop(key, None)
        self._update_status()

    @staticmethod
    def _widen_messagebox(msg, width=420):
        layout = msg.layout()
        if layout is not None:
            from PyQt5.QtWidgets import QSizePolicy, QSpacerItem
            spacer = QSpacerItem(width, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
            layout.addItem(spacer, layout.rowCount(), 0, 1, layout.columnCount())

    def _confirm_shutdown_all(self):
        """Confirm and force-stop all running modules."""
        if not self.processes:
            return
        running_list = self._format_running_modules_list()
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle('Shutdown All Modules')
        msg.setText('Force stop all running modules?')
        msg.setInformativeText(running_list + '\n\nData might be lost.')
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        self._widen_messagebox(msg)
        if msg.exec_() == QMessageBox.Yes:
            self.shutdown_all()

    def _confirm_exit(self):
        """Show exit confirmation dialog. Returns True if the user confirmed."""
        if not self.processes:
            return True
        running_list = self._format_running_modules_list()
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle('Exit Launcher')
        msg.setText('Shutdown all running processes and exit?')
        msg.setInformativeText(running_list + '\n\nData might be lost.')
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        self._widen_messagebox(msg)
        return msg.exec_() == QMessageBox.Yes

    def request_close(self):
        self.close()

    def closeEvent(self, event):
        if self._confirm_exit():
            self.shutdown_all()
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    configure_app(app)
    try:
        from dashpva.utils import SizeManager
        _size_manager = SizeManager(app)  # noqa: F841 — kept alive on stack during exec_
    except Exception:
        pass
    dlg = LauncherDialog()
    dlg.show()
    app.exec_()


if __name__ == '__main__':
    main()
