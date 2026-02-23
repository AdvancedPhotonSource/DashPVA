import sys
import os, subprocess
from collections import OrderedDict
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from .registry import VIEWS


class LauncherDialog(QDialog):
    def __init__(self):
        super(LauncherDialog, self).__init__()
        uic.loadUi('gui/dashpva.ui', self)
        self.processes = {}
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_processes)
        self._timer.start()

        # Build all sections and buttons dynamically from the registry
        self._insert_registry_sections()

        if hasattr(self, 'btn_settings'):
            self.btn_settings.clicked.connect(
                lambda: self.launch(
                    'settings',
                    [sys.executable, 'viewer/settings/settings_dialog.py'],
                    self.btn_settings,
                    'Settings — Running…'
                )
            )
        if hasattr(self, 'btn_exit'):
            self.btn_exit.clicked.connect(self.request_close)
        if hasattr(self, 'btn_shutdown_all'):
            self.btn_shutdown_all.clicked.connect(self._confirm_shutdown_all)

        self._update_status()

        # Static tip in the info footer, if present
        try:
            if hasattr(self, 'lbl_info') and self.lbl_info is not None:
                self.lbl_info.setText("Note: On first time use loading may take a while")
        except Exception:
            pass

    def _insert_registry_sections(self):
        """Build all section headers and buttons from the VIEWS registry.

        Entries are grouped by their 'section' key.  Sections appear in the
        order they are first encountered in the registry list.  Each section
        gets a bold header label followed by its buttons, inserted just above
        the status label / bottom bar.
        """
        layout = self.layout()
        if layout is None:
            return

        # Determine insertion point — just above the status label or bottom bar
        insert_at = -1
        target_status = getattr(self, 'lbl_status', None)
        if target_status is not None:
            idx = layout.indexOf(target_status)
            if idx >= 0:
                insert_at = idx
        if insert_at < 0:
            target_bar = getattr(self, 'horizontalLayout', None)
            if target_bar is not None:
                idx = layout.indexOf(target_bar)
                if idx >= 0:
                    insert_at = idx
        if insert_at < 0:
            insert_at = layout.count()

        # Group entries by section, preserving first-seen order
        sections: OrderedDict[str, list] = OrderedDict()
        for entry in VIEWS:
            section = entry.get('section', 'Other')
            sections.setdefault(section, []).append(entry)

        # Render each section
        for section_name, entries in sections.items():
            # Section header
            header = QLabel(section_name, self)
            header.setStyleSheet("font-weight: bold; color: #34495e; font-size: 12px;")
            header.setAlignment(Qt.AlignCenter)
            layout.insertWidget(insert_at, header)
            insert_at += 1

            # Buttons
            for entry in entries:
                try:
                    label = entry.get('label', entry.get('key', ''))
                    tooltip = entry.get('tooltip', '')
                    btn = QPushButton(label, self)
                    if tooltip:
                        btn.setToolTip(tooltip)
                    layout.insertWidget(insert_at, btn)
                    insert_at += 1
                    btn.clicked.connect(
                        lambda _=False, e=entry, b=btn: self.launch(
                            e['key'], e['cmd'], b, e['running_text']
                        )
                    )
                except Exception:
                    pass

    def launch(self, key, cmd, button, running_text, quiet=False):
        """Start a child process and update UI indicators."""
        if key in self.processes and self.processes[key]['popen'].poll() is None:
            # Already running
            return
        original_text = button.text()
        button.setEnabled(False)
        button.setText(f"{original_text} — Launching…")

        kwargs = {}
        if quiet:
            kwargs['stdout'] = subprocess.DEVNULL
            kwargs['stderr'] = subprocess.DEVNULL

        try:
            p = subprocess.Popen(cmd, **kwargs)
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
                # Process ended
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
            self.lbl_status.setText('No modules running' if count == 0 else f'{count} module(s) running')
        if hasattr(self, 'btn_exit'):
            self.btn_exit.setEnabled(True)
        if hasattr(self, 'btn_shutdown_all'):
            self.btn_shutdown_all.setEnabled(count > 0)

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
        """Attempt graceful terminate, then force kill if still alive."""
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=timeout)
                except Exception:
                    pass
            if p.poll() is None:
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

    def _confirm_shutdown_all(self):
        """Confirm and force-stop all running modules."""
        count = len(self.processes)
        if count == 0:
            return
        text = f"{self._format_running_modules_list()}\n\nAre you sure you want to force stop all running modules?\n\nData might be lost."
        resp = QMessageBox.question(
            self,
            'Shutdown All Modules',
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if resp == QMessageBox.Yes:
            self.shutdown_all()

    def request_close(self):
        """Prompt to force stop modules before exiting if any are running."""
        try:
            any_running = any(entry['popen'].poll() is None for entry in self.processes.values())
        except Exception:
            any_running = False
        if any_running:
            text = f"{self._format_running_modules_list()}\n\nForce stop all and exit?\n\nData might be lost."
            resp = QMessageBox.question(
                self,
                'Exit Launcher',
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if resp == QMessageBox.Yes:
                self.shutdown_all()
                self.close()
        else:
            self.close()

    def closeEvent(self, event):
        """On close, prompt to force-stop modules if any are running."""
        try:
            any_running = any(entry['popen'].poll() is None for entry in self.processes.values())
        except Exception:
            any_running = False
        if any_running:
            text = f"{self._format_running_modules_list()}\n\nForce stop all and exit?\n\nData might be lost."
            resp = QMessageBox.question(
                self,
                'Exit Launcher',
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if resp == QMessageBox.Yes:
                self.shutdown_all()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    dlg = LauncherDialog()
    dlg.show()
    app.exec_()


if __name__ == '__main__':
    main()
