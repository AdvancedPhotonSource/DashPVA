import sys
import os, subprocess, sys
from pathlib import Path
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from viewer.views_registry.registry import VIEWS


class LauncherDialog(QDialog):
    def __init__(self):
        super(LauncherDialog, self).__init__()
        uic.loadUi('gui/dashpva.ui', self)
        self.processes = {}
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_processes)
        self._timer.start()

        # Wire buttons to launchers
        if hasattr(self, 'btn_hkl3d_viewer'):
            self.btn_hkl3d_viewer.clicked.connect(
                lambda: self.launch(
                    'hkl3d_viewer',
                    [sys.executable, 'viewer/hkl_3d_viewer.py'],
                    self.btn_hkl3d_viewer,
                    'HKL 3D Viewer — Running…'
                )
            )
        if hasattr(self, 'btn_hkl3d_slicer'):
            self.btn_hkl3d_slicer.clicked.connect(
                lambda: self.launch(
                    'hkl3d_slicer',
                    [sys.executable, 'viewer/hkl_3d_slice_window.py'],
                    self.btn_hkl3d_slicer,
                    'HKL 3D Slicer — Running…'
                )
            )
        if hasattr(self, 'btn_area_detector'):
            self.btn_area_detector.clicked.connect(
                lambda: self.launch(
                    'area_detector',
                    [sys.executable, 'viewer/area_det_viewer.py'],
                    self.btn_area_detector,
                    'Area Detector Viewer — Running…'
                )
            )
        if hasattr(self, 'btn_pva_setup'):
            self.btn_pva_setup.clicked.connect(
                lambda: self.launch(
                    'pva_setup',
                    [sys.executable, 'pva_setup/pva_workflow_setup_dialog.py'],
                    self.btn_pva_setup,
                    'PVA Workflow Setup — Running…'
                )
            )
        if hasattr(self, 'btn_sim_setup'):
            self.btn_sim_setup.clicked.connect(
                lambda: self.launch(
                    'sim_setup',
                    [sys.executable, 'consumers/sim_rsm_data.py'],
                    self.btn_sim_setup,
                    'caIOC(Name) — Running…',
                    quiet=True
                )
            )
        if hasattr(self, 'btn_workbench'):
            self.btn_workbench.clicked.connect(
                lambda: self.launch(
                    'workbench',
                    [sys.executable, 'viewer/workbench/workbench.py'],
                    self.btn_workbench,
                    'Workbench — Running…'
                )
            )
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
            # Exit remains enabled; closing will prompt if processes are running
            self.btn_exit.setEnabled(True)
        if hasattr(self, 'btn_shutdown_all'):
            # Enable Shutdown All only when there are running modules
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
