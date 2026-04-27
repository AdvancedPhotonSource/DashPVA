import subprocess

import settings
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog


class ReleaseCheckWorker(QThread):
    result = pyqtSignal(bool, str, str)  # has_update, tag_name, release_notes
    error = pyqtSignal(str)

    def run(self):
        try:
            import requests
            resp = requests.get(
                'https://api.github.com/repos/AdvancedPhotonSource/DashPVA/releases/latest',
                timeout=10,
                headers={'Accept': 'application/vnd.github+json'},
            )
            if resp.status_code == 404:
                self.error.emit('No release found')
                return
            resp.raise_for_status()
            data = resp.json()
            tag = data.get('tag_name', '')
            notes = data.get('body', '') or ''
            has_update = tag.lstrip('v') != str(settings.__VERSION__)
            self.result.emit(has_update, tag, notes)
        except Exception as exc:
            self.error.emit(str(exc))


class PullWorker(QThread):
    line = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def run(self):
        try:
            proc = subprocess.Popen(
                ['git', 'pull', 'origin', 'main'],
                cwd=str(settings.PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for ln in proc.stdout:
                self.line.emit(ln.rstrip())
            proc.wait()
            self.finished.emit(proc.returncode == 0)
        except Exception as exc:
            self.line.emit(f'ERROR: {exc}')
            self.finished.emit(False)


class UpdateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('gui/install/update_dialog.ui', self)

        self._check_worker = None
        self._pull_worker = None

        self.btn_update.clicked.connect(self._start_pull)
        self.btn_close.clicked.connect(self.accept)

        self._start_check()

    def _start_check(self):
        self._check_worker = ReleaseCheckWorker()
        self._check_worker.result.connect(self._on_check_result)
        self._check_worker.error.connect(self._on_check_error)
        self._check_worker.start()

    def _on_check_result(self, has_update, tag, notes):
        if has_update:
            self.lbl_status.setText(f'Update available: {tag}')
            self.lbl_status.setStyleSheet('font-size: 13px; color: #E67E22; font-weight: 600;')
            if notes.strip():
                self.txt_notes.setPlainText(notes)
                self.txt_notes.setVisible(True)
            self.btn_update.setVisible(True)
        else:
            self.lbl_status.setText(f'✓ v{settings.__VERSION__} is the latest')
            self.lbl_status.setStyleSheet('font-size: 13px; color: #27AE60; font-weight: 600;')

    def _on_check_error(self, msg):
        self.lbl_status.setText(f'Could not check for updates: {msg}')
        self.lbl_status.setStyleSheet('font-size: 13px; color: #7A8394;')

    def _start_pull(self):
        self.btn_update.setEnabled(False)
        self.lbl_status.setText('Pulling update…')
        self.lbl_status.setStyleSheet('font-size: 13px; color: #4B6EF5;')
        self.txt_notes.clear()
        self.txt_notes.setVisible(True)
        self._pull_worker = PullWorker()
        self._pull_worker.line.connect(self.txt_notes.appendPlainText)
        self._pull_worker.finished.connect(self._on_pull_finished)
        self._pull_worker.start()

    def _on_pull_finished(self, success):
        if success:
            self.lbl_status.setText('Update complete — restart DashPVA to apply')
            self.lbl_status.setStyleSheet('font-size: 13px; color: #27AE60; font-weight: 600;')
        else:
            self.lbl_status.setText('Update failed — see output above')
            self.lbl_status.setStyleSheet('font-size: 13px; color: #E74C3C; font-weight: 600;')
