import subprocess
import sys

from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox

import dashpva.settings as settings
from dashpva.gui import ui_path
from dashpva.gui.theme_colors import (
    ERROR,
    FONT_BODY,
    FONT_HEADING,
    FONT_LARGE,
    FONT_SUBHEADING,
    INFO,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    status_style,
)


def _block_html(heading, body, command=None):
    """Build the update-blocked warning using the theme's colours and font sizes."""
    html = (
        f'<div style="{status_style(ERROR, bold=True, size=FONT_HEADING)}">{heading}</div>'
        f'<div style="{status_style(TEXT_PRIMARY, size=FONT_SUBHEADING)} margin-top:10px;">{body}</div>'
    )
    if command:
        html += (
            f'<div style="{status_style(INFO, size=FONT_SUBHEADING)} '
            f'font-family:monospace; margin-top:8px;">{command}</div>'
        )
    return html


def _parse_version(version):
    """Parse '1.0.3' into (1, 0, 3), ignoring non-numeric parts."""
    return tuple(int(p) for p in version.split('.') if p.isdigit())


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
            has_update = _parse_version(tag.lstrip('v')) > _parse_version(str(settings.__VERSION__))
            self.result.emit(has_update, tag, notes)
        except Exception as exc:
            self.error.emit(str(exc))


class PullWorker(QThread):
    line = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, tag, force=False, parent=None):
        super().__init__(parent)
        self.tag = tag
        self.force = force

    def run(self):
        # Fetch the release tag and check it out so both code and version match
        # the release, instead of pulling the tip of main. force=True discards a
        # feature branch / uncommitted changes when the user opted to override.
        # install.sh --update then re-syncs the env so the installed package
        # metadata (and the reported version) match the checked-out release; a
        # bare checkout updates the source but leaves the old installed version.
        checkout = ['git', 'checkout', self.tag]
        if self.force:
            checkout.insert(2, '-f')
        commands = [
            ['git', 'fetch', 'origin', 'tag', self.tag],
            checkout,
            ['bash', 'install.sh', '--update'],
        ]
        try:
            for cmd in commands:
                self.line.emit(f'$ {" ".join(cmd)}')
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(settings.PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for ln in proc.stdout:
                    self.line.emit(ln.rstrip())
                proc.wait()
                if proc.returncode != 0:
                    self.finished.emit(False)
                    return
            self.finished.emit(True)
        except Exception as exc:
            self.line.emit(f'ERROR: {exc}')
            self.finished.emit(False)


class UpdateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(ui_path("install", "update_dialog.ui"), self)
        self.btn_update.setProperty("role", "info")
        self.lbl_notes_header.setStyleSheet(status_style(TEXT_SECONDARY, bold=True))

        self._check_worker = None
        self._pull_worker = None
        self._latest_tag = None

        self.btn_update.clicked.connect(self._start_pull)
        self.btn_close.clicked.connect(self.accept)

        self._start_check()

    def _start_check(self):
        self._check_worker = ReleaseCheckWorker()
        self._check_worker.result.connect(self._on_check_result)
        self._check_worker.error.connect(self._on_check_error)
        self._check_worker.start()

    def _on_check_result(self, has_update, tag, notes):
        self._latest_tag = tag
        if has_update:
            self.lbl_status.setText(f'Update available:   v{settings.__VERSION__}   →   {tag}')
            self.lbl_status.setStyleSheet(status_style(WARNING, bold=True, size=FONT_LARGE))
            self.btn_update.setVisible(True)
        else:
            self.lbl_status.setText(f'v{settings.__VERSION__}   ✓   Up to date')
            self.lbl_status.setStyleSheet(status_style(SUCCESS, bold=True, size=FONT_LARGE))
        self._show_notes(notes)

    def _show_notes(self, notes):
        """Reveal the Release Notes section (divider + header + text), or keep it
        hidden when there are no notes for the latest release."""
        text = (notes or '').strip()
        self.lbl_notes_header.setText('Release Notes')
        self.txt_notes.setMarkdown(text)
        for w in (self.line_notes, self.lbl_notes_header, self.txt_notes):
            w.setVisible(bool(text))

    def _on_check_error(self, msg):
        self.lbl_status.setText(f'Could not check for updates: {msg}')
        self.lbl_status.setStyleSheet(status_style(TEXT_SECONDARY, size=FONT_BODY))

    def _preflight_block_reason(self):
        """Return (html_warning, overridable) if updating in place would overwrite
        the user's work, else None. A feature branch or uncommitted changes are
        overridable — the caller offers Yes/No to force through; a git error we
        cannot interpret is a hard stop (overridable=False)."""
        try:
            root = str(settings.PROJECT_ROOT)
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=root, text=True, stderr=subprocess.STDOUT,
            ).strip()
            dirty = bool(subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=root, text=True, stderr=subprocess.STDOUT,
            ).strip())
        except Exception as exc:
            return _block_html('COULD NOT VERIFY GIT STATE',
                               f'The update was stopped to be safe.<br>{exc}'), False
        override_cmds = (f'git fetch origin tag {self._latest_tag}<br>'
                         f'git checkout -f {self._latest_tag}')
        if branch not in ('main', 'HEAD'):
            return _block_html(
                '⚠  YOU ARE ON A DIFFERENT BRANCH',
                f"You are on {branch}, not main. Continuing checks out the release "
                f"tag, moving you off your branch and overwriting any uncommitted "
                f"changes.<br><br>These commands will be run in the terminal:",
                override_cmds,
            ), True
        if dirty:
            return _block_html(
                '⚠  YOU HAVE CHANGES THAT CAN BE OVERWRITTEN',
                "Uncommitted changes in the repository will be overwritten by the "
                "update.<br><br>These commands will be run in the terminal:",
                override_cmds,
            ), True
        return None

    def _start_pull(self):
        if not self._latest_tag:
            return
        reason = self._preflight_block_reason()
        force = False
        if reason is not None:
            html, overridable = reason
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setTextFormat(Qt.RichText)
            msg.setText(html)
            if not overridable:
                msg.setWindowTitle('Update blocked')
                msg.exec_()
                return
            msg.setWindowTitle('Overwrite changes and update?')
            override_btn = msg.addButton('Override', QMessageBox.AcceptRole)
            override_btn.setProperty('role', 'error')
            cancel_btn = msg.addButton('Cancel', QMessageBox.RejectRole)
            msg.setDefaultButton(cancel_btn)
            msg.exec_()
            if msg.clickedButton() is not override_btn:
                return
            force = True
        self.btn_update.setEnabled(False)
        self.lbl_status.setText(f'Updating to {self._latest_tag}…')
        self.lbl_status.setStyleSheet(status_style(INFO, size=FONT_BODY))
        self.lbl_notes_header.setText('Update log')
        self.txt_notes.clear()
        for w in (self.line_notes, self.lbl_notes_header, self.txt_notes):
            w.setVisible(True)
        self._pull_worker = PullWorker(self._latest_tag, force=force)
        self._pull_worker.line.connect(self.txt_notes.append)
        self._pull_worker.finished.connect(self._on_pull_finished)
        self._pull_worker.start()

    def _on_pull_finished(self, success):
        if success:
            self.lbl_status.setText('Update complete — restart to apply the new version')
            self.lbl_status.setStyleSheet(status_style(SUCCESS, bold=True, size=FONT_BODY))
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle('Update complete')
            box.setText('DashPVA has been updated. Restart now to apply the new version?')
            restart_btn = box.addButton('Restart DashPVA', QMessageBox.AcceptRole)
            restart_btn.setProperty('role', 'info')
            box.addButton('Later', QMessageBox.RejectRole)
            box.exec_()
            if box.clickedButton() is restart_btn:
                self._restart()
        else:
            self.lbl_status.setText('Update failed — see output above')
            self.lbl_status.setStyleSheet(status_style(ERROR, bold=True, size=FONT_BODY))

    def _restart(self):
        """Relaunch the launcher with the updated code, then quit this instance.
        The Close button remains for users who prefer to restart later."""
        subprocess.Popen(
            [sys.executable, '-m', 'dashpva.viewer.launcher.launcher'],
            cwd=str(settings.PROJECT_ROOT),
            start_new_session=True,
        )
        QApplication.instance().quit()
