import subprocess

from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QMessageBox

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
    """Build a large, emphasised warning for the update-blocked dialog, using
    the theme's colours and font sizes."""
    html = (
        f'<div style="font-size:{FONT_HEADING}; font-weight:800; color:{ERROR};">{heading}</div>'
        f'<div style="font-size:{FONT_SUBHEADING}; color:{TEXT_PRIMARY}; margin-top:10px;">{body}</div>'
    )
    if command:
        html += (
            f'<div style="font-family:monospace; font-size:{FONT_SUBHEADING}; font-weight:700; '
            f'color:{INFO}; margin-top:8px;">{command}</div>'
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

    def __init__(self, tag, parent=None):
        super().__init__(parent)
        self.tag = tag

    def run(self):
        # Fetch the release tag and check it out so both code and version match
        # the release, instead of pulling the tip of main.
        commands = [
            ['git', 'fetch', 'origin', 'tag', self.tag],
            ['git', 'checkout', self.tag],
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
        """Return an HTML warning if updating in place is unsafe, else None. Guards
        a developer's work: the tag checkout would move them off a feature branch
        or overwrite uncommitted changes, so block unless on main (or already
        detached from a prior update) with a clean tree."""
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
                               f'The update was stopped to be safe.<br>{exc}')
        if branch not in ('main', 'HEAD'):
            return _block_html(
                '⚠  YOU HAVE CHANGED THE BRANCH',
                f"You are on <b>{branch}</b>, not <b>main</b>. Updating checks out the "
                f"release tag and <b>could destroy the work on your branch.</b>"
                f"<br><br>Switch to main, then click Update again:",
                'git checkout main',
            )
        if dirty:
            return _block_html(
                '⚠  YOU HAVE CHANGES THAT CAN BE OVERWRITTEN',
                "Uncommitted changes in the repository <b>could be overwritten</b> "
                "by the update.<br><br>Store your changes and click Update again:",
                'git stash push -m "my changes"',
            )
        return None

    def _start_pull(self):
        if not self._latest_tag:
            return
        reason = self._preflight_block_reason()
        if reason is not None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle('Update blocked')
            msg.setTextFormat(Qt.RichText)
            msg.setText(reason)
            msg.exec_()
            return
        self.btn_update.setEnabled(False)
        self.lbl_status.setText(f'Updating to {self._latest_tag}…')
        self.lbl_status.setStyleSheet(status_style(INFO, size=FONT_BODY))
        self.lbl_notes_header.setText('Update log')
        self.txt_notes.clear()
        for w in (self.line_notes, self.lbl_notes_header, self.txt_notes):
            w.setVisible(True)
        self._pull_worker = PullWorker(self._latest_tag)
        self._pull_worker.line.connect(self.txt_notes.append)
        self._pull_worker.finished.connect(self._on_pull_finished)
        self._pull_worker.start()

    def _on_pull_finished(self, success):
        if success:
            self.lbl_status.setText('Update complete — restart DashPVA to apply')
            self.lbl_status.setStyleSheet(status_style(SUCCESS, bold=True, size=FONT_BODY))
        else:
            self.lbl_status.setText('Update failed — see output above')
            self.lbl_status.setStyleSheet(status_style(ERROR, bold=True, size=FONT_BODY))
