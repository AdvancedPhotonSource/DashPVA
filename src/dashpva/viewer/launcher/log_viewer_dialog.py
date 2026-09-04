# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

import sys
from html import escape
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QApplication, QDialog

from dashpva.gui import configure_app, ui_path
from dashpva.gui.theme_colors import (
    LOG_DEBUG,
    LOG_DEFAULT,
    LOG_ERROR,
    LOG_INFO,
    LOG_WARNING,
)

_UI_PATH = ui_path("log_viewer.ui")
_MAX_LINES = 1000
#: How often the open dialog checks the log for new lines. A poll is one stat
#: call; the file is only re-read when its size actually changed.
_POLL_INTERVAL_MS = 1000

_LEVEL_COLORS = {
    'ERROR':   LOG_ERROR,
    'WARNING': LOG_WARNING,
    'WARN':    LOG_WARNING,
    'DEBUG':   LOG_DEBUG,
    'INFO':    LOG_INFO,
}


def _resolve_log_file() -> str:
    try:
        import dashpva.settings as app_settings
        log_path = app_settings.LOG_PATH or './logs'
    except Exception:
        log_path = './logs'
    return str(Path(log_path).expanduser() / 'general.log')


def _line_color(line: str) -> str:
    for level, color in _LEVEL_COLORS.items():
        if f' {level} ' in line or line.startswith(level):
            return color
    return LOG_DEFAULT


def _lines_to_html(lines: list) -> str:
    parts = ['<pre style="margin:0;padding:0;">']
    for line in lines:
        color = _line_color(line)
        parts.append(f'<span style="color:{color};">{escape(line.rstrip())}</span>')
    parts.append('</pre>')
    return '\n'.join(parts)


class LogViewerDialog(QDialog):
    """Tail of the DashPVA log, following new lines while it is open.

    Example:
        dlg = LogViewerDialog()
        dlg.show()          # opens at the newest line and keeps up with it
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(_UI_PATH, self)
        self.log_file = _resolve_log_file()
        self.lbl_log_path.setText(f"Log file: {self.log_file}")
        self.btn_refresh.clicked.connect(self._refresh)
        self.btn_clear.clicked.connect(self._clear_view)
        self.btn_close.clicked.connect(self.close)
        #: Size the view was last drawn from, so a poll can skip an idle file.
        self._last_size = -1
        #: Byte offset Clear was pressed at. Everything before it stays hidden,
        #: so the next line written does not drag the whole tail back in.
        self._clear_offset = 0
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_log)
        self._load_log()

    def showEvent(self, event):
        super().showEvent(event)
        # Deferred: the scrollbar has no final maximum until the document has
        # been laid out, so scrolling inside setHtml lands short of the end.
        # Reload first: a dialog shown again after being hidden would
        # otherwise open on whatever was current when it closed.
        self._load_log()
        QTimer.singleShot(0, self._scroll_to_bottom)
        self._poll_timer.start()

    def hideEvent(self, event):
        self._poll_timer.stop()
        super().hideEvent(event)

    def _clear_view(self):
        """Empty the window. The file on disk is left alone.

        The current end of the file is remembered, so only lines written from
        now on are shown -- otherwise the next poll reloads the tail and every
        cleared line comes straight back. Refresh undoes it.
        """
        try:
            self._clear_offset = Path(self.log_file).stat().st_size
        except OSError:
            self._clear_offset = 0
        self._last_size = self._clear_offset
        self.text_log.clear()

    def _refresh(self):
        """Reload the full tail, including anything hidden by Clear."""
        self._clear_offset = 0
        self._load_log()

    def _at_bottom(self) -> bool:
        """True when the view is scrolled to the end, within a line or so."""
        sb = self.text_log.verticalScrollBar()
        return sb.value() >= sb.maximum() - sb.singleStep()

    def _scroll_to_bottom(self):
        """Put the caret on the last line and show it.

        The scrollbar maximum alone is not enough: a long document is laid out
        lazily, so it can still be growing when this runs and setValue lands
        short of the end.
        """
        self.text_log.moveCursor(QTextCursor.End)
        self.text_log.ensureCursorVisible()
        sb = self.text_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _poll_log(self):
        """Redraw only when the log actually grew, and only if following."""
        if not self.chk_follow.isChecked():
            return
        try:
            size = Path(self.log_file).stat().st_size
        except OSError:
            return
        if size == self._last_size:
            return
        # Reading from a smaller file means it was rotated or truncated, which
        # a tail read handles on its own -- both paths just reload the tail.
        self._load_log(keep_position=not self._at_bottom())

    def _load_log(self, keep_position: bool = False):
        try:
            path = Path(self.log_file)
            if not path.exists():
                self.text_log.setPlainText(f"Log file not found:\n{self.log_file}")
                self._last_size = -1
                return
            size = path.stat().st_size
            if self._clear_offset > size:
                # Rotated or truncated since Clear; there is nothing hidden left.
                self._clear_offset = 0
            with open(path, 'rb') as f:
                # Byte offset, so the read has to be binary -- a text-mode seek
                # only accepts values that came from tell().
                f.seek(self._clear_offset)
                raw = f.read()
            lines = raw.decode('utf-8', errors='replace').splitlines()
            self._last_size = size
            tail = lines[-_MAX_LINES:] if len(lines) > _MAX_LINES else lines
            offset = self.text_log.verticalScrollBar().value()
            self.text_log.setHtml(_lines_to_html(tail))
            if keep_position:
                self.text_log.verticalScrollBar().setValue(offset)
            else:
                self._scroll_to_bottom()
        except Exception as e:
            self.text_log.setPlainText(f"Error reading log:\n{e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    configure_app(app)
    dlg = LogViewerDialog()
    dlg.show()
    sys.exit(app.exec_())
