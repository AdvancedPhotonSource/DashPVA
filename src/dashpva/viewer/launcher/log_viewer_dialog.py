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
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(_UI_PATH, self)
        self.log_file = _resolve_log_file()
        self.lbl_log_path.setText(f"Log file: {self.log_file}")
        self.btn_refresh.clicked.connect(self._load_log)
        self.btn_close.clicked.connect(self.close)
        self._load_log()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        sb = self.text_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _load_log(self):
        try:
            path = Path(self.log_file)
            if not path.exists():
                self.text_log.setPlainText(f"Log file not found:\n{self.log_file}")
                return
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            tail = lines[-_MAX_LINES:] if len(lines) > _MAX_LINES else lines
            self.text_log.setHtml(_lines_to_html(tail))
            self._scroll_to_bottom()
        except Exception as e:
            self.text_log.setPlainText(f"Error reading log:\n{e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    configure_app(app)
    dlg = LogViewerDialog()
    dlg.show()
    sys.exit(app.exec_())
