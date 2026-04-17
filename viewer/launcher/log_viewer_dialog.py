import os
import sys
from pathlib import Path
from html import escape
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QApplication

_UI_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'gui', 'log_viewer.ui')
_MAX_LINES = 1000

_LEVEL_COLORS = {
    'ERROR':   '#cc0000',
    'WARNING': '#e06c00',
    'WARN':    '#e06c00',
    'DEBUG':   '#6666aa',
    'INFO':    '#888888',
}


def _resolve_log_file() -> str:
    try:
        import settings as app_settings
        log_path = app_settings.LOG_PATH or './logs'
    except Exception:
        log_path = './logs'
    return str(Path(log_path).expanduser() / 'general.log')


def _line_color(line: str) -> str:
    for level, color in _LEVEL_COLORS.items():
        if f' {level} ' in line or line.startswith(level):
            return color
    return '#222222'


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
            sb = self.text_log.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception as e:
            self.text_log.setPlainText(f"Error reading log:\n{e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlg = LogViewerDialog()
    dlg.show()
    sys.exit(app.exec_())
