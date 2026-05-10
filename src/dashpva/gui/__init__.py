import sys
from pathlib import Path

_GUI_DIR = Path(__file__).resolve().parent


def ui_path(*parts: str) -> str:
    """Return the absolute path to a .ui file in the gui/ package directory."""
    return str(_GUI_DIR.joinpath(*parts))


def configure_app(app):
    """Apply platform-specific styling to a QApplication.

    On macOS, Qt5's native style can render QMessageBox and QLabel text
    with poor contrast in dark mode.  This applies a minimal stylesheet
    fix that forces text to follow the system palette.  No-op on Linux.
    """
    if sys.platform == "darwin":
        app.setStyleSheet(
            "QMessageBox QLabel { color: palette(text); }"
        )
