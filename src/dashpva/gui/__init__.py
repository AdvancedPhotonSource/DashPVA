from pathlib import Path

_GUI_DIR = Path(__file__).resolve().parent


def ui_path(*parts: str) -> str:
    """Return the absolute path to a .ui file in the gui/ package directory."""
    return str(_GUI_DIR.joinpath(*parts))


def configure_app(app):
    """Apply the global theme stylesheet to a QApplication.

    Forces the Fusion style so the QSS rules in theme.qss (including
    ``QDockWidget::title``) apply consistently. Native Linux styles such as
    Breeze/GTK render dock titles in native code and silently ignore most
    stylesheet rules, which is why the dock header coloring never appeared
    on those systems.
    """
    try:
        from PyQt5.QtWidgets import QStyleFactory
        app.setStyle(QStyleFactory.create("Fusion"))
    except Exception:
        pass
    qss_file = _GUI_DIR / "theme.qss"
    if qss_file.is_file():
        app.setStyleSheet(qss_file.read_text(encoding="utf-8"))
