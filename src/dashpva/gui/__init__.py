from pathlib import Path

_GUI_DIR = Path(__file__).resolve().parent


def ui_path(*parts: str) -> str:
    """Return the absolute path to a .ui file in the gui/ package directory."""
    return str(_GUI_DIR.joinpath(*parts))


def configure_app(app):
    """Apply the global theme stylesheet to a QApplication.

    Colors come from ``theme_colors`` via ``$NAME`` substitution so the stylesheet
    keeps a single source of truth (e.g. ``background-color: $SUCCESS;``). Unknown
    ``$NAME`` tokens are left untouched (``safe_substitute``).
    """
    import os
    from string import Template

    from PyQt5.QtWidgets import QStyleFactory

    from dashpva.gui import theme_colors

    module_label = os.environ.get('DASHPVA_MODULE_LABEL')
    if module_label:
        app.setApplicationName(f'DashPVA — {module_label}')
        app.setApplicationDisplayName(f'DashPVA — {module_label}')

    app.setStyle(QStyleFactory.create("Fusion"))

    qss_file = _GUI_DIR / "theme.qss"
    if qss_file.is_file():
        qss = Template(qss_file.read_text(encoding="utf-8"))
        app.setStyleSheet(qss.safe_substitute(vars(theme_colors)))
