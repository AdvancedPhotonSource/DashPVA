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
    from string import Template

    from dashpva.gui import theme_colors

    qss_file = _GUI_DIR / "theme.qss"
    if qss_file.is_file():
        qss = Template(qss_file.read_text(encoding="utf-8"))
        app.setStyleSheet(qss.safe_substitute(vars(theme_colors)))
