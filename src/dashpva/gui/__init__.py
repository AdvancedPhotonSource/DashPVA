from pathlib import Path

_GUI_DIR = Path(__file__).resolve().parent


def ui_path(*parts: str) -> str:
    """Return the absolute path to a .ui file in the gui/ package directory."""
    return str(_GUI_DIR.joinpath(*parts))
