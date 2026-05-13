"""
Single source of truth for DashPVA UI colors and font sizes.

Static styles live in theme.qss (loaded by configure_app).
Dynamic styles (runtime state changes) import constants from here.
When changing a value, update theme.qss to match.
"""

# -- Status colors ------------------------------------------------------------
SUCCESS = "#27AE60"
SUCCESS_HOVER = "#2ECC71"
ERROR = "#E74C3C"
ERROR_HOVER = "#C0392B"
WARNING = "#E67E22"
WARNING_HOVER = "#D35400"
INFO = "#2980B9"
INFO_HOVER = "#3498DB"

# -- Text colors ---------------------------------------------------------------
TEXT_PRIMARY = "#2C3E50"
TEXT_SECONDARY = "#7A8394"
TEXT_MUTED = "#9BA5B5"

# -- Surface / border ----------------------------------------------------------
SURFACE = "#F8F9FA"
SURFACE_ALT = "#E9ECEF"
BORDER = "#DEE2E6"

# -- Font sizes ----------------------------------------------------------------
FONT_HEADING = "22px"
FONT_SUBHEADING = "14px"
FONT_BODY = "12px"
FONT_CAPTION = "11px"
FONT_SMALL = "9pt"


def status_style(color: str, *, bold: bool = False,
                 size: str = FONT_CAPTION) -> str:
    weight = " font-weight: 600;" if bold else ""
    return f"font-size: {size}; color: {color};{weight}"
