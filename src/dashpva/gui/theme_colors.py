"""
Single source of truth for DashPVA UI colors and font sizes.

Static styles live in theme.qss, which is loaded by configure_app() and
templated against this module via $VARIABLE substitution.  Dynamic styles
(runtime state changes) import constants from here directly.
"""

# -- Status colors ------------------------------------------------------------
SUCCESS = "#27AE60"
SUCCESS_HOVER = "#2ECC71"
SUCCESS_PRESSED = "#1E8449"
ERROR = "#E74C3C"
ERROR_HOVER = "#C0392B"
ERROR_PRESSED = "#A93226"
WARNING = "#E67E22"
WARNING_HOVER = "#D35400"
INFO = "#2980B9"
INFO_HOVER = "#3498DB"

# -- Text colors ---------------------------------------------------------------
TEXT_PRIMARY = "#2C3E50"
TEXT_SECONDARY = "#7A8394"
TEXT_MUTED = "#9BA5B5"

# -- Disabled state ------------------------------------------------------------
DISABLED_BG = "#9E9E9E"

# -- Surface / border ----------------------------------------------------------
SURFACE = "#F8F9FA"
SURFACE_ALT = "#E9ECEF"
BORDER = "#DEE2E6"
SIM_BG = "#D6EAF8"

# -- Dock title bar (slate header + white text; mirrored in theme.qss) ---------
DOCK_HEADER_BG = "#2C3E50"
DOCK_HEADER_TEXT = "#FFFFFF"

# -- ROI / Stats colors --------------------------------------------------------
ROI_COLORS = ['#E05A5A', '#4A90D9', '#3EAF8E', '#A878D0']
ROI_STATS_COLORS = {
    'Stats1': ROI_COLORS[0],
    'Stats2': ROI_COLORS[1],
    'Stats3': ROI_COLORS[2],
    'Stats4': ROI_COLORS[3],
}
# Individual aliases for $VARIABLE substitution in theme.qss
ROI_1 = ROI_COLORS[0]
ROI_2 = ROI_COLORS[1]
ROI_3 = ROI_COLORS[2]
ROI_4 = ROI_COLORS[3]

# -- Lock banner (ROI calculator locked state) ---------------------------------
LOCK_BG = "#FFD6D6"
LOCK_BORDER = "#A94442"

# -- Log level colors (reuse status palette for consistency) -------------------
LOG_ERROR = ERROR
LOG_WARNING = WARNING
LOG_DEBUG = "#8E7CC3"
LOG_INFO = TEXT_SECONDARY
LOG_DEFAULT = TEXT_PRIMARY

# -- Font sizes (use pt to avoid QFont warnings on macOS high-DPI) -------------
FONT_HEADING = "17pt"
FONT_SUBHEADING = "11pt"
FONT_BODY = "10pt"
FONT_CAPTION = "9pt"
FONT_SMALL = "8pt"


def status_style(color: str, *, bold: bool = False,
                 size: str = FONT_CAPTION) -> str:
    weight = " font-weight: 600;" if bold else ""
    return f"font-size: {size}; color: {color};{weight}"
