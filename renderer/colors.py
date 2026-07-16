"""
error_type -> highlight color.

the actual color mapping (ERROR_COLORS) lives in config/constants.py -
model.schemas.ErrorSpan.highlight_color already reads from there for the
frontend's CSS. this module is the PDF-drawing side: converting those hex
strings into the (r, g, b) float tuples reportlab expects.
"""

from config.constants import ERROR_COLORS

DEFAULT_COLOR = "#CCCCCC"


def get_hex(error_type: str) -> str:
    return ERROR_COLORS.get(error_type, DEFAULT_COLOR)


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """'#RRGGBB' -> (r, g, b) floats in [0, 1] - what reportlab's setFillColorRGB expects."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b)


def get_rgb(error_type: str) -> tuple[float, float, float]:
    return hex_to_rgb01(get_hex(error_type))